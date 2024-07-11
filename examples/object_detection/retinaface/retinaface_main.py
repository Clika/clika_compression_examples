import argparse
import math
import sys
import warnings
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
from clika_compression import Settings, TensorBoardCallback, clika_compress, clika_resume
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "Pytorch_Retinaface"))
from data import WiderFaceDetection, cfg_mnet, cfg_re50, preproc
from layers.functions.prior_box import PriorBox
from layers.modules import MultiBoxLoss
from models.retinaface import RetinaFace
from utils.box_utils import decode


def load_checkpoints(ckpt, model) -> torch.nn.Module:
    """
    Load model from checkpoint

    REFERENCE:
    https://github.com/biubug6/Pytorch_Retinaface/blob/b984b4b775b2c4dced95c1eadd195a5c7d32a60b/train.py#L59-L71
    """
    state_dict = torch.load(ckpt)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        print(
            f"[INFO] ckpt keys do not match. trying `load_checkpoints`(https://github.com/biubug6/Pytorch_Retinaface/blob/b984b4b775b2c4dced95c1eadd195a5c7d32a60b/train.py#L59-L71)"
        )
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == "module.":
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    return model


# Define Class/Function Wrappers
# ==================================================================================================================== #


class CriterionWrapper(MultiBoxLoss):
    """Wrapper around the `MultiBoxLoss` to satisfy CLIKA SDK Loss restriction.
    (https://docs.clika.io/docs/compression-constraints/cco_input_requirements#Loss-Function)

    REFERENCE:
    https://github.com/biubug6/Pytorch_Retinaface/blob/b984b4b775b2c4dced95c1eadd195a5c7d32a60b/train.py#L82-L87
    """

    def __init__(self, *attrs, priors):
        self.priors = priors
        super().__init__(*attrs)

    def forward(self, predictions, targets) -> dict:
        targets, img_shapes = targets
        self.priors = self.priors.to(targets[0].device)
        loss_l, loss_c, loss_landm = super().forward(predictions, self.priors, targets)

        # Making the loss function conform with CLIKA Compression constrains by returning a dict
        # This way we can see the name of each loss during the logs of the training process
        return {"loss_l": loss_l, "loss_c": loss_c, "loss_landm": loss_landm}


def _detection_collate(batch: list) -> Tuple[Tensor, Tuple[List[Tensor], List[Tuple]]]:
    """custom collate function for dynamic input shapes

    REFERENCE:
    https://github.com/biubug6/Pytorch_Retinaface/blob/b984b4b775b2c4dced95c1eadd195a5c7d32a60b/data/wider_face.py#L79-L101
    """
    imgs = []
    targets = []
    img_shapes = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
                _, height, width = tup.size()
                img_shapes.append((height, width))
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return torch.stack(imgs, 0), (targets, img_shapes)


def get_loader(
    data_dir: Path, batch_size: int, workers: int, image_size: Optional[int] = None, train=False
) -> DataLoader:
    """Factory function that generates train/eval Dataloader."""
    if train:
        data_path = str(data_dir.joinpath("train", "label.txt"))
        dataset = WiderFaceDetection(data_path, preproc(image_size, rgb_means=(104, 117, 123)))  # BGR order
        shuffle = True
    else:
        data_path = str(data_dir.joinpath("val", "label.txt"))
        dataset = WiderFaceEvalDataset(data_path, size=image_size)
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=_detection_collate,
        pin_memory=False,
    )
    return loader


class MetricWrapper(MeanAveragePrecision):
    """MeanAveragePrecision class wrapper that postprocess model's logit outputs and compute mAP"""

    def __init__(
        self,
        priorbox: PriorBox,
        prior_cache: Dict[tuple, Tensor],
        variances: list,
        nms_thresh: float,
        conf_thresh: float,
        eval_origin_img: bool = False,
        **kwargs: Any,
    ):
        """
        RetinaFace metric wrapper.

        Args:
            priorbox: PriorBox object initialized from training cfg e.g. `cfg_mnet`, `cfg_re50`
            prior_cache: dictionary of image_shape: PriorBox output
            variances: variances registered inside training cfg e.g. `cfg_mnet`, `cfg_re50`
            nms_thresh: non-max-suppression(IoU) threshold for measuring mAP
            conf_thresh: confidence threshold for measuring mAP
            eval_origin_img: if set True evaluate on dynamic image sizes else follow training image size
            **kwargs:
        """
        super().__init__(**kwargs)
        self.variances = variances
        self.nms_thresh: float = nms_thresh
        self.conf_thresh: float = conf_thresh

        self.priorbox = priorbox
        self.eval_origin_img = eval_origin_img
        self.prior_cache = prior_cache

    def _decode_outputs(
        self, batch_bbox_regressions: Tensor, batch_classifications: Tensor, img_shapes: List
    ) -> List[Tensor]:
        """Post-process model's logit output
        Return nms filtered output bboxes & scores based on `self.nms_thresh` and `self.conf_thresh` hyperparameters.

        REFERENCE:
        https://github.com/biubug6/Pytorch_Retinaface/blob/b984b4b775b2c4dced95c1eadd195a5c7d32a60b/test_widerface.py#L130-L170
        """
        _device = batch_bbox_regressions.device
        BATCH_SIZE: int = batch_bbox_regressions.shape[0]

        batch_scores = torch.softmax(batch_classifications, -1)[..., 1][..., None]
        batch_inds = batch_scores >= self.conf_thresh
        filtered_batch_scores = [batch_scores[i][batch_inds[i]] for i in range(BATCH_SIZE)]
        detections: list = []
        for i in range(BATCH_SIZE):
            scores = filtered_batch_scores[i]
            if len(scores) == 0:
                detections.append(None)
                continue
            inds = batch_inds[i]
            bbox_regressions = batch_bbox_regressions[i]

            if self.eval_origin_img and img_shapes[i] not in self.prior_cache:
                self.priorbox.image_size = img_shapes[i]
                self.priorbox.feature_maps = [
                    [math.ceil(self.priorbox.image_size[0] / step), math.ceil(self.priorbox.image_size[1] / step)]
                    for step in self.priorbox.steps
                ]
                _priors = self.prior_cache.get(img_shapes[i], self.priorbox.forward())
                self.prior_cache.setdefault(img_shapes[i], _priors)
            else:
                _priors = self.prior_cache[img_shapes[i]]

            _priors = _priors.to(_device)[inds.ravel()]
            bbox_regressions = bbox_regressions[inds.ravel()]
            bboxes = decode(loc=bbox_regressions, priors=_priors, variances=self.variances)
            img_sizes = torch.tensor(img_shapes[i])
            img_sizes = torch.flip(img_sizes, (-1,))
            img_sizes = torch.cat((img_sizes, img_sizes), dim=-1)
            bboxes = bboxes * img_sizes.to(_device)

            order = torch.argsort(scores, descending=True)
            bboxes = bboxes[order]
            scores = scores[order]

            dets = torch.hstack((bboxes, scores[:, None])).to(torch.float32)
            keep = torchvision.ops.nms(bboxes, scores, iou_threshold=self.nms_thresh)
            dets = dets[keep, :]
            detections.append(dets)
        return detections

    def update(self, output: List[Tensor], targets: Tuple[Tensor, tuple]):
        if len(output) == 2:
            bbox_regressions, classifications = output
        else:
            bbox_regressions, classifications, _ = output
        _device = bbox_regressions.device
        targets, img_shapes = targets
        targets = [target.to(_device) for target in targets]

        detections = self._decode_outputs(bbox_regressions, classifications, img_shapes)
        for i in range(len(detections)):
            if detections[i] is None:
                continue
            pred_boxes, pred_score = torch.split(detections[i], [4, 1], -1)
            pred_labels = torch.ones(pred_boxes.shape[0]).to(_device)  # 0 for heads 1 for non-heads
            pred_boxes.clamp_(min=0)

            targets_boxes = targets[i] * torch.tensor(
                [
                    img_shapes[i][1],
                    img_shapes[i][0],
                    img_shapes[i][1],
                    img_shapes[i][0],
                ],
                device=_device,
            )
            target_labels = torch.ones(targets_boxes.shape[0]).to(_device)
            # calculating MeanAveragePrecision
            super().update(
                [
                    {
                        "labels": pred_labels.ravel().long(),
                        "scores": pred_score.ravel(),
                        "boxes": pred_boxes,  # xyxy
                    }
                ],
                [{"labels": target_labels.ravel().long(), "boxes": targets_boxes}],
            )

    def compute(self) -> dict:
        results: dict = super().compute()
        results.pop("classes", None)
        return results


class WiderFaceEvalDataset(WiderFaceDetection):
    """Custom eval loader
    No eval dataloader defined inside `biubug6/Pytorch_Retinaface` repository.

    REFERENCE
    https://github.com/biubug6/Pytorch_Retinaface/blob/b984b4b775b2c4dced95c1eadd195a5c7d32a60b/data/wider_face.py#L9
    """

    def __init__(self, txt_path: str, size: Optional[int] = None):
        super().__init__(txt_path, None)
        self.size = size

        self.pad_offset = 32
        self.bgr_mean = np.array([104, 117, 123], dtype="float32")

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        # preprocess without augmentation
        # https://github.com/biubug6/Pytorch_Retinaface/blob/b984b4b775b2c4dced95c1eadd195a5c7d32a60b/data/data_augment.py#L203-L206
        if self.size is None:
            _ = height % self.pad_offset
            hb = 0 if _ == 0 else self.pad_offset - _
            _ = width % self.pad_offset
            wb = 0 if _ == 0 else self.pad_offset - _
            if hb != 0 or wb != 0:
                img = cv2.copyMakeBorder(
                    img,
                    hb // 2,
                    hb - hb // 2,
                    wb // 2,
                    wb - wb // 2,
                    borderType=cv2.BORDER_CONSTANT,
                    value=self.bgr_mean.tolist(),
                )
        else:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        img = img.astype("float32") - self.bgr_mean
        img = img.transpose(2, 0, 1)  # HWC --> CHW

        labels = self.words[index]
        annotations = np.zeros((0, 4))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 4))
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        target[:, 0::2] /= width
        target[:, 1::2] /= height

        return torch.from_numpy(img), target


# ==================================================================================================================== #


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="CLIKA RetinaFace Example")

    ace_parser = parser.add_argument_group("CLIKA ACE configuration")
    ace_parser.add_argument("--config", type=str, default="config.yml", help="ACE config yaml path")

    model_parser = parser.add_argument_group("Model configuration")
    model_parser.add_argument("--output_dir", type=Path, default="outputs", help="Path to save clika related files for the SDK")
    model_parser.add_argument("--data", type=Path, default="widerface", help="Dataset directory")
    model_parser.add_argument("--ckpt", type=Path, default="Resnet50_Final.pth", help="Path to load the model checkpoints (e.g. .pth, .pompom)")
    model_parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation (default: 8)")
    model_parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    model_parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for data loading (default: 4)")
    model_parser = parser.add_argument_group("Additional arguments")
    model_parser.add_argument("--model_type", type=str, choices=["resnet50", "mobilenetv1"], default="resnet50", help="RetinaFace model backbone (default: resnet50)")
    model_parser.add_argument("--nms_threshold", type=float, default=0.4, help="NMS threshold to evaluate mAP (default: 0.4)")
    model_parser.add_argument("--confidence_threshold", type=float, default=0.02, help="Confidence threshold to evaluate mAP (default: 0.02)")
    model_parser.add_argument("--eval_origin_img", action='store_true', help="Use origin image size to evaluate (Evaluation Loader batch size will be fixed to 1)")
    args = parser.parse_args()

    if args.data.exists() is False:
        raise FileNotFoundError(f"Unknown directory: {args.data}")
    # fmt: on

    return args


def main():
    args = parse_args()
    settings = Settings.load_from_path(args.config)

    pprint(args)
    pprint(settings)
    _train_from_scratch = settings.training_settings.is_training_from_scratch
    if (_train_from_scratch is True) and (args.ckpt is not None):
        warnings.warn(
            f"Conflicting arguments `train_from_scratch={_train_from_scratch}` and `ckpt={args.ckpt}` "
            f"=> Enable `train_from_scratch` and ignore ckpt"
        )

    """
    Define Model
    ====================================================================================================================
    """
    if args.model_type == "mobilenetv1":
        cfg = cfg_mnet
    elif args.model_type == "resnet50":
        cfg = cfg_re50
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    image_size = cfg["image_size"]
    image_shape = (image_size, image_size)
    model = RetinaFace(cfg=cfg)

    _resume = False
    if (_train_from_scratch is False) and (args.ckpt is not None):
        if args.ckpt.suffix == ".pompom":
            print("[INFO] .pompom file provided as ckpt, resuming compression ...")
            _resume = True
        else:
            print(f"[INFO] loading custom ckpt from {args.ckpt}")
            model = load_checkpoints(str(args.ckpt), model)
    torch.fx.symbolic_trace(model)

    """
    Define Loss Function
    ====================================================================================================================
    """
    priorbox = PriorBox(cfg, image_shape)
    _priors = priorbox.forward()
    _num_classes = 2  # [face, not_face]
    train_losses = CriterionWrapper(_num_classes, 0.35, True, 0, True, 7, 0.35, False, priors=_priors)

    train_losses = {"loss_sum": train_losses}
    eval_losses = None

    """
    Define Optimizer
    ====================================================================================================================
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    """
    Define Dataloaders
    ====================================================================================================================
    """
    get_train_loader = partial(
        get_loader,
        data_dir=args.data,
        batch_size=args.batch_size,
        workers=args.workers,
        image_size=image_size,
        train=True,
    )
    get_eval_loader = partial(
        get_loader,
        data_dir=args.data,
        batch_size=1 if args.eval_origin_img else args.batch_size,
        workers=args.workers,
        image_size=None if args.eval_origin_img else image_size,
        train=False,
    )

    """
    Define Metric Wrapper
    ====================================================================================================================
    """
    train_metrics = None
    eval_metrics = MetricWrapper(
        priorbox=priorbox,
        prior_cache={image_shape: _priors},
        variances=cfg["variance"],
        nms_thresh=args.nms_threshold,
        conf_thresh=args.confidence_threshold,
        eval_origin_img=args.eval_origin_img,
    )
    eval_metrics = {"mAP": eval_metrics}

    """
    RUN Compression
    ====================================================================================================================
    """
    if _resume is True:
        clika_resume(
            clika_state_path=args.ckpt,
            init_training_dataset_fn=get_train_loader,
            init_evaluation_dataset_fn=get_eval_loader,
            optimizer=optimizer,
            training_losses=train_losses,
            training_metrics=train_metrics,
            evaluation_losses=eval_losses,
            evaluation_metrics=eval_metrics,
            callbacks=[TensorBoardCallback(output_path=args.output_dir)],
            new_settings=settings,
            resume_start_epoch=None,
        )
    else:
        clika_compress(
            output_path=args.output_dir,
            settings=settings,
            model=model,
            init_training_dataset_fn=get_train_loader,
            init_evaluation_dataset_fn=get_eval_loader,
            optimizer=optimizer,
            training_losses=train_losses,
            training_metrics=train_metrics,
            evaluation_losses=eval_losses,
            evaluation_metrics=eval_metrics,
            callbacks=[TensorBoardCallback(output_path=args.output_dir)],
        )


if __name__ == "__main__":
    main()
