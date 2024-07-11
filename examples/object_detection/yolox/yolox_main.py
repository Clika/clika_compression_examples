import argparse
import sys
import warnings
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np
import torch
from clika_compression import Settings, TensorBoardCallback, clika_compress, clika_resume
from torch.utils.data import Dataset
from torchmetrics.detection import MeanAveragePrecision

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "YOLOX"))
from yolox.data import COCODataset
from yolox.data.data_augment import preproc
from yolox.exp.build import get_exp_by_name
from yolox.models import YOLOX, YOLOXHead
from yolox.utils import postprocess

# Define Class/Function Wrappers
# ==================================================================================================================== #


class YOLOXWrapper(YOLOX):
    def __init__(self, backbone, head):
        super().__init__(backbone, head)

    def forward(self, x):
        fpn_outs = self.backbone(x)
        outputs = self.head(fpn_outs)

        return outputs


class YOLOXHeadWrapper(YOLOXHead):
    def __init__(self):
        pass

    def forward(self, x):
        """YOLOXHead.forward returns a loss instead of logits when .training=True
        Discard loss function and return pure logits.
        """
        outputs = []

        # REFERENCE:
        # https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/models/yolo_head.py#L149-L161
        for k, (cls_conv, reg_conv, _x) in enumerate(zip(self.cls_convs, self.reg_convs, x)):
            _x = self.stems[k](_x)
            cls_x = _x
            reg_x = _x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # ignore sigmoid gradients
            # REFERENCE:
            # https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/models/yolo_head.py#L187-L191
            _ = obj_output.sigmoid()
            _obj_output = _.detach() - obj_output.detach() + obj_output
            _ = cls_output.sigmoid()
            _cls_output = _.detach() - cls_output.detach() + cls_output

            output = torch.cat([reg_output, _obj_output, _cls_output], 1)
            outputs.append(output)

        # REFERENCE:
        # https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/models/yolo_head.py#L205C3-L209
        outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)

        return outputs


class CriterionWrapper:
    """Separated loss function from YOLOXHead.forward that satisfies CLIKA SDK Loss restriction.
    (https://docs.clika.io/docs/compression-constraints/cco_input_requirements#Loss-Function)
    """

    def __init__(self, hw=((80, 80), (40, 40), (20, 20))):
        self.head = YOLOXHead(num_classes=80)
        for n in ["cls_convs", "reg_convs", "cls_preds", "reg_preds", "obj_preds", "stems"]:
            setattr(self.head, n, None)  # empty forward graph
        self.head.use_l1 = True
        self.hw = hw

    def __call__(self, p, targets):
        targets = targets.to(p.device)

        # undo flatten, permute, concat
        # REFERENCE:
        # https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/models/yolo_head.py#L205-L209
        p = torch.permute(p, (0, 2, 1))
        N, _, C = p.shape
        p_1, p_2, p_3 = torch.split(p, [hw[0] * hw[1] for hw in self.hw], dim=-1)
        p_1 = p_1.reshape(N, -1, self.hw[0][0], self.hw[0][1])
        p_2 = p_2.reshape(N, -1, self.hw[1][0], self.hw[1][1])
        p_3 = p_3.reshape(N, -1, self.hw[2][0], self.hw[2][1])

        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        origin_preds = []

        for k, (stride_this_level, _p) in enumerate(zip(self.head.strides, [p_1, p_2, p_3])):
            # undo cat + undo sigmoid(skip gradients)
            # REFERENCE:
            # https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/models/yolo_head.py#L187-L189
            reg_output, obj_output, cls_output = _p.split([4, 1, 80], dim=1)

            _ = torch.log(obj_output + 1e-7) - torch.log(1 - obj_output)
            obj_output = _.detach() - obj_output.detach() + obj_output
            _ = torch.log(cls_output + 1e-7) - torch.log(1 - cls_output)
            cls_output = _.detach() - cls_output.detach() + cls_output

            # REFERENCE:
            # https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/models/yolo_head.py#L164
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.head.get_output_and_grid(output, k, stride_this_level, p[0].type())
            x_shifts.append(grid[:, :, 0])  # repeat of shifts (e.g. (0~79) * 80)
            y_shifts.append(grid[:, :, 1])  # repeat of shifts (e.g. (0, 0, 0, ... 0, 1, 1, 1, ... 1, ...) * 80)
            expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(p[0][0]))
            outputs.append(output)

            # =========== L1 loss =========== #
            batch_size = reg_output.shape[0]
            hsize, wsize = reg_output.shape[-2:]
            reg_output = reg_output.view(batch_size, 1, 4, hsize, wsize)
            reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
            origin_preds.append(reg_output.clone())
            # =========== L1 loss =========== #

        # REFERENCE:
        # https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/models/yolo_head.py#L194C25-L194C35
        loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head.get_losses(
            imgs=None,
            x_shifts=x_shifts,
            y_shifts=y_shifts,
            expanded_strides=expanded_strides,
            labels=targets,
            outputs=torch.cat(outputs, 1),
            origin_preds=origin_preds,
            dtype=p.dtype,
        )

        loss_dict = {"iou_loss": iou_loss, "conf_loss": conf_loss, "cls_loss": cls_loss, "l1_loss": l1_loss}
        return loss_dict


def _collate_fn_train(batch):
    """Parse dataloader output to satisfy CLIKA SDK input requirements.
    (https://docs.clika.io/docs/compression-constraints/cco_input_requirements#Dataset-Dataloader)
    """
    inputs, targets, _img_infos, img_ids = zip(*batch)  # transposed
    inputs = torch.from_numpy(np.stack(inputs, 0))
    target = torch.from_numpy(np.stack(targets, 0))
    return inputs, target


def get_train_loader_(exp, batch_size):
    """Factory function that generates train/eval Dataloader.

    REFERENCE:
    https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/exp/yolox_base.py#L155
    """
    loader = exp.get_data_loader(
        batch_size=batch_size,
        is_distributed=False,
        no_aug=True,  # turn off mosaic
        cache_img=None,
    )
    exp.dataset = None
    loader.collate_fn = _collate_fn_train
    return loader


def _collate_fn_eval(batch):
    """Parse dataloader output to satisfy CLIKA SDK input requirements.
    (https://docs.clika.io/docs/compression-constraints/cco_input_requirements#Dataset-Dataloader)
    """
    inputs, targets, _img_infos, img_ids = zip(*batch)  # transposed
    inputs = torch.from_numpy(np.stack(inputs, 0))
    target = list(map(torch.from_numpy, targets))
    img_shapes = [tuple(_img.shape[1:3]) for _img in inputs]
    return inputs, (target, img_shapes, _img_infos, img_ids)


def _val_preproc(img, target, input_dim):
    img, r = preproc(img, input_dim)
    return img, target * r


def get_eval_loader_(data_dir, batch_size, workers, image_size):
    """Factory function that generates eval Dataloader.

    REFERENCE:
    https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/exp/yolox_base.py#L312
    """
    dataset = COCODataset(
        data_dir=data_dir,
        json_file="instances_val2017.json",
        name="val2017",
        img_size=image_size,  # DEFAULT
        preproc=_val_preproc,
    )
    sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": False,
        "sampler": sampler,
        "collate_fn": _collate_fn_eval,
    }
    loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    return loader


class MetricWrapper(MeanAveragePrecision):
    """MeanAveragePrecision class wrapper that postprocess model's logit outputs and compute mAP"""

    def __init__(self, strides: tuple, nms_thresh: float, conf_thresh: float, hw: list, **kwargs: Any):
        super().__init__(**kwargs)
        self.strides = strides
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.class_agnostic = False
        self.num_classes = 80
        self.head = YOLOXHead(num_classes=self.num_classes)
        for n in ["cls_convs", "reg_convs", "cls_preds", "reg_preds", "obj_preds", "stems"]:
            setattr(self.head, n, None)  # empty forward graph
        self.head.hw = hw

    def update(self, outputs: torch.Tensor, targets: tuple):
        _device = outputs.device

        labels, img_shapes, img_info, img_id = targets
        labels = [l.to(_device) for l in labels]

        # REFERENCE:
        # https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/models/yolo_head.py#L210-L211
        decoded_outputs = self.head.decode_outputs(outputs.cpu(), outputs.dtype).to(_device)

        # convert output from cxcywh -> xyxy format
        detections = postprocess(
            decoded_outputs,
            num_classes=self.num_classes,
            conf_thre=self.conf_thresh,
            nms_thre=self.nms_thresh,
            class_agnostic=self.class_agnostic,
        )

        for p, t in zip(detections, labels):
            if p is None:
                continue
            box, obj_conf, class_conf, class_label = p.split([4, 1, 1, 1], 1)
            super().update(
                [
                    {
                        "labels": class_label.ravel().long(),
                        "scores": (obj_conf * class_conf).ravel(),
                        "boxes": box,  # xyxy
                    }
                ],
                [{"labels": t[:, -1].ravel().long(), "boxes": t[:, :4]}],
            )

    def compute(self) -> dict:
        results: dict = super().compute()
        results.pop("classes", None)
        return results


# ==================================================================================================================== #


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="CLIKA YOLOX Example")

    ace_parser = parser.add_argument_group("CLIKA ACE configuration")
    ace_parser.add_argument("--config", type=str, default="config.yml", help="ACE config yaml path")

    model_parser = parser.add_argument_group("Model configuration")
    model_parser.add_argument("--output_dir", type=Path, default="outputs", help="Path to save clika related files for the SDK")
    model_parser.add_argument("--data", type=Path, default="COCO", help="Dataset directory")
    model_parser.add_argument("--ckpt", type=Path, default="yolox_s.pth", help="Path to load the model checkpoints (e.g. .pth, .pompom)")
    model_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation (default: 32)")
    model_parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    model_parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for data loading (default: 1)")
    model_parser = parser.add_argument_group("Additional arguments")
    model_parser.add_argument("--image_size", nargs="+", type=int, default=(640, 640), help="Image size used for both training & evaluation (default: (640, 640))")
    model_parser.add_argument("--nms_threshold", type=float, default=0.65, help="NMS threshold to evaluate mAP (default: 0.65)")
    model_parser.add_argument("--confidence_threshold", type=float, default=0.01, help="Confidence threshold to evaluate mAP (default: 0.01)")

    args = parser.parse_args()

    if args.data.exists() is False:
        raise FileNotFoundError(f"Unknown directory: {args.data}")
    assert len(args.image_size) == 2
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
    exp = get_exp_by_name("yolox-s")  # TODO(@JLEE): model_type as argparse?
    exp.input_size = args.image_size

    model = exp.get_model()
    model = YOLOXWrapper(model.backbone, model.head)

    # replace head
    _old_head = model.head
    _new_head = YOLOXHeadWrapper()
    _new_head.__dict__ = _old_head.__dict__
    model.head = _new_head

    _resume = False
    _optimizer_state_dict = None
    if (_train_from_scratch is False) and (args.ckpt is not None):
        if args.ckpt.suffix == ".pompom":
            print("[INFO] .pompom file provided as ckpt, resuming compression ...")
            _resume = False
        else:
            print(f"[INFO] loading custom ckpt from {args.ckpt}")
            ckpt = torch.load(str(args.ckpt))
            model.load_state_dict(ckpt["model"])

            _optimizer_state_dict = ckpt.get("optimizer", None)
    torch.fx.symbolic_trace(model)

    """
    Define Loss Function
    ====================================================================================================================
    """
    model.head.use_l1 = True  # add additional L1 loss
    train_losses = {"total": CriterionWrapper()}
    eval_losses = None

    """
    Define Optimizer
    ====================================================================================================================
    """
    optimizer = exp.get_optimizer(args.batch_size)
    if _optimizer_state_dict:  # if ckpt provided
        print("[INFO] using optimizer state dict from the provided checkpoint (`--lr` cli argument is ignored)")
        optimizer.load_state_dict(ckpt["optimizer"])

    # overwrite the lr with the value that was set by the user
    optimizer.defaults["lr"] = args.lr
    for group in optimizer.param_groups:
        group["lr"] = args.lr

    """
    Define Dataloaders
    ====================================================================================================================
    """
    # TODO(@JLEE): augment settings per model_type
    exp.data_dir = args.data
    exp.shear = 0
    exp.degrees = 0
    exp.hsv_prob = 0
    exp.translate = 0
    exp.flip_prob = 0
    exp.mixup_prob = 0
    exp.mosaic_prob = 0
    exp.enable_mixup = False
    exp.multiscale_range = 0
    exp.input_size = args.image_size
    exp.data_num_workers = args.workers

    get_train_loader = partial(get_train_loader_, exp=exp, batch_size=args.batch_size)
    get_eval_loader = partial(
        get_eval_loader_,
        data_dir=args.data,
        batch_size=args.batch_size,
        workers=args.workers,
        image_size=args.image_size,
    )

    """
    Define Metric Wrapper
    ====================================================================================================================
    """
    _strides = model.head.strides
    _hw = [(exp.input_size[0] // s, exp.input_size[1] // s) for s in _strides]
    eval_metrics = MetricWrapper(
        strides=_strides, nms_thresh=args.nms_threshold, conf_thresh=args.confidence_threshold, hw=_hw
    )
    eval_metrics = {"mAP": eval_metrics}
    train_metrics = None

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
