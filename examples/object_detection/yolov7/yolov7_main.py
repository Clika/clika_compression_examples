import argparse
import sys
import warnings
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from clika_compression import Settings, TensorBoardCallback, clika_compress, clika_resume
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "yolov7"))
from models.common import RepConv
from models.yolo import Detect, Model
from utils.datasets import InfiniteDataLoader, LoadImagesAndLabels
from utils.general import colorstr, labels_to_class_weights, non_max_suppression, xywhn2xyxy
from utils.loss import ComputeLossOTA

DATA_YAML = str(BASE_DIR.joinpath("yolov7", "data", "coco.yaml"))
HYP_YAML = str(BASE_DIR.joinpath("yolov7", "data", "hyp.scratch.p5.yaml"))


# Define Class/Function Wrappers
# ==================================================================================================================== #
class YoloV7(Model):
    """
    Currently CLIKA SDK can only trace models with tensor inputs.
    Original YoloV7 model take 3 input arguments (x, augment, profile)
    (augment, profile) are constant variables.  If there are Constants they need to be set with some value.
    We're just removing those because they're not important.

    Override original model to hide augment, profile from the SDK
    https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/models/yolo.py#L581
    """

    def __init__(self, *attrs, **kwargs):
        super().__init__(*attrs, **kwargs)

    def forward(self, x):
        return super().forward(x, False, False)


class DetectWrapper(Detect):
    def __init__(self):
        pass

    def forward(self, x):
        """
        Detect is the last component of YoloV7 graph
        Since we are focusing on the fine-tunable graph ignore `self.end2end`, `self.include_nms` and `self.concat`
        Assume `self.training = True`

        REFERENCE:
        https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/models/yolo.py#L42
        """
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        return x


def get_optimizer(model: nn.Module, lr: float, momentum: float, weight_decay: float):
    """Optimizer used inside WongKinYiu/yolov7::train.py

    REFERENCE:
    https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/train.py#L115-L188
    """
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for _, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optimizer = torch.optim.SGD(pg0, lr=lr, momentum=momentum, nesterov=True)

    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    return optimizer


class DatasetWrapper(LoadImagesAndLabels):
    def __init__(self, *attrs, **kwargs):
        self.train = kwargs.pop("train")
        super().__init__(*attrs, **kwargs)

    def __getitem__(self, item):
        imgs, targets, _, shapes = super().__getitem__(item)
        imgs = imgs.to(torch.float32) / 255

        if self.train is True:
            return imgs, targets
        else:
            return imgs, (targets, shapes)


def _collate_fn(batch):
    """Parse dataloader output to satisfy CLIKA SDK input requirements.
    (https://docs.clika.io/docs/compression-constraints/cco_input_requirements#Dataset-Dataloader)

    REFERENCE:
    https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/utils/datasets.py#L632
    """
    sizes = None
    img, label = zip(*batch)  # transposed
    if isinstance(label[0], tuple):
        sizes = [_[1][0] for _ in label]
        label = [_[0] for _ in label]
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()

    if sizes is None:  # train
        return torch.stack(img, 0), torch.cat(label, 0)
    else:  # eval
        return torch.stack(img, 0), (torch.cat(label, 0), [img[0].shape for _ in range(len(img))])


def get_loader(
    data_dir: str,
    imgsz: int,
    batch_size: int,
    stride: int,
    hyp=None,
    rect=False,
    augment=False,
    pad=0.0,
    workers=4,
    prefix="",
    train=True,
) -> DataLoader:
    """Factory function that generates train/eval Dataloader.

    REFERENCE:
    https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/utils/datasets.py#L65
    """
    dataset = DatasetWrapper(
        data_dir,
        imgsz,
        batch_size,
        augment=augment,  # augment images
        hyp=hyp,  # augmentation hyperparameters
        rect=rect,  # rectangular training
        cache_images=False,
        single_cls=False,
        stride=stride,
        pad=pad,
        image_weights=False,
        prefix=prefix,
        train=train,
    )

    batch_size = min(batch_size, len(dataset))

    dataloader = InfiniteDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=None,
        pin_memory=False,
        collate_fn=_collate_fn,
    )
    return dataloader


class CriterionWrapper(object):
    """Wrapper around the `ComputeLossOTA` to satisfy CLIKA SDK Loss restriction.
    (https://docs.clika.io/docs/compression-constraints/cco_input_requirements#Loss-Function)
    """

    def __init__(self, model: nn.Module, img_size: int, batch_size: int):
        self.fake_obj = [np.zeros([3, img_size]) for _ in range(batch_size)]
        self.loss_fn = ComputeLossOTA(model.cuda(), False)  # model.cuda() to move anchors inside Detect module to cuda

    def __call__(self, p, targets):
        # Originally, `ComputeLossOTA.__call__` takes 3 arguments (p, targets, imgs)
        # https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/utils/loss.py#L582
        #
        # Wrap __call__ function to take 2 inputs (`predictions` and `targets`) instead of 3
        # `ComputeLossOTA(..., imgs=...)` argument is only used one time to return height of an image
        # https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/utils/loss.py#L662
        loss, loss_items = self.loss_fn(p, targets, self.fake_obj)
        return loss


class MetricWrapper(MeanAveragePrecision):
    """MeanAveragePrecision class wrapper that postprocess model's logit outputs and compute mAP"""

    def __init__(
        self,
        stride: Tensor,
        conf_thresh: float,
        nms_thresh: float,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # REFERENCE:
        # https://github.com/WongKinYiu/yolov7/blob/a207844b1ce82d204ab36d87d496728d3d2348e7/cfg/deploy/yolov7.yaml#L6
        anchors = torch.tensor(
            [
                [12, 16, 19, 36, 40, 28],
                [36, 75, 76, 55, 72, 146],
                [142, 110, 192, 243, 459, 401],
            ],
            dtype=torch.float32,
        )

        self.add_state("anchors", default=anchors, persistent=False)
        self.add_state("strides", default=stride, persistent=False)
        self.conf_thresh: float = conf_thresh
        self.nms_thresh: float = nms_thresh

    def update(self, outputs: List[Tensor], targets: Tuple[Tensor, tuple]):
        # `Detect` when .training=False
        # REFERENCE:
        # https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/models/yolo.py#L52-L63
        _device = outputs[0].device
        targets, img_sizes = targets
        targets = targets.to(_device)

        # cells to bboxes (cell-wise info --> img-wise info)
        anchor_grid = self.anchors.clone().view(len(outputs), 1, -1, 1, 1, 2).to(_device)
        grid = [torch.zeros(1, device=_device)] * len(outputs)  # init grid
        z = []
        for i, output in enumerate(outputs):
            (
                bs,
                num_anchors,
                ny,
                nx,
                out_channels,
            ) = output.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

            if grid[i].shape[2:4] != output.shape[2:4]:
                yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
                grid[i] = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().to(_device)

            y = output.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + grid[i]) * self.strides[i].to(_device)  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1, out_channels))
        detections = torch.cat(z, dim=1)

        # non max suppression
        # REFERENCE:
        # https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/utils/general.py#L608
        detections = non_max_suppression(
            detections,
            multi_label=False,
            conf_thres=self.conf_thresh,
            iou_thres=self.nms_thresh,
        )
        for i, d in enumerate(detections):
            cur_img_shape: tuple = img_sizes[i]
            _ts = targets[targets[:, 0] == i]  # filter targets
            _ts[:, 2:] = xywhn2xyxy(_ts[:, 2:], h=cur_img_shape[1], w=cur_img_shape[2], padh=0, padw=0)
            super().update(
                [
                    {
                        "labels": d[:, -1].ravel().long(),
                        "scores": d[:, -2].ravel(),
                        "boxes": d[:, 0:4],  # xyxy
                    }
                ],
                [{"labels": _ts[:, 1].ravel().long(), "boxes": _ts[:, 2:]}],
            )

    def compute(self) -> dict:
        results: dict = super().compute()
        results.pop("classes", None)
        return results


# ==================================================================================================================== #


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="CLIKA YOLOv7 Example")

    ace_parser = parser.add_argument_group("CLIKA ACE configuration")
    ace_parser.add_argument("--config", type=str, default="config.yml", help="ACE config yaml path")

    model_parser = parser.add_argument_group("Model configuration")
    model_parser.add_argument("--output_dir", type=Path, default="outputs", help="Path to save clika related files for the SDK")
    model_parser.add_argument("--data", type=Path, default="coco", help="Dataset directory")
    model_parser.add_argument("--ckpt", type=Path, default="yolov7.pt", help="Path to load the model checkpoints (e.g. .pth, .pompom)")
    model_parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation (default: 16)")
    model_parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    model_parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for data loading (default: 4)")
    model_parser = parser.add_argument_group("Additional arguments")
    model_parser.add_argument("--image_size", type=int, default=640, help="Image size used for both training & evaluation (default: 640)")
    model_parser.add_argument("--nms_threshold", type=float, default=0.65, help="NMS threshold to evaluate mAP (default: 0.65)")
    model_parser.add_argument("--confidence_threshold", type=float, default=0.001, help="Confidence threshold to evaluate mAP (default: 0.001)")

    args = parser.parse_args()

    if args.data.exists() is False:
        raise FileNotFoundError(f"Unknown directory: {args.data}")
    # fmt: on

    return args


def main():
    global DATA_YAML, HYP_YAML

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
    with open(DATA_YAML) as fp:
        data_dict = yaml.load(fp, Loader=yaml.SafeLoader)
    class_names = data_dict["names"]  # class names (e.g) 80 cls
    num_classes = int(data_dict["nc"])

    with open(HYP_YAML) as fp:
        hyp = yaml.load(fp, Loader=yaml.SafeLoader)

    model = YoloV7(
        str(BASE_DIR.joinpath("yolov7", "cfg", "deploy", "yolov7.yaml")),
        ch=3,
        nc=num_classes,
        anchors=None,
    )

    _resume = False
    if (_train_from_scratch is False) and (args.ckpt is not None):
        if args.ckpt.suffix == ".pompom":
            print("[INFO] .pompom file provided as ckpt, resuming compression ...")
            _resume = True
        else:
            print(f"[INFO] loading custom ckpt from {args.ckpt}")
            ckpt = torch.load(str(args.ckpt))  # load checkpoint
            model = YoloV7(ckpt["model"].yaml, ch=3, nc=num_classes, anchors=None)
            state_dict = ckpt["model"].float().state_dict()  # FP16 to FP32
            model.load_state_dict(state_dict, strict=True)

    # replace detect module
    _old_detect = model.model[-1]
    _new_detect = DetectWrapper()
    _new_detect.__dict__ = _old_detect.__dict__
    model.model[-1] = _new_detect
    for m in model.modules():
        if isinstance(m, RepConv):
            m.fuse_repvgg_block()
    torch.fx.symbolic_trace(model)

    """
    Define Optimizer
    ====================================================================================================================
    """
    _nominal_batch_size = 64
    _accumulate = max(round(_nominal_batch_size / args.batch_size), 1)

    # scale weight_decay according to batch_size
    hyp["weight_decay"] *= args.batch_size * _accumulate / _nominal_batch_size
    optimizer = get_optimizer(model, args.lr, momentum=hyp["momentum"], weight_decay=hyp["weight_decay"])

    """
    Define Dataloaders
    ====================================================================================================================
    """
    train_path = str(args.data.joinpath("train2017.txt"))  # data_dict["train"]
    test_path = str(args.data.joinpath("val2017.txt"))  # data_dict["val"]
    grid_size = max(int(model.stride.max()), 32)  # grid size = max stride
    detection_heads = model.model[-1].nl  # number of detection_heads

    assert args.image_size % grid_size == 0, f"input image_size must be divisible by stride {grid_size}"

    get_train_loader = partial(
        get_loader,
        data_dir=train_path,
        imgsz=args.image_size,
        batch_size=args.batch_size,
        stride=grid_size,
        hyp=hyp,
        augment=True,
        workers=args.workers,
        prefix=colorstr("train: "),
        train=True,
    )
    get_eval_loader = partial(
        get_loader,
        data_dir=test_path,
        imgsz=args.image_size,
        batch_size=args.batch_size * 2,
        stride=grid_size,
        hyp=hyp,
        rect=True,
        workers=args.workers,
        pad=0.5,
        prefix=colorstr("val: "),
        train=False,
    )

    """
    ETC
    ====================================================================================================================
    """
    hyp["box"] *= 3.0 / detection_heads  # scale to layers
    hyp["cls"] *= num_classes / 80.0 * 3.0 / detection_heads  # scale to classes and layers
    hyp["obj"] *= (args.image_size / 640) ** 2 * 3.0 / detection_heads  # scale to image size and layers
    hyp["label_smoothing"] = 0
    model.nc = num_classes  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)

    # attach class weights
    _labels = get_train_loader().dataset.labels
    model.class_weights = labels_to_class_weights(_labels, num_classes).to("cuda") * num_classes
    model.names = class_names

    """
    Define Loss Function
    ====================================================================================================================
    """
    train_losses = {"loss_sum": CriterionWrapper(model, img_size=args.image_size, batch_size=args.batch_size)}
    eval_losses = None

    """
    Define Metric Wrapper
    ====================================================================================================================
    """
    train_metrics = None
    eval_metrics = MetricWrapper(
        stride=model.stride,
        conf_thresh=args.confidence_threshold,
        nms_thresh=args.nms_threshold,
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
