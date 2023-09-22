import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from typing import Optional, List, Any, Dict, Callable, Union
import types
import argparse
from functools import partial
import warnings
from pathlib import Path
import yaml

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics.detection import MeanAveragePrecision

from clika_compression import PyTorchCompressionEngine, QATQuantizationSettings, DeploymentSettings_TensorRT_ONNX, \
    DeploymentSettings_TFLite
from clika_compression.settings import (
    generate_default_settings, LayerQuantizationSettings, ModelCompileSettings
)

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "yolov7"))
from models.yolo import Model, Detect
from utils.datasets import LoadImagesAndLabels, InfiniteDataLoader
from utils.loss import ComputeLossOTA
from utils.general import colorstr, labels_to_class_weights, non_max_suppression, xywhn2xyxy

ANCHORS = torch.tensor([
    [12, 16, 19, 36, 40, 28],
    [36, 75, 76, 55, 72, 146],
    [142, 110, 192, 243, 459, 401]
], dtype=torch.float32)
STRIDES = torch.tensor([8, 16, 32], dtype=torch.float32)

DATA_YAML = str(BASE_DIR.joinpath("yolov7", "data", "coco.yaml"))
HYP_YAML = str(BASE_DIR.joinpath("yolov7", "data", "hyp.scratch.p5.yaml"))
IMG_SIZE = 640
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.65

COMPDTYPE = Union[Dict[str, Union[Callable, torch.nn.Module]], None]

DEPLOYMENT_DICT = {
    "trt": DeploymentSettings_TensorRT_ONNX(graph_author="CLIKA",
                                            graph_description=None,
                                            input_shapes_for_deployment=[(None, 3, None, None)]),

    "tflite": DeploymentSettings_TFLite(graph_author="CLIKA",
                                        graph_description=None,
                                        input_shapes_for_deployment=[(None, 3, None, None)]),
}


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


def _detect_forward_WRAPPER(self, x):
    """
    Detect is the last component of YoloV7 graph
    Since we are focusing on the fine-tunable graph ignore `self.end2end`, `self.include_nms` and `self.concat`
    Assume `self.training = True`
    https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/models/yolo.py#L42
    """
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
        bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
    return x


def replace_DETECT(model):
    """
    Loop through the model components and replace `Detect` module's forward method
    """
    for n, m in model.named_children():
        if isinstance(m, Detect):
            m.forward = types.MethodType(_detect_forward_WRAPPER, m)
        else:
            replace_DETECT(m)


def get_optimizer(model: nn.Module,
                  lr: float,
                  momentum: float,
                  weight_decay: float):
    """
    Optimizer used from train.py
    https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/train.py#L115-L188
    """
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, "im"):
            if hasattr(v.im, "implicit"):
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, "imc"):
            if hasattr(v.imc, "implicit"):
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, "imb"):
            if hasattr(v.imb, "implicit"):
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, "imo"):
            if hasattr(v.imo, "implicit"):
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, "ia"):
            if hasattr(v.ia, "implicit"):
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, "attn"):
            if hasattr(v.attn, "logit_scale"):
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, "q_bias"):
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, "v_bias"):
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, "relative_position_bias_table"):
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, "rbr_dense"):
            if hasattr(v.rbr_dense, "weight_rbr_origin"):
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, "weight_rbr_avg_conv"):
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, "weight_rbr_pfir_conv"):
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, "weight_rbr_1x1_kxk_idconv1"):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, "weight_rbr_1x1_kxk_conv2"):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, "weight_rbr_gconv_dw"):
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, "weight_rbr_gconv_pw"):
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, "vector"):
                pg0.append(v.rbr_dense.vector)

    optimizer = torch.optim.SGD(pg0, lr=lr, momentum=momentum, nesterov=True)

    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    return optimizer


class DATASET_WRAPPER(LoadImagesAndLabels):
    """
    Due to CLIKA SDK restriction, Every DataLoader should return tuple of length 2
    Wrap original Dataset class to return tuple of length 2 instead of tuple of length 4
    https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/utils/datasets.py#L662
    """

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
    sizes = None
    img, label = zip(*batch)  # transposed
    if isinstance(label[0], tuple):
        sizes = [_[1][0] for _ in label]
        label = [_[0] for _ in label]
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()

    if sizes is None:
        return torch.stack(img, 0), torch.cat(label, 0)
    else:
        return torch.stack(img, 0), (
            torch.cat(label, 0), [img[0].shape for _ in range(len(img))])


def get_loader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
               workers=8, image_weights=False, prefix="", train=True) -> torch.utils.data.DataLoader:
    """
    Return train/eval Dataloader
    https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/utils/datasets.py#L65
    """
    dataset = DATASET_WRAPPER(path, imgsz, batch_size,
                              augment=augment,  # augment images
                              hyp=hyp,  # augmentation hyperparameters
                              rect=rect,  # rectangular training
                              cache_images=cache,
                              single_cls=opt.single_cls,
                              stride=int(stride),
                              pad=pad,
                              image_weights=image_weights,
                              prefix=prefix,
                              train=train)

    batch_size = min(batch_size, len(dataset))
    sampler = None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader

    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=workers,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=_collate_fn)
    return dataloader


class CRITERION_WRAPPER(object):
    """
    CLIKA SDK restriction forces 2 arguments (model_output, label) for a loss function.
    Originally, `ComputeLossOTA.__call__` takes 3 arguments (p, targets, imgs)
    https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/utils/loss.py#L582

    Wrap __call__ function to take 2 inputs (`predictions` and `targets`) instead of 3
    inside `ComputeLossOTA` `imgs` argument is only used one time to return height of an image
    https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/utils/loss.py#L662
    """

    def __init__(self, model, config):
        self.fake_obj = [np.zeros([3, IMG_SIZE]) for _ in range(config.batch_size)]
        self.loss_fn = ComputeLossOTA(model, False)

    def __call__(self, p, targets):
        loss, loss_items = self.loss_fn(p, targets, self.fake_obj)
        return loss


class MeanAveragePrecisionWrapper(MeanAveragePrecision):
    """
    A custom metric class that inherits from a torchmetrics object.
    We use this class to preform postprocessing to the model's outputs before calculating the MeanAveragePrecision
    """

    def __init__(self,
                 anchors: Tensor,
                 strides: Tensor,
                 conf_thres: float,
                 iou_thres: float,
                 box_format: str = "xyxy",
                 iou_type: str = "bbox",
                 iou_thresholds: Optional[List[float]] = None,
                 rec_thresholds: Optional[List[float]] = None,
                 max_detection_thresholds: Optional[List[int]] = None,
                 class_metrics: bool = False,
                 **kwargs: Any):
        super().__init__(box_format, iou_type, iou_thresholds, rec_thresholds, max_detection_thresholds, class_metrics,
                         **kwargs)
        self.add_state("anchors", default=anchors, persistent=False)
        self.add_state("strides", default=strides, persistent=False)
        self.conf_thres: float = conf_thres
        self.iou_thres: float = iou_thres

    def update(self, outputs: List[Tensor], targets: Tensor):
        """
        detections = model output, [(N, num_anchors, S, S, num_classes + 5),] * 3
        targets = target label, (idx, cls, cxcywh)

        https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/models/yolo.py#L52-L63
        """
        targets, img_sizes = targets
        targets = targets.to(outputs[0].device)

        # cells to bboxes (cell-wise info --> img-wise info)
        _device = outputs[0].device
        anchor_grid = self.anchors.clone().view(len(outputs), 1, -1, 1, 1, 2)
        grid = [torch.zeros(1, device=_device)] * len(outputs)  # init grid
        z = []
        for i, output in enumerate(outputs):
            bs, num_anchors, ny, nx, out_channels = output.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

            if grid[i].shape[2:4] != output.shape[2:4]:
                yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
                grid[i] = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().to(output.device)

            y = output.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * self.strides[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1, out_channels))
        detections = torch.cat(z, dim=1)

        # non max suppression
        # https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/utils/general.py#L608
        detections = non_max_suppression(detections,
                                         multi_label=False,
                                         conf_thres=self.conf_thres,
                                         iou_thres=self.iou_thres)
        for i, d in enumerate(detections):
            cur_img_shape: tuple = img_sizes[i]
            _ts = targets[targets[:, 0] == i]  # filter targets
            _ts[:, 2:] = xywhn2xyxy(_ts[:, 2:], h=cur_img_shape[1], w=cur_img_shape[2], padh=0, padw=0)
            super().update(
                [{
                    "labels": d[:, -1].ravel().long(),
                    "scores": d[:, -2].ravel(),
                    "boxes": d[:, 0:4],  # xyxy
                }],
                [{
                    "labels": _ts[:, 1].ravel().long(),
                    "boxes": _ts[:, 2:]
                }])

    def compute(self) -> dict:
        results: dict = super().compute()
        results.pop("classes", None)
        return results


# ==================================================================================================================== #


def resume_compression(
        config: argparse.Namespace,
        get_train_loader: Callable,
        get_eval_loader: Callable,
        train_losses: COMPDTYPE,
        train_metrics: COMPDTYPE,
        eval_losses: COMPDTYPE = None,
        eval_metrics: COMPDTYPE = None
):
    engine = PyTorchCompressionEngine()

    mcs = ModelCompileSettings(
        optimizer=None,
        training_losses=train_losses,
        training_metrics=train_metrics,
        evaluation_losses=eval_losses,
        evaluation_metrics=eval_metrics,
    )
    final = engine.resume(
        clika_state_path=config.ckpt,
        model_compile_settings=mcs,
        init_training_dataset_fn=get_train_loader,
        init_evaluation_dataset_fn=get_eval_loader,
        settings=None
    )
    engine.deploy(
        clika_state_path=final,
        output_dir_path=config.output_dir,
        file_suffix=None,
        input_shapes=None,
        graph_author="CLIKA",
        graph_description="Demo - Created by clika.io",
    )


def run_compression(
        config: argparse.Namespace,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        get_train_loader: Callable,
        get_eval_loader: Callable,
        train_losses: COMPDTYPE,
        train_metrics: COMPDTYPE,
        eval_losses: COMPDTYPE = None,
        eval_metrics: COMPDTYPE = None
):
    global DEPLOYMENT_DICT

    engine = PyTorchCompressionEngine()
    settings = generate_default_settings()

    settings.deployment_settings = DEPLOYMENT_DICT[config.target_framework]
    settings.global_quantization_settings = QATQuantizationSettings()
    settings.global_quantization_settings.weights_num_bits = config.weights_num_bits
    settings.global_quantization_settings.activations_num_bits = config.activations_num_bits

    settings.training_settings.num_epochs = config.epochs
    settings.training_settings.steps_per_epoch = config.steps_per_epoch
    settings.training_settings.evaluation_steps = config.evaluation_steps
    settings.training_settings.stats_steps = config.stats_steps
    settings.training_settings.print_interval = config.print_interval
    settings.training_settings.print_num_steps_averaging = config.ma_window_size
    settings.training_settings.save_interval = config.save_interval
    settings.training_settings.reset_training_dataset_between_epochs = config.reset_train_data
    settings.training_settings.reset_evaluation_dataset_between_epochs = config.reset_eval_data
    settings.training_settings.grads_accumulation_steps = config.grads_acc_steps
    settings.training_settings.mixed_precision = config.mixed_precision
    settings.training_settings.lr_warmup_epochs = config.lr_warmup_epochs
    settings.training_settings.lr_warmup_steps_per_epoch = config.lr_warmup_steps_per_epoch
    settings.training_settings.use_fp16_weights = config.fp16_weights
    settings.training_settings.use_gradients_checkpoint = config.gradients_checkpoint

    # Skip quantization for last layers
    layer_names_to_skip = {
        "reshape", "permute", "conv_92", "shape",
        "reshape_1", "permute_1", "conv_93", "shape_1",
        "reshape_2", "permute_2", "conv_94", "shape_2",
    }
    for x in layer_names_to_skip:
        settings.set_quantization_settings_for_layer(x, LayerQuantizationSettings(skip_quantization=True))

    mcs = ModelCompileSettings(
        optimizer=optimizer,
        training_losses=train_losses,
        training_metrics=train_metrics,
        evaluation_losses=eval_losses,
        evaluation_metrics=eval_metrics,
    )
    final = engine.optimize(
        output_path=config.output_dir,
        settings=settings,
        model=model,
        model_compile_settings=mcs,
        init_training_dataset_fn=get_train_loader,
        init_evaluation_dataset_fn=get_eval_loader,
        is_training_from_scratch=config.train_from_scratch

    )
    engine.deploy(
        clika_state_path=final,
        output_dir_path=config.output_dir,
        file_suffix=None,
        input_shapes=None,
        graph_author="CLIKA",
        graph_description="Demo - Created by clika.io",
    )


def main(config):
    global BASE_DIR, ANCHORS, STRIDES, DATA_YAML, HYP_YAML, IMG_SIZE, CONF_THRESHOLD, IOU_THRESHOLD

    print("\n".join(f"{k}={v}" for k, v in vars(config).items()))  # pretty print argparse

    config.data = config.data if os.path.isabs(config.data) else str(BASE_DIR / config.data)
    if os.path.exists(config.data) is False:
        raise FileNotFoundError("Could not find default dataset please check `--data`")

    config.output_dir = config.output_dir if os.path.isabs(config.output_dir) else str(BASE_DIR / config.output_dir)

    """
    Define Model
    ====================================================================================================================
    """
    # data_dict from "data/coco.yaml"
    with open(DATA_YAML) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    _names = data_dict["names"]  # class names (e.g) 80 cls
    _num_classes = int(data_dict["nc"])

    # hyp from "data/hyp.scratch.p5.yaml"
    with open(HYP_YAML) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    model = YoloV7(
        str(BASE_DIR.joinpath("yolov7", "cfg", "deploy", "yolov7.yaml")),
        ch=3,
        nc=_num_classes,
        anchors=None
    ).cuda()

    resume_compression_flag = False
    if (config.train_from_scratch is False) and (config.ckpt is not None):
        config.ckpt = config.ckpt if os.path.isabs(config.ckpt) else str(BASE_DIR / config.ckpt)

        if config.ckpt.rsplit(".", 1)[-1] == "pompom":
            warnings.warn(".pompom file provided, resuming compression (argparse attributes ignored)")
            resume_compression_flag = True
        else:
            resume_compression_flag = False
            print(f"loading ckpt from {config.ckpt}")
            ckpt = torch.load(config.ckpt, map_location="cuda")  # load checkpoint
            model = YoloV7(ckpt["model"].yaml, ch=3, nc=_num_classes, anchors=None).cuda()
            state_dict = ckpt["model"].float().state_dict()  # to FP32
            model.load_state_dict(state_dict, strict=True)

    replace_DETECT(model=model)
    print("Detect module replaced")

    """
    Define Optimizer
    ====================================================================================================================
    """
    _nominal_batch_size = 64
    accumulate = max(round(_nominal_batch_size / config.batch_size), 1)
    hyp["weight_decay"] *= config.batch_size * accumulate / _nominal_batch_size  # scale weight_decay according to batch_size
    optimizer = get_optimizer(model, config.lr, momentum=hyp["momentum"], weight_decay=hyp["weight_decay"])

    """
    Define Dataloaders
    ====================================================================================================================
    """
    train_path = str(Path(config.data).joinpath("train2017.txt"))  # data_dict["train"]
    test_path = str(Path(config.data).joinpath("val2017.txt"))  # data_dict["val"]
    _grid_size = max(int(model.stride.max()), 32)  # grid size = max stride
    _detection_heads = model.model[-1].nl  # number of detection_heads

    get_train_loader = partial(get_loader,
                               path=train_path,
                               imgsz=IMG_SIZE,
                               batch_size=config.batch_size,
                               stride=_grid_size,
                               opt=config,
                               hyp=hyp,
                               augment=True,
                               cache=False,
                               rect=False,
                               workers=config.workers,
                               image_weights=False,
                               prefix=colorstr("train: "),
                               train=True)
    get_eval_loader = partial(get_loader,
                              path=test_path,
                              imgsz=IMG_SIZE,
                              batch_size=config.batch_size * 2,
                              stride=_grid_size,
                              opt=config,
                              hyp=hyp,
                              cache=False,
                              rect=True,
                              workers=config.workers,
                              pad=0.5,
                              prefix=colorstr("val: "),
                              train=False)

    """
    ETC
    ====================================================================================================================
    """
    hyp["box"] *= 3. / _detection_heads  # scale to layers
    hyp["cls"] *= _num_classes / 80. * 3. / _detection_heads  # scale to classes and layers
    hyp["obj"] *= (IMG_SIZE / 640) ** 2 * 3. / _detection_heads  # scale to image size and layers
    hyp["label_smoothing"] = 0
    model.nc = _num_classes  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    _labels = get_train_loader().dataset.labels
    model.class_weights = labels_to_class_weights(_labels, _num_classes).to("cuda") * _num_classes  # attach class weights
    model.names = _names

    """
    Define Loss Function
    ====================================================================================================================
    """
    train_losses = CRITERION_WRAPPER(model, config)
    train_losses = {"loss_sum": train_losses}
    eval_losses = None

    """
    Define Metric Wrapper
    ====================================================================================================================
    """
    train_metrics = None
    eval_metrics = MeanAveragePrecisionWrapper(anchors=torch.clone(ANCHORS),
                                               strides=torch.clone(STRIDES),
                                               conf_thres=CONF_THRESHOLD,
                                               iou_thres=IOU_THRESHOLD)
    eval_metrics = {"mAP": eval_metrics}

    """
    RUN COE
    ====================================================================================================================    
    """
    if resume_compression_flag is True:
        resume_compression(
            config=config,
            get_train_loader=get_train_loader,
            get_eval_loader=get_eval_loader,
            train_losses=train_losses,
            train_metrics=train_metrics,
            eval_losses=eval_losses,
            eval_metrics=eval_metrics
        )
    else:
        run_compression(
            config=config,
            model=model,
            optimizer=optimizer,
            get_train_loader=get_train_loader,
            get_eval_loader=get_eval_loader,
            train_losses=train_losses,
            train_metrics=train_metrics,
            eval_losses=eval_losses,
            eval_metrics=eval_metrics
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIKA YOLOv7 Example")
    parser.add_argument("--target_framework", type=str, default="trt", choices=["tflite", "trt"], help="choose the target framework TensorFlow Lite or TensorRT")
    parser.add_argument("--data", type=str, default="coco", help="Dataset directory")

    # CLIKA Engine Training Settings
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Number of steps per epoch")
    parser.add_argument("--evaluation_steps", type=int, default=None, help="Number of steps for evaluation")
    parser.add_argument("--stats_steps", type=int, default=50, help="Number of steps for scans")
    parser.add_argument("--print_interval", type=int, default=50, help="COE print log interval")
    parser.add_argument("--ma_window_size", type=int, default=20, help="Moving average window size (default: 20)")
    parser.add_argument("--save_interval", action="store_true", default=None, help="Save interval")
    parser.add_argument("--reset_train_data", action="store_true", default=False, help="Reset training dataset between epochs")
    parser.add_argument("--reset_eval_data", action="store_true", default=False, help="Reset evaluation dataset between epochs")
    parser.add_argument("--grads_acc_steps", type=int, default=4, help="gradient accumulation steps")
    parser.add_argument("--no_mixed_precision", action="store_false", default=True, dest="mixed_precision", help="Not using Mixed Precision")
    parser.add_argument("--lr_warmup_epochs", type=int, default=1, help="Learning Rate used in the Learning Rate Warmup stage (default: 1)")
    parser.add_argument("--lr_warmup_steps_per_epoch", type=int, default=500, help="Number of steps per epoch used in the Learning Rate Warmup stage")
    parser.add_argument("--fp16_weights", action="store_true", default=False, help="Use FP16 weight (can reduce VRAM usage)")
    parser.add_argument("--gradients_checkpoint", action="store_true", default=False, help="Use gradient checkpointing")

    # Model Training Setting
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model (default: 10)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for data loading (default: 4)")
    parser.add_argument("--ckpt", type=str, default="yolov7.pt", help="Path to load the model checkpoints (e.g. .pth, .pompom)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for saving results and checkpoints (default: outputs)")
    parser.add_argument("--train_from_scratch", action="store_true", help="Train the model from scratch")

    # Quantization Config
    parser.add_argument("--weights_num_bits", type=int, default=8, help="How many bits to use for the Weights for Quantization")
    parser.add_argument("--activations_num_bits", type=int, default=8, help="How many bits to use for the Activation for Quantization")

    args = parser.parse_args()
    args.single_cls = False  # required for dataloader

    main(config=args)
