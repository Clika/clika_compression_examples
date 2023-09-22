import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from typing import Optional, List, Any, Dict, Callable, Union
import types
import argparse
from functools import partial
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchmetrics.detection import MeanAveragePrecision

from clika_compression import PyTorchCompressionEngine, QATQuantizationSettings, DeploymentSettings_TensorRT_ONNX, \
    DeploymentSettings_TFLite
from clika_compression.settings import (
    generate_default_settings, LayerQuantizationSettings, ModelCompileSettings
)

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "YOLOX"))
from yolox.exp.build import get_exp_by_name
from yolox.models import YOLOXHead
from yolox.data import (
    COCODataset,
    TrainTransform,
    ValTransform,
    YoloBatchSampler,
    DataLoader,
    InfiniteSampler,
    MosaicDetection,
    worker_init_reset_seed,
)
from yolox.utils import postprocess

VAL_ANN = "instances_val2017.json"
MODEL_NAME = "yolox-s"
IMG_SIZE = (640, 640)
CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.65

COMPDTYPE = Union[Dict[str, Union[Callable, torch.nn.Module]], None]

DEPLOYMENT_DICT = {
    "trt": DeploymentSettings_TensorRT_ONNX(graph_author="CLIKA",
                                            graph_description=None,
                                            input_shapes_for_deployment=[(None, 3, None, None)]),
    "tflite": DeploymentSettings_TFLite(graph_author="CLIKA",
                                        graph_description=None,
                                        input_shapes_for_deployment=[(None, 3, None, None)])
}


# Define Class/Function Wrappers
# ==================================================================================================================== #
def _head_forward_WRAPPER(self, x):
    head_outputs = []

    # https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/models/yolo_head.py#L149-L161
    for k, (cls_conv, reg_conv, _x) in enumerate(
            zip(self.cls_convs, self.reg_convs, x)
    ):
        _x = self.stems[k](_x)
        cls_x = _x
        reg_x = _x

        cls_feat = cls_conv(cls_x)
        cls_output = self.cls_preds[k](cls_feat)

        reg_feat = reg_conv(reg_x)
        reg_output = self.reg_preds[k](reg_feat)
        obj_output = self.obj_preds[k](reg_feat)

        # https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/models/yolo_head.py#L187-L191
        head_outputs.extend([reg_output, obj_output, cls_output])

    return head_outputs


def replace_HEAD(model):
    """
    YOLOX.forward() take 3 arguments (xin, labels, imgs)
    labels, imgs are only provided when loss need to be calculated (self.training=True)

    We should not include loss calculation inside the traced graph
    Replace forward call with `_head_forward_WRAPPER`
    """
    for n, m in model.named_children():
        if isinstance(m, YOLOXHead):
            m.forward = types.MethodType(_head_forward_WRAPPER, m)
        else:
            replace_HEAD(m)  # recur


class CRITERION_WRAPPER(object):
    """
    CLIKA SDK restriction forces 2 arguments (model_output, label) for a loss function.
    Isolate loss calculation from `model.forward()`
    https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/models/yolo_head.py#L194-L203
    """

    def __init__(self, head_module):
        self.head_module = head_module

    def __call__(self, p, targets):
        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        origin_preds = []

        p = [p[i * 3:i * 3 + 3] for i in range(3)]

        targets = targets.to(p[0][0].device)

        for k, (cls_conv, reg_conv, stride_this_level, _p) in enumerate(
                zip(self.head_module.cls_convs, self.head_module.reg_convs, self.head_module.strides, p)
        ):
            # concat (bbox[4], conf[1], cls[1])
            reg_output, obj_output, cls_output = _p
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.head_module.get_output_and_grid(
                output, k, stride_this_level, p[0][0].type()
            )
            x_shifts.append(grid[:, :, 0])  # repeat of shifts (e.g. (0~79) * 80)
            y_shifts.append(grid[:, :, 1])  # repeat of shifts (e.g. (0, 0, 0, ... 0, 1, 1, 1, ... 1, ...) * 80)
            expanded_strides.append(
                torch.zeros(1, grid.shape[1])
                .fill_(stride_this_level)
                .type_as(p[0][0])
            )
            outputs.append(output)

            # =========== L1 loss =========== #
            batch_size = reg_output.shape[0]
            hsize, wsize = reg_output.shape[-2:]
            reg_output = reg_output.view(
                batch_size, 1, 4, hsize, wsize
            )
            reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                batch_size, -1, 4
            )
            origin_preds.append(reg_output.clone())
            # =========== L1 loss =========== #

        loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head_module.get_losses(
            imgs=None,
            x_shifts=x_shifts,
            y_shifts=y_shifts,
            expanded_strides=expanded_strides,
            labels=targets,
            outputs=torch.cat(outputs, 1),
            origin_preds=origin_preds,
            dtype=p[0][0].dtype,
        )

        loss_dict = {
            "total_loss": loss,
        }
        return loss_dict


class TRAIN_WRAPPER(MosaicDetection):
    """
    Due to CLIKA SDK restriction, Every DataLoader should return tuple of length 2
    Wrap original Dataset class to return tuple of length 2 instead of tuple of length 4
    https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/data/datasets/mosaicdetection.py#L154
    """

    def __init__(self, *attrs, **kwargs):
        super().__init__(*attrs, **kwargs)

    def __getitem__(self, item):
        inputs, targets, _, _ = super().__getitem__(item)  # img always input_size
        return inputs, targets


def get_train_loader_(exp, config):
    """
    Get train Dataset from `exp` object and return DataLoader
    https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/exp/yolox_base.py#L155
    """
    dataset = exp.get_dataset(cache=False)
    dataset = TRAIN_WRAPPER(
        dataset=dataset,
        mosaic=True,
        img_size=exp.input_size,
        preproc=TrainTransform(
            max_labels=120,
            flip_prob=exp.flip_prob,
            hsv_prob=exp.hsv_prob),
        degrees=exp.degrees,
        translate=exp.translate,
        mosaic_scale=exp.mosaic_scale,
        mixup_scale=exp.mixup_scale,
        shear=exp.shear,
        enable_mixup=exp.enable_mixup,
        mosaic_prob=exp.mosaic_prob,
        mixup_prob=exp.mixup_prob,
    )

    sampler = InfiniteSampler(len(dataset))

    batch_sampler = YoloBatchSampler(
        sampler=sampler,
        batch_size=config.batch_size,
        drop_last=True,
        mosaic=True,
    )

    dataloader_kwargs = {
        "num_workers": config.workers,
        "pin_memory": True,
        "batch_sampler": batch_sampler,
        "worker_init_fn": worker_init_reset_seed
    }
    loader = DataLoader(dataset, **dataloader_kwargs)

    return loader


class TEST_WRAPPER(COCODataset):
    """
    Due to CLIKA SDK restriction, Every DataLoader should return tuple of length 2
    Wrap original Dataset class to return tuple of length 2 instead of tuple of length 4
    https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/data/datasets/coco.py#L188
    """

    def __init__(self, *attrs, **kwargs):
        super().__init__(*attrs, **kwargs)

    def __getitem__(self, item):
        img, target_before, img_info, img_id = self.pull_item(item)
        target_before = torch.from_numpy(target_before)
        target = target_before
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        # img = img * 1 / 255.0
        target = torch.cat((torch.zeros((target_before.shape[0], 1)), target_before), 1)
        return img, (target, img, img_info, img_id)


#
def get_eval_loader_(config):
    """
    Get eval Dataset and return DataLoader
    https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/exp/yolox_base.py#L312
    """

    def collate_fn(batch):
        img, target = zip(*batch)  # transposed
        (target, img, img_info, img_id) = zip(*target)
        for i, l in enumerate(target):
            l[:, 0] = i  # add target image index for build_targets()
        imgs = torch.from_numpy(np.stack(img, 0))
        target = torch.cat(target, 0)
        img_shapes = tuple([
            tuple(imgs[i].size()) for i in range(len(imgs))
        ])
        return imgs, (target, img_shapes, img_info, img_id)

    valdataset = TEST_WRAPPER(
        data_dir=config.data,
        json_file=VAL_ANN,
        name="val2017",
        img_size=IMG_SIZE,
        preproc=ValTransform(legacy=False),
    )

    sampler = torch.utils.data.SequentialSampler(valdataset)

    dataloader_kwargs = {
        "num_workers": config.workers,
        "pin_memory": True,
        "sampler": sampler,
        "batch_size": config.batch_size,
        "collate_fn": collate_fn
    }
    loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    return loader


class MeanAveragePrecisionWrapper(MeanAveragePrecision):
    """
    A custom metric class that inherits from a torchmetrics object.
    We use this class to preform postprocessing to the model's outputs before calculating the MeanAveragePrecision
    """

    def __init__(self,
                 strides: tuple,
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
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.strides = strides
        self.num_classes = 80
        self.class_agnostic = False

    def _decode_outputs(self, outputs, heads_hw):
        """
        cellwise info to imagewise info
        https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/models/yolo_head.py#L215
        """
        _device = outputs.device
        grids = []
        _strides = []
        for (hsize, wsize), stride in zip(heads_hw, self.strides):
            yv, xv = torch.meshgrid(torch.arange(hsize), torch.arange(wsize), indexing="ij")
            yv = yv.to(_device)
            xv = xv.to(_device)
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            _strides.append(
                torch.full((*shape, 1), stride, device=_device))  # (1, 80*80, 1), (1, 40*40, 1), (1, 20*20, 1)

        grids = torch.cat(grids, dim=1)  # (1, 8400, 2)
        _strides = torch.cat(_strides, dim=1)  # (1, 8400, 1)

        outputs[..., :2] = (outputs[..., :2] + grids) * _strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * _strides
        return outputs

    def update(self, outputs: List[torch.Tensor], targets: torch.Tensor):
        """
        Because some parts of HEAD.forward() has been skipped by `replace_HEAD`
        Re-do the rest of calculation (e.g. sigmoid, concat)
        """
        labels, img_shapes, img_info, img_id = targets
        labels = labels.to(outputs[0].device)

        outputs = [outputs[i * 3:i * 3 + 3] for i in range(3)]
        multi_H_outputs = []
        for o in outputs:
            reg_output, obj_output, cls_output = o
            _single_H_output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )
            multi_H_outputs.append(_single_H_output)

        outputs_hw = [x.shape[-2:] for x in multi_H_outputs]
        merged_outputs = torch.cat(
            [x.flatten(start_dim=2) for x in multi_H_outputs], dim=2
        ).permute(0, 2, 1)

        merged_outputs = self._decode_outputs(merged_outputs, outputs_hw)

        # convert output from cxcywh -> xyxy format
        detections = postprocess(merged_outputs, num_classes=self.num_classes, conf_thre=self.conf_thres, nms_thre=self.iou_thres, class_agnostic=False)

        for i, d in enumerate(detections):
            if d is None:
                continue
            box, obj_conf, class_conf, class_label = d.split([4, 1, 1, 1], 1)
            _ts = labels[labels[:, 0] == i][:, 1:]
            super().update(
                [{
                    "labels": class_label.ravel().long(),
                    "scores": (obj_conf * class_conf).ravel(),
                    "boxes": box,  # xyxy
                }],
                [{
                    "labels": _ts[:, -1].ravel().to(torch.int32).long(),
                    "boxes": _ts[:, :4]
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
        "conv_62", "conv_65", "conv_66",
        "conv_74", "conv_70", "conv_73",
        "conv_78", "conv_82", "conv_81",
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
    global BASE_DIR, VAL_ANN, MODEL_NAME, IMG_SIZE, CONF_THRESHOLD, IOU_THRESHOLD

    print("\n".join(f"{k}={v}" for k, v in vars(config).items()))  # pretty print argparse

    config.data = config.data if os.path.isabs(config.data) else str(BASE_DIR / config.data)
    if os.path.exists(config.data) is False:
        raise FileNotFoundError("Could not find default dataset please check `--data`")

    config.output_dir = config.output_dir if os.path.isabs(config.output_dir) else str(BASE_DIR / config.output_dir)

    """
    Define Model
    ====================================================================================================================
    """
    device = "cuda"

    exp = get_exp_by_name(MODEL_NAME)
    exp.input_size = IMG_SIZE

    model = exp.get_model()
    model.to(device).float()

    replace_HEAD(model)

    resume_compression_flag = False
    _optimizer_state_dict = None
    if config.train_from_scratch is False:
        config.ckpt = config.ckpt if os.path.isabs(config.ckpt) else str(BASE_DIR / config.ckpt)

        if config.ckpt.rsplit(".", 1)[-1] == "pompom":
            warnings.warn(".pompom file provided, resuming compression (argparse attributes ignored)")
            resume_compression_flag = True
        else:
            print(f"loading ckpt from {config.ckpt}")
            ckpt = torch.load(config.ckpt, map_location=device)
            model.load_state_dict(ckpt["model"])

            _optimizer_state_dict = ckpt["optimizer"]

    """
    Define Loss Function
    ====================================================================================================================
    """
    model.head.use_l1 = True  # add additional L1 loss

    train_losses = CRITERION_WRAPPER(head_module=model.head)
    train_losses = {"loss_sum": train_losses}
    eval_losses = None

    """
    Define Optimizer
    ====================================================================================================================
    """
    optimizer = exp.get_optimizer(config.batch_size)
    if _optimizer_state_dict:  # if ckpt provided
        warnings.warn("using optimizer state dict from checkpoint (`--lr` cmd argument is ignored)")
        optimizer.load_state_dict(ckpt["optimizer"])
    # override the lr with the value that was set by the user
    optimizer.defaults["lr"] = config.lr
    for group in optimizer.param_groups:
        group["lr"] = config.lr


    """
    Define Dataloaders
    ====================================================================================================================
    """
    exp.data_dir = config.data

    get_train_loader = partial(get_train_loader_, exp=exp, config=config)
    get_eval_loader = partial(get_eval_loader_, config=config)

    """
    Define Metric Wrapper
    ====================================================================================================================
    """
    eval_metrics = MeanAveragePrecisionWrapper(
        strides=(8, 16, 32),
        conf_thres=CONF_THRESHOLD,
        iou_thres=IOU_THRESHOLD)
    eval_metrics = {"mAP": eval_metrics}
    train_metrics = None

    """
    RUN Compression
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
    parser = argparse.ArgumentParser(description="CLIKA YOLOX Example")
    parser.add_argument("--target_framework", type=str, default="trt", choices=["tflite", "trt"], help="choose the target framework TensorFlow Lite or TensorRT")
    parser.add_argument("--data", type=str, default="COCO", help="Dataset directory")

    # CLIKA Engine Training Settings
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Number of steps per epoch")
    parser.add_argument("--evaluation_steps", type=int, default=None, help="Number of steps for evaluation")
    parser.add_argument("--stats_steps", type=int, default=50, help="Number of steps for scans")
    parser.add_argument("--print_interval", type=int, default=50, help="COE print log interval")
    parser.add_argument("--ma_window_size", type=int, default=20, help="Number of steps for averaging print")
    parser.add_argument("--save_interval", action="store_true", default=None, help="Save interval compressed files each X epoch as .pompom files")
    parser.add_argument("--reset_train_data", action="store_true", default=False, help="Reset training dataset between epochs")
    parser.add_argument("--reset_eval_data", action="store_true", default=False, help="Reset evaluation dataset between epochs")
    parser.add_argument("--grads_acc_steps", type=int, default=4, help="Number of gradient accumulation steps (default: 4)")
    parser.add_argument("--no_mixed_precision", action="store_false", default=True, dest="mixed_precision", help="Not using Mixed Precision")
    parser.add_argument("--lr_warmup_epochs", type=int, default=1, help="Learning Rate used in the Learning Rate Warmup stage (default: 1)")
    parser.add_argument("--lr_warmup_steps_per_epoch", type=int, default=500, help="Number of steps per epoch used in the Learning Rate Warmup stage")
    parser.add_argument("--fp16_weights", action="store_true", default=False, help="Use FP16 weight (can reduce VRAM usage)")
    parser.add_argument("--gradients_checkpoint", action="store_true", default=False, help="Use gradient checkpointing")

    # Model Training Setting
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model (default: 100)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training and evaluation (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    parser.add_argument("--workers", type=int, default=3, help="Number of worker processes for data loading (default: 3)")
    parser.add_argument("--ckpt", type=str, default="yolox_s.pth", help="Path to load the model checkpoints (e.g. .pth, .pompom)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for saving results and checkpoints (default: outputs)")
    parser.add_argument("--train_from_scratch", action="store_true", help="Train the model from scratch")

    # Quantization Config
    parser.add_argument("--weights_num_bits", type=int, default=8, help="How many bits to use for the Weights for Quantization")
    parser.add_argument("--activations_num_bits", type=int, default=8, help="How many bits to use for the Activation for Quantization")

    args = parser.parse_args()
    main(args)
