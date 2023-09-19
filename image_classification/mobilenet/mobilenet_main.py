import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from typing import Union, Dict, Callable
import argparse
from functools import partial
import warnings
from pathlib import Path
from collections import namedtuple

import torch
from torch.optim import RMSprop
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, MobileNet_V3_Small_Weights
import torchmetrics

from clika_compression import PyTorchCompressionEngine, QATQuantizationSettings, DeploymentSettings_TensorRT_ONNX, \
    DeploymentSettings_TFLite, LayerSettings
from clika_compression.settings import (
    generate_default_settings, LayerQuantizationSettings, ModelCompileSettings
)

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "vision"))
from train import load_data
import utils

NUM_CLASSES = 1000  # imagenet 1k
VAL_RESIZE_SIZE: int = 256
VAL_CROP_SIZE: int = 224
TRAIN_CROP_SIZE: int = 224
MODEL_SIZE: str = "small"

MODEL_DICT = {"large": mobilenet_v3_large,
              "small": mobilenet_v3_small}
WEIGHT_DICT = {"large": MobileNet_V3_Large_Weights.IMAGENET1K_V2,
               "small": MobileNet_V3_Small_Weights.IMAGENET1K_V1}

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
def batch_accuracy(outputs, targets, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = targets.size(0)
        if targets.ndim == 2:
            targets = targets.max(dim=1)[1]

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res[0]


class MultiClassAccuracy(torchmetrics.classification.accuracy.MulticlassAccuracy):

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        return super().update(preds, target)

    def compute(self) -> torch.Tensor:
        results = super().compute()
        return results * 100.0


def get_loader(config, train=True):
    """
    factory function to return train/eval dataloaders

    https://github.com/pytorch/vision/blob/2030d208ba1044b97b8ceab91852858672a56cc8/references/classification/train.py#L217-L219
    """
    global VAL_RESIZE_SIZE, VAL_CROP_SIZE, TRAIN_CROP_SIZE

    train_dir = os.path.join(config.data, "train")
    val_dir = os.path.join(config.data, "val")

    kwargs = {
        "val_resize_size": VAL_RESIZE_SIZE,
        "val_crop_size": VAL_CROP_SIZE,
        "train_crop_size": TRAIN_CROP_SIZE,
        "interpolation": "bilinear",
        "cache_dataset": False,
        "auto_augment": None,
        "random_erase": 0.2,
        "ra_magnitude": 9,
        "augmix_severity": 3,
        "test_only": False,
        "weights": None,
        "backend": "PIL",
        "distributed": False,
        "ra_sampler": False,
        "ra_reps": 3,
    }
    opt = namedtuple("FAKE_OPT", kwargs.keys())(*kwargs.values())
    if train is True:
        dataset, _, sampler, _ = load_data(train_dir, val_dir, opt)
    else:
        _, dataset, _, sampler = load_data(train_dir, val_dir, opt)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.workers,
        pin_memory=True,
        collate_fn=None,
    )
    return loader


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

    # Set Training Settings
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
    settings.set_quantization_settings_for_layer("linear_1", LayerQuantizationSettings(skip_quantization=True))

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
    global BASE_DIR, NUM_CLASSES, DEPLOYMENT_DICT, COMPDTYPE, VAL_RESIZE_SIZE, VAL_CROP_SIZE, TRAIN_CROP_SIZE, MODEL_SIZE, MODEL_DICT, WEIGHT_DICT

    print("\n".join(f"{k}={v}" for k, v in vars(config).items()))  # pretty print argparse

    if config.data is None:
        imagenet_path = BASE_DIR / "ILSVRC/Data/CLS-LOC"
        imagenette_path = BASE_DIR / "imagenette2-160"
        if imagenet_path.exists():
            config.data = str(imagenet_path)
            print(f"setting {config.data} as default dataset")
        elif imagenette_path.exists():
            config.data = str(imagenette_path)
            print(f"setting {config.data} as default dataset")
        else:
            raise FileNotFoundError("Could not set default dataset path. Please confirm data folders exist.")
    else:
        config.data = config.data if os.path.isabs(config.data) else str(BASE_DIR / config.data)

    config.output_dir = str(BASE_DIR / config.output_dir)

    """
    Define Model
    ====================================================================================================================
    """
    resume_compression_flag = False

    if config.train_from_scratch is True:
        model = MODEL_DICT[MODEL_SIZE]()
    else:
        model = MODEL_DICT[MODEL_SIZE](weights=WEIGHT_DICT[MODEL_SIZE])

    if (config.train_from_scratch is False) and (config.ckpt is not None):
        config.ckpt = config.ckpt if os.path.isabs(config.ckpt) else str(BASE_DIR / config.ckpt)

        if config.ckpt.rsplit(".", 1)[-1] == "pompom":
            warnings.warn(".pompom file provided, resuming compression (argparse attributes ignored)")
            resume_compression_flag = True
        else:
            print(f"loading ckpt from {config.ckpt}")
            state_dict = torch.load(config.ckpt)
            model.load_state_dict(state_dict["model"])

    """
    Define Loss Function
    ====================================================================================================================
    """
    compute_loss_fn = torch.nn.CrossEntropyLoss()
    train_losses = eval_losses = {"CrossEntropyLoss": compute_loss_fn}

    """
    Define Optimizer
    ====================================================================================================================
    """
    parameters = utils.set_weight_decay(
        model,
        weight_decay=0.00001,
        norm_weight_decay=None,
        custom_keys_weight_decay=None
    )
    optimizer = RMSprop(parameters, lr=config.lr, momentum=0.9, weight_decay=0.00001, alpha=0.9, eps=0.0316)

    """
    Define Dataloaders
    ====================================================================================================================
    """
    get_train_loader = partial(get_loader, config=config, train=True)
    get_eval_loader = partial(get_loader, config=config, train=False)

    """
    Define Metric Wrapper
    ====================================================================================================================
    """
    eval_metrics = {"top1": MultiClassAccuracy(num_classes=NUM_CLASSES, top_k=1),
                    "top5": MultiClassAccuracy(num_classes=NUM_CLASSES, top_k=5)}
    train_metrics = {"batch_acc_top1": partial(batch_accuracy, topk=(1,)),
                     "batch_acc_top5": partial(batch_accuracy, topk=(5,))}

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
    parser = argparse.ArgumentParser(description="CLIKA MobileNet Example")
    parser.add_argument("--target_framework", type=str, default="trt", choices=["tflite", "trt"], help="choose the target framework TensorFlow Lite or TensorRT")
    parser.add_argument("--data", type=str, default=None, help="Dataset directory")

    # CLIKA Engine Training Settings
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Number of steps per epoch")
    parser.add_argument("--evaluation_steps", type=int, default=None, help="Number of steps for evaluation")
    parser.add_argument("--stats_steps", type=int, default=50, help="Number of steps for scans")
    parser.add_argument("--print_interval", type=int, default=50, help="COE print log interval")
    parser.add_argument("--ma_window_size", type=int, default=20, help="Number of steps for averaging print")
    parser.add_argument("--save_interval", action="store_true", default=None, help="Save interval compressed files each X epoch as .pompom files")
    parser.add_argument("--reset_train_data", action="store_true", default=False, help="Reset training dataset between epochs")
    parser.add_argument("--reset_eval_data", action="store_true", default=False, help="Reset evaluation dataset between epochs")
    parser.add_argument("--grads_acc_steps", type=int, default=1, help="Number of gradient accumulation steps (default: 1)")
    parser.add_argument("--no_mixed_precision", action="store_false", default=True, dest="mixed_precision", help="Not using Mixed Precision")
    parser.add_argument("--lr_warmup_epochs", type=int, default=1, help="Learning Rate used in the Learning Rate Warmup stage (default: 1)")
    parser.add_argument("--lr_warmup_steps_per_epoch", type=int, default=500, help="Number of steps per epoch used in the Learning Rate Warmup stage")
    parser.add_argument("--fp16_weights", action="store_true", default=False, help="Use FP16 weight (can reduce VRAM usage)")
    parser.add_argument("--gradients_checkpoint", action="store_true", default=False, help="Use gradient checkpointing")

    # Model Training Setting
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model (default: 10)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for data loading (default: 4)")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to load the model checkpoints (e.g. .pth, .pompom)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for saving results and checkpoints (default: outputs)")
    parser.add_argument("--train_from_scratch", action="store_true", help="Train the model from scratch")

    # Quantization Config
    parser.add_argument("--weights_num_bits", type=int, default=8, help="How many bits to use for the Weights for Quantization")
    parser.add_argument("--activations_num_bits", type=int, default=8, help="How many bits to use for the Activation for Quantization")

    args = parser.parse_args()

    main(args)
