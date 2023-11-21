import argparse
import os
import random
import sys
import warnings
from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import Union, Dict, Callable

import numpy as np
import torch
import torchmetrics
import torchvision.transforms.functional as TF
from PIL import Image
from clika_compression import (
    PyTorchCompressionEngine,
    QATQuantizationSettings,
    DeploymentSettings_TensorRT_ONNX,
    DeploymentSettings_TFLite,
    DeploymentSettings_ONNXRuntime_ONNX,
)
from clika_compression.settings import generate_default_settings, ModelCompileSettings
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "IMDN"))
from model.architecture import IMDN
from model.block import CCALayer
from data.DIV2K import div2k
from utils import load_state_dict

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

SCALE = 4
COMPDTYPE = Union[Dict[str, Union[Callable, torch.nn.Module]], None]

deployment_kwargs = {
    "graph_author": "CLIKA",
    "graph_description": None,
    "input_shapes_for_deployment": [(None, 3, None, None)],
}
DEPLOYMENT_DICT = {
    "trt": DeploymentSettings_TensorRT_ONNX(**deployment_kwargs),
    "ort": DeploymentSettings_ONNXRuntime_ONNX(**deployment_kwargs),
    "tflite": DeploymentSettings_TFLite(**deployment_kwargs),
}


# Define Class/Function Wrappers
# ==================================================================================================================== #
def replace_CCALayer(model):
    """
    Currently not supported operation
    Replace with identity layer
    https://github.com/Zheng222/IMDN/blob/8f158e6a5ac9db6e5857d9159fd4a6c4214da574/model/block.py#L82-L86
    """
    for _, m in model.named_children():
        if isinstance(m, CCALayer):
            m.contrast = torch.nn.Identity()
        else:
            replace_CCALayer(m)


def get_train_loader_(config):
    kwargs = {
        "scale": SCALE,
        "root": config.data + "/div2k",
        "ext": ".png",
        "n_colors": 3,
        "rgb_range": 1,
        "n_train": 800,  # num of training images
        "patch_size": 192,
        "phase": "train",
        "test_every": 1000,
        "batch_size": config.batch_size,
    }
    opt = namedtuple("FAKE_OPT", kwargs.keys())(*kwargs.values())
    dataset = div2k(opt)
    loader = DataLoader(
        dataset,
        config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        drop_last=True,
    )
    return loader


def get_eval_loader_(config):
    dataset = REDS(config.data + "/REDS4")
    loader = DataLoader(
        dataset, config.batch_size, shuffle=True, num_workers=config.workers
    )
    return loader


# clonned IMDN repo does not contain proper evaluation loader
class REDS(Dataset):
    """
    REDS dataset (Evaluation Dataset, not for Training)
    x4 scaling (x2, x3 not available)
    """

    def __init__(self, data_dir: str):
        """
        :param data_dir: test data folder
        """
        data_dir = Path(data_dir)
        dir_hr = data_dir.joinpath("GT")
        self.dir_hr = sorted((str(f) for f in dir_hr.rglob("*.png")))
        dir_lr = data_dir.joinpath("sharp_bicubic")
        self.dir_lr = sorted((str(f) for f in dir_lr.rglob("*.png")))

    def __getitem__(self, idx):
        """
        returning a tuple of (sample,label) to conform with CLIKA Compression constraints,
        https://docs.clika.io/docs/next/compression-constrains/cco_inputs_requirements#Dataset-Dataloader
        """
        hr_img = Image.open(self.dir_hr[idx])
        lr_img = Image.open(self.dir_lr[idx])

        return TF.to_tensor(lr_img), TF.to_tensor(hr_img)

    def __len__(self):
        return len(self.dir_hr)


# ==================================================================================================================== #


def resume_compression(
    config: argparse.Namespace,
    get_train_loader: Callable,
    get_eval_loader: Callable,
    train_losses: COMPDTYPE,
    train_metrics: COMPDTYPE,
    eval_losses: COMPDTYPE = None,
    eval_metrics: COMPDTYPE = None,
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
        settings=None,
        multi_gpu=config.multi_gpu,
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
    eval_metrics: COMPDTYPE = None,
):
    global DEPLOYMENT_DICT, RANDOM_SEED

    engine = PyTorchCompressionEngine()
    settings = generate_default_settings()
    # fmt: off
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
    settings.training_settings.random_seed = RANDOM_SEED
    # fmt: on
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
        is_training_from_scratch=config.train_from_scratch,
        multi_gpu=config.multi_gpu,
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
    global BASE_DIR, DEPLOYMENT_DICT, COMPDTYPE

    print(
        "\n".join(f"{k}={v}" for k, v in vars(config).items())
    )  # pretty print argparse

    config.data = (
        config.data if os.path.isabs(config.data) else str(BASE_DIR / config.data)
    )
    if os.path.exists(config.data) is False:
        raise FileNotFoundError("Could not find default dataset please check `--data`")

    config.output_dir = (
        config.output_dir
        if os.path.isabs(config.output_dir)
        else str(BASE_DIR / config.output_dir)
    )

    """
    Define Model
    ====================================================================================================================
    """
    model = IMDN(upscale=SCALE)

    resume_compression_flag = False
    if config.train_from_scratch is False:
        config.ckpt = (
            config.ckpt if os.path.isabs(config.ckpt) else str(BASE_DIR / config.ckpt)
        )

        if config.ckpt.rsplit(".", 1)[-1] == "pompom":
            warnings.warn(
                ".pompom file provided, resuming compression (argparse attributes ignored)"
            )
            resume_compression_flag = True
        else:
            print(f"loading ckpt from {config.ckpt}")
            state_dict = load_state_dict(config.ckpt)
            model.load_state_dict(state_dict)

    replace_CCALayer(model=model)
    model.cuda()

    """
    Define Loss Function
    ====================================================================================================================
    """
    compute_loss_fn = torch.nn.L1Loss()
    train_losses = {"l1_loss": compute_loss_fn}
    eval_losses = {"l1_loss": compute_loss_fn}

    """
    Define Optimizer
    ====================================================================================================================
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    """
    Define Dataloaders
    ====================================================================================================================
    """
    get_train_loader = partial(get_train_loader_, config=config)
    get_eval_loader = partial(get_eval_loader_, config=config)

    """
    Define Metric Wrapper
    ====================================================================================================================
    """
    train_metrics = None
    eval_metrics = {"psnr": torchmetrics.PeakSignalNoiseRatio()}

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
            eval_metrics=eval_metrics,
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
            eval_metrics=eval_metrics,
        )


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description='CLIKA IMDN Example')
    parser.add_argument('--target_framework', type=str, default='trt', choices=['tflite', 'ort', 'trt'], help='choose the target framework TensorFlow Lite or TensorRT')
    parser.add_argument('--data', type=str, default='dataset', help='Dataset directory')

    # CLIKA Engine Training Settings
    parser.add_argument('--steps_per_epoch', type=int, default=None, help='Number of steps per epoch')
    parser.add_argument('--evaluation_steps', type=int, default=None, help='Number of steps for evaluation')
    parser.add_argument('--stats_steps', type=int, default=50, help='Number of steps for scans')
    parser.add_argument('--print_interval', type=int, default=50, help='COE print log interval')
    parser.add_argument('--ma_window_size', type=int, default=20, help='Number of steps for averaging print')
    parser.add_argument('--save_interval', action='store_true', default=None, help='Save interval compressed files each X epoch as .pompom files')
    parser.add_argument('--reset_train_data', action='store_true', default=False, help='Reset training dataset between epochs')
    parser.add_argument('--reset_eval_data', action='store_true', default=False, help='Reset evaluation dataset between epochs')
    parser.add_argument('--grads_acc_steps', type=int, default=1, help='Number of gradient accumulation steps (default: 1)')
    parser.add_argument('--no_mixed_precision', action='store_false', default=True, dest='mixed_precision', help='Not using Mixed Precision')
    parser.add_argument('--lr_warmup_epochs', type=int, default=1, help='Learning Rate used in the Learning Rate Warmup stage (default: 1)')
    parser.add_argument('--lr_warmup_steps_per_epoch', type=int, default=500, help='Number of steps per epoch used in the Learning Rate Warmup stage')
    parser.add_argument('--fp16_weights', action='store_true', default=False, help='Use FP16 weight (can reduce VRAM usage)')
    parser.add_argument('--gradients_checkpoint', action='store_true', default=False, help='Use gradient checkpointing')

    # Model Training Setting
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model (default: 100)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and evaluation (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for the optimizer (default: 1e-5)')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes for data loading (default: 4)')
    parser.add_argument('--ckpt', type=str, default='IMDN/checkpoints/IMDN_x4.pth', help='Path to load the model checkpoints (e.g. .pth, .pompom)')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for saving results and checkpoints (default: outputs)')
    parser.add_argument('--train_from_scratch', action='store_true', help='Train the model from scratch')
    parser.add_argument('--multi_gpu', action='store_true', help='Use Multi-GPU Distributed Compression')

    # Quantization Config
    parser.add_argument('--weights_num_bits', type=int, default=8, help='How many bits to use for the Weights for Quantization')
    parser.add_argument('--activations_num_bits', type=int, default=8, help='How many bits to use for the Activation for Quantization')

    args = parser.parse_args()
    main(config=args)
