import argparse
import os
import random
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Union, Dict

import numpy as np
import torch
import torchmetrics
import torchvision
from clika_compression import (
    PyTorchCompressionEngine,
    QATQuantizationSettings,
    DeploymentSettings_TensorRT_ONNX,
    DeploymentSettings_TFLite,
    DeploymentSettings_ONNXRuntime_ONNX,
)
from clika_compression.settings import generate_default_settings, ModelCompileSettings
from torch.utils.data import Dataset

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).parent
COMPDTYPE = Union[Dict[str, Union[Callable, torch.nn.Module]], None]

deployment_kwargs = {
    "graph_author": "CLIKA",
    "graph_description": None,
    "input_shapes_for_deployment": [(None, 1, 28, 28)],
}
DEPLOYMENT_DICT = {
    "trt": DeploymentSettings_TensorRT_ONNX(**deployment_kwargs),
    "ort": DeploymentSettings_ONNXRuntime_ONNX(**deployment_kwargs),
    "tflite": DeploymentSettings_TFLite(**deployment_kwargs),
}


# Define Class/Function Wrappers
# ==================================================================================================================== #
def get_loader(data_dir, batch_size, num_workers=0, train=True):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=train,
        transform=transform,
        target_transform=None,
        download=True,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return data_loader


class MultiClassAccuracy(torchmetrics.classification.accuracy.MulticlassAccuracy):
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        return super().update(preds, target)

    def compute(self) -> torch.Tensor:
        results = super().compute()
        return results * 100.0


# ==================================================================================================================== #


# https://github.com/pytorch/examples/blob/7f7c222b355abd19ba03a7d4ba90f1092973cdbc/mnist/main.py#L11
class BasicMNIST(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(
            x, kernel_size=(2, 2), stride=None, padding=(0, 0), dilation=(1, 1)
        )
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


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

    # guarantee static location
    config.output_dir = (
        config.output_dir
        if os.path.isabs(config.output_dir)
        else str(BASE_DIR / config.output_dir)
    )
    config.data = (
        config.data if os.path.isabs(config.data) else str(BASE_DIR / config.data)
    )

    """
    Define Model
    ====================================================================================================================
    """
    resume_compression_flag = False

    model = BasicMNIST()
    if (config.train_from_scratch is False) and (config.ckpt is not None):
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
            state_dict = torch.load(config.ckpt)
            model.load_state_dict(state_dict["model"])

    """
    Define Loss Function
    ====================================================================================================================
    """
    compute_loss_fn = lambda a, b: torch.nn.functional.nll_loss(
        torch.nn.functional.log_softmax(a, dim=1), b
    )
    train_losses = {"NLL_loss": compute_loss_fn}
    eval_losses = {"NLL_loss": compute_loss_fn}

    """
    Define Optimizer
    ====================================================================================================================
    """
    optimizer = torch.optim.Adadelta(params=model.parameters(), lr=config.lr)

    """
    Define Dataloaders
    ====================================================================================================================
    """
    get_train_loader = partial(
        get_loader, config.data, config.batch_size, config.workers, True
    )
    get_eval_loader = partial(
        get_loader, config.data, config.batch_size, config.workers, False
    )

    """
    Define Metric Wrapper
    ====================================================================================================================
    """
    metric_fn = MultiClassAccuracy(num_classes=10)
    train_metrics = {"acc": metric_fn}
    eval_metrics = {"acc": metric_fn}

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
    parser = argparse.ArgumentParser(description='CLIKA MNIST Example')
    parser.add_argument('--target_framework', type=str, default='trt', choices=['tflite', 'ort', 'trt'], help='choose the target framework TensorFlow Lite or TensorRT')
    parser.add_argument('--data', type=str, default='.', help='Dataset directory')

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
    parser.add_argument('--lr_warmup_epochs', type=int, default=0, help='Learning Rate used in the Learning Rate Warmup stage (default: 0)')
    parser.add_argument('--lr_warmup_steps_per_epoch', type=int, default=0, help='Number of steps per epoch used in the Learning Rate Warmup stage')
    parser.add_argument('--fp16_weights', action='store_true', default=False, help='Use FP16 weight (can reduce VRAM usage)')
    parser.add_argument('--gradients_checkpoint', action='store_true', default=False, help='Use gradient checkpointing')

    # Model Training Setting
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for the optimizer (default: 1e-2)')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes for data loading (default: 4)')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to load the model checkpoints (e.g. .pth, .pompom)')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for saving results and checkpoints (default: outputs)')
    parser.add_argument('--train_from_scratch', action='store_true', help='Train the model from scratch')
    parser.add_argument('--multi_gpu', action='store_true', help='Use Multi-GPU Distributed Compression')

    # Quantization Config
    parser.add_argument('--weights_num_bits', type=int, default=8, help='How many bits to use for the Weights for Quantization')
    parser.add_argument('--activations_num_bits', type=int, default=8, help='How many bits to use for the Activation for Quantization')

    args = parser.parse_args()
    main(config=args)
