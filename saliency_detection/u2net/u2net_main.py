import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from typing import Union, Dict, Callable
import argparse
from functools import partial
import warnings
from pathlib import Path

import numpy as np
from skimage import color
import torch
import torchvision

from clika_compression import PyTorchCompressionEngine, QATQuantizationSettings, DeploymentSettings_TensorRT_ONNX, DeploymentSettings_TFLite
from clika_compression.settings import (
    generate_default_settings, LayerQuantizationSettings, ModelCompileSettings
)

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "U-2-Net"))
from model import U2NET
from data_loader import RescaleT, RandomCrop, SalObjDataset

COMPDTYPE = Union[Dict[str, Union[Callable, torch.nn.Module]], None]

DEPLOYMENT_DICT = {
    "trt": DeploymentSettings_TensorRT_ONNX(
        graph_author="CLIKA",
        graph_description=None,
        input_shapes_for_deployment=[(None, 3, None, None)]),

    "tflite": DeploymentSettings_TFLite(
        graph_author="CLIKA",
        graph_description=None,
        input_shapes_for_deployment=[(None, 3, None, None)])}


# Define Class/Function Wrappers
# ==================================================================================================================== #
class ToTensorLab(object):
    """Transform class defined inside `u2net/data_loader.py`
    Redefine inside u2net_main.py to avoid
    `torch.from_numpy` ValueError: Tensors with negative strides are not currently supported"""

    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):

        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        tmpLbl = np.zeros(label.shape)

        if (np.max(label) < 1e-6):
            label = label
        else:
            label = label / np.max(label)

        # change the color space
        if self.flag == 2:  # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
            tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmpImgt[:, :, 0] = image[:, :, 0]
                tmpImgt[:, :, 1] = image[:, :, 0]
                tmpImgt[:, :, 2] = image[:, :, 0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:, :, 0] = (tmpImgt[:, :, 0] - np.min(tmpImgt[:, :, 0])) / (
                        np.max(tmpImgt[:, :, 0]) - np.min(tmpImgt[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImgt[:, :, 1] - np.min(tmpImgt[:, :, 1])) / (
                        np.max(tmpImgt[:, :, 1]) - np.min(tmpImgt[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImgt[:, :, 2] - np.min(tmpImgt[:, :, 2])) / (
                        np.max(tmpImgt[:, :, 2]) - np.min(tmpImgt[:, :, 2]))
            tmpImg[:, :, 3] = (tmpImgtl[:, :, 0] - np.min(tmpImgtl[:, :, 0])) / (
                        np.max(tmpImgtl[:, :, 0]) - np.min(tmpImgtl[:, :, 0]))
            tmpImg[:, :, 4] = (tmpImgtl[:, :, 1] - np.min(tmpImgtl[:, :, 1])) / (
                        np.max(tmpImgtl[:, :, 1]) - np.min(tmpImgtl[:, :, 1]))
            tmpImg[:, :, 5] = (tmpImgtl[:, :, 2] - np.min(tmpImgtl[:, :, 2])) / (
                        np.max(tmpImgtl[:, :, 2]) - np.min(tmpImgtl[:, :, 2]))

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])
            tmpImg[:, :, 3] = (tmpImg[:, :, 3] - np.mean(tmpImg[:, :, 3])) / np.std(tmpImg[:, :, 3])
            tmpImg[:, :, 4] = (tmpImg[:, :, 4] - np.mean(tmpImg[:, :, 4])) / np.std(tmpImg[:, :, 4])
            tmpImg[:, :, 5] = (tmpImg[:, :, 5] - np.mean(tmpImg[:, :, 5])) / np.std(tmpImg[:, :, 5])

        elif self.flag == 1:  # with Lab color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            if image.shape[2] == 1:
                tmpImg[:, :, 0] = image[:, :, 0]
                tmpImg[:, :, 1] = image[:, :, 0]
                tmpImg[:, :, 2] = image[:, :, 0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.min(tmpImg[:, :, 0])) / (
                        np.max(tmpImg[:, :, 0]) - np.min(tmpImg[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.min(tmpImg[:, :, 1])) / (
                        np.max(tmpImg[:, :, 1]) - np.min(tmpImg[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.min(tmpImg[:, :, 2])) / (
                        np.max(tmpImg[:, :, 2]) - np.min(tmpImg[:, :, 2]))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])

        else:  # with rgb color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            image = image / np.max(image)
            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        # monkey patched to call copy on ndarray -> avoid ValueError
        return {'imidx': torch.from_numpy(imidx.copy()), 'image': torch.from_numpy(tmpImg.copy()), 'label': torch.from_numpy(tmpLbl.copy())}


# https://github.com/xuebinqin/U-2-Net/blob/53dc9da026650663fc8d8043f3681de76e91cfde/u2net_train.py#L31
def CRITERION_WRAPPER(preds, gt) -> torch.Tensor:
    d0, d1, d2, d3, d4, d5, d6 = preds
    loss0 = torch.nn.functional.binary_cross_entropy(d0, gt)
    loss1 = torch.nn.functional.binary_cross_entropy(d1, gt)
    loss2 = torch.nn.functional.binary_cross_entropy(d2, gt)
    loss3 = torch.nn.functional.binary_cross_entropy(d3, gt)
    loss4 = torch.nn.functional.binary_cross_entropy(d4, gt)
    loss5 = torch.nn.functional.binary_cross_entropy(d5, gt)
    loss6 = torch.nn.functional.binary_cross_entropy(d6, gt)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss


class DATASET_WRAPPER(SalObjDataset):
    def __init__(self, *attrs, **kwargs):
        super().__init__(*attrs, **kwargs)

    def __getitem__(self, item):
        """
        returning a tuple of (sample,label) to conform with CLIKA Compression constraints,
        https://docs.clika.io/docs/next/compression-constrains/cco_inputs_requirements#Dataset-Dataloader
        """
        d = super().__getitem__(item)
        return d["image"].float(), d["label"].float()


def get_loader(config: argparse.Namespace, img_dir: str, label_dir: str):
    TRAIN_IMGS_DIR = Path(config.data) / img_dir
    TRAIN_LABELS_DIR = Path(config.data) / label_dir

    imgs = list(TRAIN_IMGS_DIR.glob("*.jpg"))
    labels = list(TRAIN_LABELS_DIR.glob("*.png"))
    imgs = sorted(imgs)
    labels = sorted(labels)
    for x, y in zip(imgs, labels):
        if x.stem != y.stem:
            raise ValueError("mismatch")

    salobj_dataset = DATASET_WRAPPER(
        img_name_list=imgs,
        lbl_name_list=labels,
        transform=torchvision.transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = torch.utils.data.DataLoader(salobj_dataset,
                                                    batch_size=config.batch_size,
                                                    shuffle=True,
                                                    num_workers=config.workers)
    return salobj_dataloader


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
    layer_names_to_skip = {
        "sigmoid", "conv_118",
        "sigmoid_2", "upsample_33", "slice_33", "shape_33", "conv_113", "conv_112",
        "sigmoid_3", "upsample_34", "conv_114", "slice_34", "shape_34",
        "sigmoid_5", "upsample_36", "conv_116", "slice_36", "shape_36",
        "sigmoid_1",
        "sigmoid_4", "upsample_35", "conv_115", "slice_35", "shape_35",
        "sigmoid_6", "upsample_37", "conv_117", "slice_37", "shape_37",
        "concat_50"
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
    global DEPLOYMENT_DICT

    print("\n".join(f"{k}={v}" for k, v in vars(config).items()))  # pretty print argparse

    config.data = config.data if os.path.isabs(config.data) else str(BASE_DIR / config.data)
    if os.path.exists(config.data) is False:
        raise FileNotFoundError("Could not find default dataset please check `--data`")

    config.output_dir = config.output_dir if os.path.isabs(config.output_dir) else str(BASE_DIR / config.output_dir)

    """
    Define Model
    ====================================================================================================================
    """
    config.mixed_precision = False  # model leaf nodes are sigmoids

    resume_compression_flag = False
    model = U2NET(3, 1)
    if config.train_from_scratch is False:
        config.ckpt = config.ckpt if os.path.isabs(config.ckpt) else str(BASE_DIR / config.ckpt)

        if config.ckpt.rsplit(".", 1)[-1] == "pompom":
            warnings.warn(".pompom file provided, resuming compression (argparse attributes ignored)")
            resume_compression_flag = True
        else:
            print(f"loading ckpt from {config.ckpt}")
            state_dict = torch.load(config.ckpt)
            model.load_state_dict(state_dict)

    """
    Define Loss Function
    ====================================================================================================================
    """
    train_losses = eval_losses = {"muti_bce_loss_fusion": CRITERION_WRAPPER}
    train_metrics = eval_metrics = None

    """
    Define Optimizer
    ====================================================================================================================
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    """
    Define Dataloaders
    ====================================================================================================================
    """
    get_train_loader = partial(get_loader, config=config, img_dir="DUTS-TR/DUTS-TR-Image", label_dir="DUTS-TR/DUTS-TR-Mask")
    get_eval_loader = partial(get_loader, config=config, img_dir="DUTS-TE/DUTS-TE-Image", label_dir="DUTS-TE/DUTS-TE-Mask")

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

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIKA U2-Net Example")
    parser.add_argument("--target_framework", type=str, default="trt", choices=["tflite", "trt"], help="choose the target framework TensorFlow Lite or TensorRT")
    parser.add_argument("--data", type=str, default="duts", help="Dataset directory")

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
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model (default: 10)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation (default: 8)")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate for the optimizer (default: 0.001)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for data loading (default: 4)")
    parser.add_argument("--ckpt", type=str, default="u2net.pth", help="Path to load the model checkpoints (e.g. .pth, .pompom)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for saving results and checkpoints (default: outputs)")
    parser.add_argument("--train_from_scratch", action="store_true", help="Train the model from scratch")

    # Quantization Config
    parser.add_argument("--weights_num_bits", type=int, default=8, help="How many bits to use for the Weights for Quantization")
    parser.add_argument("--activations_num_bits", type=int, default=8, help="How many bits to use for the Activation for Quantization")

    args = parser.parse_args()

    main(config=args)