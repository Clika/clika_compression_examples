import argparse
import sys
import warnings
from functools import partial
from pathlib import Path
from pprint import pprint

import torch
import torchvision
from clika_compression import Settings, TensorBoardCallback, clika_compress, clika_resume
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader, default_collate
from torchmetrics.classification import MulticlassAccuracy

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "vision"))
import transforms
import utils
from train import load_data


class FakeOpt(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Define Class/Function Wrappers
# ==================================================================================================================== #


def batch_accuracy(outputs, targets, topk=(1,)) -> torch.Tensor:
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


class MetricWrapper(MulticlassAccuracy):
    def compute(self) -> torch.Tensor:
        return super().compute() * 100.0


def _train_collate_fn(batch, transform):
    return transform(*default_collate(batch))


def get_loader(
    data_dir: Path,
    batch_size: int,
    workers: int,
    train_crop_size: int,
    val_crop_size: int,
    val_resize_size: int,
    train=True,
) -> DataLoader:
    """Factory function that generates train/eval Dataloader.

    REFERENCE:
    https://github.com/pytorch/vision/blob/2030d208ba1044b97b8ceab91852858672a56cc8/references/classification/train.py#L217-L219
    """
    train_dir = str(data_dir / "train")
    val_dir = str(data_dir / "val")

    kwargs = {
        "val_resize_size": val_resize_size,
        "val_crop_size": val_crop_size,
        "train_crop_size": train_crop_size,
        "interpolation": "bilinear",
        "cache_dataset": False,
        "auto_augment": "ta_wide",
        "random_erase": 0.1,
        "ra_magnitude": 9,
        "augmix_severity": 3,
        "test_only": False,
        "weights": None,
        "backend": "PIL",
        "distributed": False,
        "ra_sampler": False,
        "ra_reps": 3,
    }
    opt = FakeOpt(**kwargs)

    collate_fn = None
    if train is True:
        dataset, _, sampler, _ = load_data(train_dir, val_dir, opt)

        mixup_transforms = [
            transforms.RandomMixup(1000, p=1.0, alpha=0.2),
            transforms.RandomCutmix(1000, p=1.0, alpha=1.0),
        ]
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        collate_fn = partial(_train_collate_fn, transform=mixupcutmix)
    else:
        _, dataset, _, sampler = load_data(train_dir, val_dir, opt)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    return loader


# ==================================================================================================================== #


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="CLIKA EfficientNet Example")

    ace_parser = parser.add_argument_group("CLIKA ACE configuration")
    ace_parser.add_argument("--config", type=str, default="config.yml", help="ACE config yaml path")

    model_parser = parser.add_argument_group("Model configuration")
    model_parser.add_argument("--output_dir", type=Path, default="outputs", help="Path to save clika related files for the SDK")
    model_parser.add_argument("--data", type=Path, default="ILSVRC/Data/CLS-LOC", help="Dataset directory")
    model_parser.add_argument("--ckpt", type=Path, default=None, help="Path to load the model checkpoints (e.g. .pth, .pompom)")
    model_parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and evaluation (default: 128)")
    model_parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    model_parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for data loading (default: 4)")
    model_parser = parser.add_argument_group("Additional arguments")
    model_parser.add_argument("--model_type", type=str, choices=["medium", "small"], default="small", help="EfficientNet model size (default: small)")

    args = parser.parse_args()

    if args.data.exists() is False:
        raise FileNotFoundError(f"Unknown directory: {args.data}")
    # fmt: on

    return args


def main() -> None:
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
    _weights = None if _train_from_scratch else "DEFAULT"
    if args.model_type == "small":
        model = torchvision.models.get_model("efficientnet_v2_s", weights=_weights, num_classes=1000)
        train_crop_size = 300
        val_crop_size = 384
        val_resize_size = 384
    elif args.model_type == "medium":
        model = torchvision.models.get_model("efficientnet_v2_m", weights=_weights, num_classes=1000)
        train_crop_size = 384
        val_crop_size = 480
        val_resize_size = 480
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    _resume = False
    if (_train_from_scratch is False) and (args.ckpt is not None):
        if args.ckpt.suffix == ".pompom":
            print("[INFO] .pompom file provided as ckpt, resuming compression ...")
            _resume = True
        else:
            print(f"[INFO] loading custom ckpt from {args.ckpt}")
            state_dict = torch.load(str(args.ckpt))
            model.load_state_dict(state_dict)
    torch.fx.symbolic_trace(model)

    """
    Define Loss Function
    ====================================================================================================================
    """
    train_losses = eval_losses = {"CrossEntropyLoss": torch.nn.CrossEntropyLoss()}

    """
    Define Optimizer
    ====================================================================================================================
    """
    parameters = utils.set_weight_decay(
        model,
        weight_decay=0.00002,
        norm_weight_decay=None,
        custom_keys_weight_decay=None,
    )
    optimizer = SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=0.00002)

    """
    Define Dataloaders
    ====================================================================================================================
    """
    _get_loader_baseline = partial(
        get_loader,
        data_dir=args.data,
        batch_size=args.batch_size,
        workers=args.workers,
        train_crop_size=train_crop_size,
        val_crop_size=val_crop_size,
        val_resize_size=val_resize_size,
    )
    get_train_loader = partial(_get_loader_baseline, train=True)
    get_eval_loader = partial(_get_loader_baseline, train=False)

    """
    Define Metric Wrapper
    ====================================================================================================================
    """
    eval_metrics = {
        "top1": MetricWrapper(num_classes=1000, top_k=1),
        "top5": MetricWrapper(num_classes=1000, top_k=5),
    }
    train_metrics = {
        "batch_acc_top1": partial(batch_accuracy, topk=(1,)),
        "batch_acc_top5": partial(batch_accuracy, topk=(5,)),
    }

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
