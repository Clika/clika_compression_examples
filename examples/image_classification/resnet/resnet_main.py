import argparse
import sys
import warnings
from functools import partial
from pathlib import Path
from pprint import pprint

import torch
import torchvision
from clika_compression import Settings, TensorBoardCallback, clika_compress, clika_resume
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "vision"))
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
        results = super().compute()
        return results * 100.0


def get_loader(data_dir: Path, batch_size: int, workers: int, train=True) -> DataLoader:
    """
    factory function to return train/eval dataloaders

    https://github.com/pytorch/vision/blob/2030d208ba1044b97b8ceab91852858672a56cc8/references/classification/train.py#L217-L219
    """
    train_dir = str(data_dir / "train")
    val_dir = str(data_dir / "val")

    kwargs = {
        "val_resize_size": 256,
        "val_crop_size": 224,
        "train_crop_size": 224,
        "interpolation": "bilinear",
        "cache_dataset": False,
        "auto_augment": None,
        "random_erase": 0.0,
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

    if train is True:
        dataset_train, _, sampler, _ = load_data(train_dir, val_dir, opt)
        dataset = dataset_train
    else:
        _, dataset_eval, _, sampler = load_data(train_dir, val_dir, opt)
        dataset = dataset_eval

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=False,
        collate_fn=None,
    )
    return loader


# ==================================================================================================================== #


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="CLIKA ResNet Example")

    ace_parser = parser.add_argument_group("CLIKA ACE configuration")
    ace_parser.add_argument("--config", type=str, default="config.yml", help="ACE config yaml path")

    model_parser = parser.add_argument_group("Model configuration")
    model_parser.add_argument("--output_dir", type=Path, default="outputs", help="Path to save clika related files for the SDK")
    model_parser.add_argument("--data", type=Path, default="ILSVRC/Data/CLS-LOC", help="Dataset directory")
    model_parser.add_argument("--ckpt", type=Path, default=None, help="Path to load the model checkpoints (e.g. .pth, .pompom)")
    model_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation (default: 32)")
    model_parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    model_parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for data loading (default: 4)")
    model_parser = parser.add_argument_group("Additional arguments")
    model_parser.add_argument("--model_type", type=str, choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d"], default="resnet18", help="ResNet model type (default: resnet18)")

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
    if args.model_type == "resnet18":
        model = torchvision.models.get_model("resnet18", weights=_weights, num_classes=1000)
    elif args.model_type == "resnet34":
        model = torchvision.models.get_model("resnet34", weights=_weights, num_classes=1000)
    elif args.model_type == "resnet50":
        model = torchvision.models.get_model("resnet50", weights=_weights, num_classes=1000)
    elif args.model_type == "resnet101":
        model = torchvision.models.get_model("resnet101", weights=_weights, num_classes=1000)
    elif args.model_type == "resnet152":
        model = torchvision.models.get_model("resnet152", weights=_weights, num_classes=1000)
    elif args.model_type == "resnext50_32x4d":
        model = torchvision.models.get_model("resnext50_32x4d", weights=_weights, num_classes=1000)
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
    parameters = utils.set_weight_decay(model, weight_decay=1e-4, norm_weight_decay=None, custom_keys_weight_decay=None)
    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    """
    Define Dataloaders
    ====================================================================================================================
    """
    get_train_loader = partial(
        get_loader,
        data_dir=args.data,
        batch_size=args.batch_size,
        workers=args.workers,
        train=True,
    )
    get_eval_loader = partial(
        get_loader,
        data_dir=args.data,
        batch_size=args.batch_size,
        workers=args.workers,
        train=False,
    )

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
