import argparse
import warnings
from functools import partial
from pathlib import Path
from pprint import pprint

import torch
from clika_compression import Settings, TensorBoardCallback, clika_compress, clika_deploy, clika_resume
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

BASE_DIR = Path(__file__).parent.parent


class ExampleNet(torch.nn.Module):
    def forward(self, x):
        ...


def get_loader(data_dir: str, batch_size: int, workers: int, is_train: bool) -> DataLoader:
    if is_train:
        ...
    else:
        ...


def parse_args() -> argparse.Namespace:
    # fmt: off
    _model_name = ""
    parser = argparse.ArgumentParser(description=f"CLIKA {_model_name} Example")

    ace_parser = parser.add_argument_group("CLIKA ACE configuration")
    ace_parser.add_argument("--config", type=str, default="config.yml", help="ACE config yaml path")

    model_parser = parser.add_argument_group("Model configuration")
    model_parser.add_argument("--output_dir", type=Path, default="outputs", help="Path to save clika related files for the SDK")
    model_parser.add_argument("--data", type=Path, default="...", help="Dataset directory")
    model_parser.add_argument("--ckpt", type=Path, default=None, help="Path to load the model checkpoints (e.g. .pth, .pompom)")
    model_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation (default: 32)")
    model_parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    model_parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for data loading (default: 4)")
    model_parser = parser.add_argument_group("Additional arguments")
    # some additional arguments

    args = parser.parse_args()

    if args.data.exists() is False:
        raise FileNotFoundError(f"Unknown directory: {args.data}")
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
    model = ExampleNet()

    # TODO: model must be symbolically traceable
    torch.fx.symbolic_trace(model)

    _resume = False
    if (_train_from_scratch is False) and (args.ckpt is not None):
        if args.ckpt.suffix == ".pompom":
            print("[INFO] .pompom file provided as ckpt, resuming compression ...")
            _resume = True
        else:
            print(f"[INFO] loading custom ckpt from {args.ckpt}")
            state_dict = torch.load(str(args.ckpt))
            model.load_state_dict(state_dict)

    """
    Define Dataloaders
    ====================================================================================================================
    """
    get_train_loader = partial(
        get_loader, data_dir=args.data, batch_size=args.batch_size, workers=args.workers, is_train=True
    )
    get_eval_loader = partial(
        get_loader, data_dir=args.data, batch_size=args.batch_size, workers=args.workers, is_train=False
    )

    # TODO: must return tuple of 2: each corresponding to ({input data}, {target data})
    #  e.g. (img, target_cls)
    #  e.g. (img, (bounding_box, class, filename))
    _train_input, _train_target = next(iter(get_train_loader()))
    _eval_input, _eval_target = next(iter(get_eval_loader()))

    """
    Define Optimizer
    ====================================================================================================================
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    """
    Define Loss Function
    ====================================================================================================================
    """
    train_losses = {"train_loss": torch.nn.CrossEntropyLoss()}
    eval_losses = {"eval_loss": torch.nn.CrossEntropyLoss()}

    """ METRIC """
    train_metrics = None
    eval_metrics = {"eval_metric": MulticlassAccuracy(num_classes=1000, top_k=1)}

    """ CLIKA """
    if _resume is True:
        clika_resume(
            clika_state_path=str(args.ckpt),
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
