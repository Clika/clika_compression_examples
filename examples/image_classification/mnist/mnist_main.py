import argparse
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


# Define Class/Function Wrappers
# ==================================================================================================================== #


class BasicMNIST(torch.nn.Module):
    """
    ref: https://github.com/pytorch/examples/blob/7f7c222b355abd19ba03a7d4ba90f1092973cdbc/mnist/main.py#L11
    """

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
        x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 2), stride=None, padding=(0, 0), dilation=(1, 1))
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def custom_loss_fn(pred, target):
    return torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(pred, dim=1), target)


def get_loader(batch_size: int, num_workers=0, train=True) -> DataLoader:
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = torchvision.datasets.MNIST(
        root=str(BASE_DIR),
        train=train,
        transform=transform,
        target_transform=None,
        download=True,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=True,
    )
    return data_loader


class MetricWrapper(MulticlassAccuracy):
    def compute(self) -> torch.Tensor:
        results = super().compute()
        return results * 100.0


# ==================================================================================================================== #


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="CLIKA MNIST Example")

    ace_parser = parser.add_argument_group("CLIKA ACE configuration")
    ace_parser.add_argument("--config", type=str, default="config.yml", help="ACE config yaml path")

    model_parser = parser.add_argument_group("Model configuration")
    model_parser.add_argument("--output_dir", type=Path, default="outputs", help="Path to save clika related files for the SDK")
    model_parser.add_argument("--ckpt", type=Path, default=None, help="Path to load the model checkpoints (e.g. .pth, .pompom)")
    model_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation (default: 32)")
    model_parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for the optimizer (default: 1e-2)")
    model_parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for data loading (default: 4)")

    args = parser.parse_args()
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
    model = BasicMNIST()

    _resume = False
    if (_train_from_scratch is False) and (args.ckpt is not None):
        if args.ckpt.suffix == ".pompom":
            print("[INFO] .pompom file provided as ckpt, resuming compression ...")
            _resume = True
        else:
            print(f"[INFO] loading custom ckpt from {args.ckpt}")
            state_dict = torch.load(args.ckpt)
            model.load_state_dict(state_dict)

    """
    Define Loss Function
    ====================================================================================================================
    """
    train_losses = {"NLL_loss": custom_loss_fn}
    eval_losses = {"NLL_loss": custom_loss_fn}

    """
    Define Optimizer
    ====================================================================================================================
    """
    optimizer = torch.optim.Adadelta(params=model.parameters(), lr=args.lr)

    """
    Define Dataloaders
    ====================================================================================================================
    """
    get_train_loader = partial(get_loader, args.batch_size, args.workers, True)
    get_eval_loader = partial(get_loader, args.batch_size, args.workers, False)

    """
    Define Metric Wrapper
    ====================================================================================================================
    """
    train_metrics = eval_metrics = {"acc": MetricWrapper(num_classes=10)}

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
