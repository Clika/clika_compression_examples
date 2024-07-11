import argparse
import sys
import warnings
from functools import partial
from pathlib import Path
from pprint import pprint

import torch
import torchvision.transforms.functional as TF
from clika_compression import Settings, TensorBoardCallback, clika_compress, clika_resume
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import PeakSignalNoiseRatio

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "IMDN"))
from data.DIV2K import div2k
from data.Set5_val import DatasetFromFolderVal
from model.architecture import IMDN
from model.block import CCALayer
from utils import load_state_dict


class FakeOpt(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Define Class/Function Wrappers
# ==================================================================================================================== #
def replace_CCALayer(model):
    """
    Currently not supported operation (0.3.0)
    Replace with identity layer

    REFERENCE:
    https://github.com/Zheng222/IMDN/blob/8f158e6a5ac9db6e5857d9159fd4a6c4214da574/model/block.py#L82-L86
    """
    for _, m in model.named_children():
        if isinstance(m, CCALayer):
            m.contrast = torch.nn.Identity()
        else:
            replace_CCALayer(m)


def get_loader(data_dir: Path, batch_size: int, workers: int, scale=4, eval_data="REDS", train=False) -> DataLoader:
    """Factory function that generates train/eval Dataloader."""
    if train:
        kwargs = {
            "scale": scale,
            "root": str(data_dir / "div2k"),
            "ext": ".png",
            "n_colors": 3,
            "rgb_range": 1,
            "n_train": 800,  # num of training images
            "patch_size": 192,
            "phase": "train",
            "test_every": 1000,
            "batch_size": batch_size,
        }
        opt = FakeOpt(**kwargs)
        dataset = div2k(opt)

        loader = DataLoader(
            dataset,
            batch_size,
            shuffle=True,
            num_workers=workers,
            drop_last=True,
        )
    else:
        if eval_data == "REDS":
            dataset = REDS(str(data_dir / "REDS4"), scale)
            loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=workers, pin_memory=False)
        elif eval_data == "Set5":
            dataset = DatasetFromFolderVal("IMDN/Test_Datasets/Set5/", f"IMDN/Test_Datasets/Set5_LR/x{scale}", scale)
            loader = DataLoader(dataset, 1, shuffle=False, num_workers=workers, pin_memory=False)
        else:
            raise ValueError(f"Not supported dataset `{eval_data}`")
    return loader


class REDS(Dataset):
    """REDS evaluation dataset
    x4 scaling only (x2, x3 not available)
    """

    def __init__(self, data_dir: str, scale=4):
        """
        :param data_dir: test data folder
        """
        assert scale == 4, "Only x4 scaling support for REDS"

        data_dir = Path(data_dir)
        dir_hr = data_dir.joinpath("GT")
        self.dir_hr = sorted((str(f) for f in dir_hr.rglob("*.png")))
        dir_lr = data_dir.joinpath("sharp_bicubic")
        self.dir_lr = sorted((str(f) for f in dir_lr.rglob("*.png")))

    def __getitem__(self, idx):
        """Parse dataloader output to satisfy CLIKA SDK input requirements.
        (https://docs.clika.io/docs/next/compression-constrains/cco_inputs_requirements#Dataset-Dataloader)
        """
        hr_img = Image.open(self.dir_hr[idx])
        lr_img = Image.open(self.dir_lr[idx])

        return TF.to_tensor(lr_img), TF.to_tensor(hr_img)

    def __len__(self):
        return len(self.dir_hr)


# ==================================================================================================================== #


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="CLIKA IMDN Example")

    ace_parser = parser.add_argument_group("CLIKA ACE configuration")
    ace_parser.add_argument("--config", type=str, default="config.yml", help="ACE config yaml path")

    model_parser = parser.add_argument_group("Model configuration")
    model_parser.add_argument("--output_dir", type=Path, default="outputs", help="Path to save clika related files for the SDK")
    model_parser.add_argument("--data", type=Path, default="dataset", help="Dataset directory")
    model_parser.add_argument("--eval_data", type=str, choices=["REDS", "Set5"], default="REDS", help="Evaluation dataset to use (default: REDS)")
    model_parser.add_argument("--ckpt", type=Path, default="IMDN/checkpoints/IMDN_x4.pth", help="Path to load the model checkpoints (e.g. .pth, .pompom)")
    model_parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation (default: 16)")
    model_parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    model_parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for data loading (default: 4)")
    model_parser = parser.add_argument_group("Additional arguments")
    model_parser.add_argument("--model_type", type=int, choices=[2, 3, 4], default=4, help="Super resolution scale size (default: 4)")

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
    model = IMDN(upscale=args.model_type)

    _resume = False
    if (_train_from_scratch is False) and (args.ckpt is not None):
        if args.ckpt.suffix == ".pompom":
            print("[INFO] .pompom file provided as ckpt, resuming compression ...")
            _resume = True
        else:
            print(f"[INFO] loading custom ckpt from {args.ckpt}")
            state_dict = load_state_dict(str(args.ckpt))
            model.load_state_dict(state_dict)

    replace_CCALayer(model=model)
    torch.fx.symbolic_trace(model)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    """
    Define Dataloaders
    ====================================================================================================================
    """
    _get_loader_baseline = partial(get_loader, data_dir=args.data, batch_size=args.batch_size, workers=args.workers)
    get_train_loader = partial(_get_loader_baseline, scale=args.model_type, train=True)
    get_eval_loader = partial(_get_loader_baseline, scale=args.model_type, eval_data=args.eval_data, train=False)

    """
    Define Metric Wrapper
    ====================================================================================================================
    """
    train_metrics = None
    eval_metrics = {"psnr": PeakSignalNoiseRatio()}

    """
    RUN Compression
    ====================================================================================================================
    """
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
