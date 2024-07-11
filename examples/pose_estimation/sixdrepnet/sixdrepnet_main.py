import argparse
import shutil
import sys
import warnings
from functools import partial
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
import torch
import torchvision
from clika_compression import Settings, TensorBoardCallback, clika_compress, clika_resume
from torch.utils.data import DataLoader
from torchmetrics import Metric
from typing_extensions import Optional

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "6DRepNet" / "sixdrepnet"))
import utils
from datasets import getDataset
from loss import GeodesicLoss
from model import SixDRepNet


# Define Class/Function Wrappers
# ==================================================================================================================== #
def _collate_fn(batch):
    """Parse dataloader output to satisfy CLIKA SDK input requirements.
    (https://docs.clika.io/docs/compression-constraints/cco_input_requirements#Dataset-Dataloader)
    """
    images, r_label, cont_labels, name = zip(*batch)
    if isinstance(cont_labels[0], list):
        cont_labels = [torch.tensor(_) for _ in cont_labels]
    return torch.stack(images), (torch.stack(r_label), torch.stack(cont_labels), name)


def get_loader(data_dir: Path, batch_size: int, num_workers: int = 0, is_train: bool = True) -> DataLoader:
    """Factory function that generates train/eval Dataloader."""
    if is_train:
        dataset_name = "Pose_300W_LP"
        transform = [torchvision.transforms.RandomResizedCrop(size=224, scale=(0.8, 1))]
        data_dir = data_dir.joinpath("300W_LP")
    else:
        dataset_name = "AFLW2000"
        transform = [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
        ]
        data_dir = data_dir.joinpath("AFLW2000")
    transform += [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    transform = torchvision.transforms.Compose(transform)

    dataset = getDataset(
        dataset=dataset_name,
        data_dir=str(data_dir),
        filename_list=str(data_dir.joinpath("files.txt")),
        transformations=transform,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size if is_train else 1,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=True if is_train else False,
        collate_fn=_collate_fn,
    )
    return data_loader


class SixDRepNetWrapper(SixDRepNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """
        `utils.compute_rotation_matrix_from_ortho6d` inside forward graph is moved to `CriterionWrapper`

        REFERENCE:
        https://github.com/thohemp/6DRepNet/blob/0d4ccab11f49143f3e4638890d0f307f30b070f4/sixdrepnet/model.py#L37-L45
        """
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        return x


class CriterionWrapper:
    def __init__(self):
        self.criterion = GeodesicLoss().cuda()

    def __call__(self, predictions, targets):
        predictions = utils.compute_rotation_matrix_from_ortho6d(predictions)
        return self.criterion(targets[0], predictions)


class MetricWrapper(Metric):
    """Custom Metric class that postprocess model's logit outputs and compute yaw, pitch, roll error.

    REFERENCE
    https://github.com/natanielruiz/deep-head-pose/blob/f7bbb9981c2953c2eca67748d6492a64c8243946/code/test_hopenet.py#L100
    """

    def __init__(self, visualize_dir: Optional[str] = None, **kwargs):
        """
        Args:
            visualize_dir (Optional[str], optional): the flag to save visualization result
            to ./outputs/visualization/epoch{epoch}/*.jpg. Defaults to None.
        **kwargs:
            additional keyword arguments about `torchmetric.Metric`
        """
        super().__init__(**kwargs)
        self.visualize_dir = visualize_dir
        if visualize_dir is None:
            self._visualize = False
        else:
            self._visualize = True

        self.epoch = 0
        self.save_dir = self._get_save_dir()
        if self.save_dir.exists():
            shutil.rmtree(str(self.save_dir))  # reset every run
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("yaw_error", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("pitch_error", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("roll_error", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def _get_save_dir(self):
        return BASE_DIR / Path(f"{self.visualize_dir}/epoch{self.epoch}")

    def _cal_err(self, pred_deg, gt_deg):
        _stacked = torch.stack(
            (
                torch.abs(gt_deg - pred_deg),
                torch.abs(pred_deg + 360 - gt_deg),
                torch.abs(pred_deg - 360 - gt_deg),
                torch.abs(pred_deg + 180 - gt_deg),
                torch.abs(pred_deg - 180 - gt_deg),
            )
        )
        return torch.sum(torch.min(_stacked, 0)[0])

    def update(self, outputs: torch.Tensor, targets: tuple) -> None:
        _device = outputs.device
        R_pred = utils.compute_rotation_matrix_from_ortho6d(outputs)
        r_label, cont_labels, name = targets
        r_label = r_label.to(_device)
        cont_labels = cont_labels.to(_device)

        self.total += cont_labels.size(0)

        # gt euler
        y_gt_deg = cont_labels[:, 0].float() * 180 / torch.pi
        p_gt_deg = cont_labels[:, 1].float() * 180 / torch.pi
        r_gt_deg = cont_labels[:, 2].float() * 180 / torch.pi

        euler = utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
        p_pred_deg = euler[:, 0]
        y_pred_deg = euler[:, 1]
        r_pred_deg = euler[:, 2]

        _pitch_error = self._cal_err(p_pred_deg, p_gt_deg).to(_device)
        _yaw_error = self._cal_err(y_pred_deg, y_gt_deg).to(_device)
        _roll_error = self._cal_err(r_pred_deg, r_gt_deg).to(_device)
        self.pitch_error += _pitch_error
        self.yaw_error += _yaw_error
        self.roll_error += _roll_error

        if self._visualize:
            _idx = 0
            img = cv2.imread(f"{BASE_DIR}/6DRepNet/sixdrepnet/datasets/AFLW2000/{name[_idx]}.jpg")
            utils.draw_axis(img, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], tdx=200, tdy=200, size=100)
            _y = torch.abs(y_pred_deg - y_gt_deg).item()
            _p = torch.abs(p_pred_deg - p_gt_deg).item()
            _r = torch.abs(r_pred_deg - r_gt_deg).item()
            _error_string = f"y {_y:.2f}, p {_p:.2f}, r {_r:.2f}"
            cv2.putText(
                img, _error_string, (30, img.shape[0] - 30), fontFace=1, fontScale=1, color=(0, 0, 255), thickness=2
            )
            while self.save_dir.exists() and len(list(self.save_dir.iterdir())) == 1969:
                self.epoch += 1
                self.save_dir = self._get_save_dir()
            self.save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.save_dir.joinpath(f"{name[_idx]}.jpg")), img)

    def compute(self):
        yaw = self.yaw_error / self.total
        pitch = self.pitch_error / self.total
        roll = self.roll_error / self.total
        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "total": yaw + pitch + roll,
        }


# ==================================================================================================================== #


def parse_args() -> argparse.Namespace:
    # fmt: off
    _model_name = "sixdrepnet"
    parser = argparse.ArgumentParser(description=f"CLIKA {_model_name} Example")

    ace_parser = parser.add_argument_group("CLIKA ACE configuration")
    ace_parser.add_argument("--config", type=str, default="config.yml", help="ACE config yaml path")

    model_parser = parser.add_argument_group("Model configuration")
    model_parser.add_argument("--output_dir", type=Path, default="outputs", help="Path to save clika related files for the SDK")
    model_parser.add_argument("--data", type=Path, default="6DRepNet/sixdrepnet/datasets", help="Dataset directory")
    model_parser.add_argument("--ckpt", type=Path, default="6DRepNet_300W_LP_AFLW2000.pth", help="Path to load the model checkpoints (e.g. .pth, .pompom)")
    model_parser.add_argument("--batch_size", type=int, default=80, help="Batch size for training and evaluation (default: 80)")
    model_parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    model_parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for data loading (default: 4)")

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

    model = SixDRepNetWrapper(
        backbone_name="RepVGG-B1g2",
        backbone_file=str(args.ckpt),
        deploy=True,
        pretrained=False,
    )
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
        get_loader,
        data_dir=args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        is_train=True,
    )
    get_eval_loader = partial(
        get_loader,
        data_dir=args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        is_train=False,
    )

    """
    Define Optimizer
    ====================================================================================================================
    """
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    """
    Define Loss Function
    ====================================================================================================================
    """
    criterion = CriterionWrapper()
    train_losses = {"train_loss": criterion}
    eval_losses = {"eval_loss": criterion}

    """
    Define Metric Wrapper
    ====================================================================================================================
    """
    metric_fn = MetricWrapper(visualize_dir=str(args.output_dir.joinpath("visualization")))
    train_metrics = None  # {"train(AFLW2000)": metric_fn}
    eval_metrics = {"eval_metric": metric_fn}

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
