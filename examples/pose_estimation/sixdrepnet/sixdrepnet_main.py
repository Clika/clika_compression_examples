import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Union

import lightning as pl
import torch
import torchvision
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
from torch import distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.optim import Optimizer
from torchmetrics import SumMetric

BASE_DIR = Path(__file__).parent

from clika_ace import ClikaModule
from example_utils.common import dist_utils as utils
from example_utils.common.pl_callbacks import RichModelSummary, TQDMProgressBar
from example_utils.common.pl_utils import tune_batch_size
sys.path.insert(0, str(BASE_DIR.joinpath("6DRepNet")))
from sixdrepnet.datasets import getDataset
from sixdrepnet.loss import GeodesicLoss
from sixdrepnet.model import SixDRepNet


class Pose300W_AFLW2000_DataModule(pl.LightningDataModule):
    _args: argparse.Namespace
    train_batch_size: int
    test_batch_size: int
    _train_loader: Optional[torch.utils.data.DataLoader]
    _test_loader: Optional[torch.utils.data.DataLoader]

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._args = args
        self.train_batch_size = self._args.batch_size
        self.test_batch_size = self._args.batch_size
        self._train_loader = None
        self._test_loader = None

    @staticmethod
    def create_dataloaders(
        args: argparse.Namespace, batch_size: int, is_train: bool = False
    ) -> torch.utils.data.DataLoader:
        """Initialize dataloaders (train, eval)"""
        if is_train:
            dataset_name = "Pose_300W_LP"
            transform = [torchvision.transforms.RandomResizedCrop(size=224, scale=(0.8, 1))]
            data_dir = args.data_path.joinpath("300W_LP")
        else:
            dataset_name = "AFLW2000"
            transform = [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
            ]
            data_dir = args.data_path.joinpath("AFLW2000")
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

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size if is_train else 1,  # TODO: batched evaluation
            num_workers=args.workers,
            pin_memory=False,
            shuffle=True if is_train else False,
        )
        return data_loader

    def train_dataloader(self):
        if self._train_loader is None or self._train_loader.batch_size != self.train_batch_size:
            self._train_loader = self.create_dataloaders(
                args=self._args, batch_size=self.train_batch_size, is_train=True
            )
        return self._train_loader

    def test_dataloader(self):
        if self._test_loader is None or self._test_loader.batch_size != self.test_batch_size:
            self._test_loader = self.create_dataloaders(
                args=self._args, batch_size=self.test_batch_size, is_train=False
            )
        return self._test_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()


class SixDRepNetModule(pl.LightningModule):
    _args: argparse.Namespace
    _model: torch.nn.Module
    _total_sum_metric: SumMetric
    _loss_fn: GeodesicLoss
    BEST_CKPT_METRIC_NAME: str = "val/loss_total"
    BEST_CKPT_METRIC_NAME_MODE: str = "min"
    _grad_scaler: torch.cuda.amp.GradScaler
    _autocast_ctx: torch.cuda.amp.autocast

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._args = args
        self.automatic_optimization = False

        """MODEL"""
        self._model = SixDRepNet(
            backbone_name="RepVGG-B1g2",
            backbone_file=str(BASE_DIR.joinpath("checkpoints", "6DRepNet_300W_LP_AFLW2000.pth")),
            deploy=True,
            pretrained=False,
        ).to(self._args.device)

        """METRIC"""
        self._total_sum_metric = SumMetric()

        """LOSS"""
        self._loss_fn = GeodesicLoss()

        """ETC"""
        self.save_hyperparameters(vars(args))

        # we create the GradScaler with enabled=True so that it initializes the internal vars
        if utils.is_dist_avail_and_initialized():
            self._grad_scaler = ShardedGradScaler(enabled=True)
        else:
            self._grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        self._grad_scaler._enabled = False
        self._autocast_ctx = nullcontext()
        if self._args.amp is not None:
            if self._args.amp == "fp16":
                self._autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16, cache_enabled=False)
                self._grad_scaler._enabled = True
            else:
                # bf16
                self._autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16, cache_enabled=False)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._model.zero_grad(set_to_none)

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        """Return training loss"""
        xs, ys, _, _ = batch
        xs = xs.cuda()
        ys = ys.cuda()
        with self._autocast_ctx:
            y_hat = self._model(xs)
            loss = self._loss_fn(y_hat, ys)
        self.log("train/loss", loss.detach(), prog_bar=True)
        self.manual_backward(loss)
        return loss

    def test_step(self, batch, batch_idx: int, prefix: str = "test") -> STEP_OUTPUT:
        """Collect logits and postprocess"""
        xs, ys, _, _ = batch
        xs = xs.cuda()
        ys = ys.cuda()
        with self._autocast_ctx:
            y_hat = self._model(xs)
            loss = self._loss_fn(y_hat, ys)
        self._total_sum_metric.update(loss)
        return None

    def validation_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return self.test_step(*args, **kwargs, prefix="val")

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        """Clip gradients attached to `model.parameters()`"""
        if gradient_clip_val is None:
            return

        gradient_clip_algorithm = gradient_clip_algorithm or "norm"  # type: Optional[Literal["norm", "value"]]
        if gradient_clip_algorithm == "norm":
            norm_type = 2.0
        elif gradient_clip_algorithm == "value":
            norm_type = 1.0

        if hasattr(self._model, "clip_grad_norm_"):  # FSDP
            norm: torch.Tensor = self._model.clip_grad_norm_(max_norm=gradient_clip_val, norm_type=norm_type)
        else:  # DDP, SingleGPU
            norm: torch.Tensor = torch.nn.utils.clip_grad_norm_(
                parameters=self._model.parameters(), max_norm=gradient_clip_val, norm_type=norm_type
            )
        self.log("train/grad_norm", norm, prog_bar=True)

    def manual_backward(self, loss: torch.Tensor, **kwargs) -> None:
        optimizer = self.optimizers(use_pl_optimizer=True)
        loss_scaled = self._grad_scaler.scale(loss)  # scale fp16 loss
        loss_scaled.backward()
        self._grad_scaler.unscale_(optimizer)  # unscale gradients inside optimizer
        self.configure_gradient_clipping(None, self._args.clip_grad_norm)  # clip unscaled gradients
        self._grad_scaler.step(optimizer)  # optimizer.step()
        self._grad_scaler.update()  # update grad scaler

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        """Called in the training loop before anything happens for that batch"""
        self.zero_grad()

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """Called in the training loop after the batch"""
        optimizer = self.optimizers(use_pl_optimizer=True)
        scheduler = self.lr_schedulers()
        self.zero_grad(set_to_none=True)
        scheduler.step()
        lr: float = [g["lr"] for g in optimizer.param_groups][0]
        self.log("train/lr", lr, prog_bar=True)

    def on_train_epoch_start(self) -> None:
        """Called in the training loop at the very beginning of the epoch"""
        pass

    def on_train_epoch_end(self) -> None:
        """Called in the training loop at the very end of the epoch"""
        pass

    def on_test_epoch_start(self) -> None:
        """Reset metrics before the test loop begin"""
        self._total_sum_metric.reset()

    def on_test_epoch_end(self, prefix: str = "test") -> None:
        """Compute metrics"""
        _avg: torch.Tensor = self._total_sum_metric.compute() / self._total_sum_metric.update_count
        self.log(f"{prefix}/loss_total", _avg, on_epoch=True)
        self._total_sum_metric.reset()

    def on_validation_epoch_start(self) -> None:
        """Called in the validation loop at the very beginning of the epoch"""
        self.on_test_epoch_start()

    def on_validation_epoch_end(self):
        """Called in the validation loop at the very end of the epoch"""
        self.on_test_epoch_end("val")

    def configure_optimizers(self):
        """Configure optimizer and LRScheduler used for training"""
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._args.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._args.steps_per_epoch // 2, eta_min=self._args.lr * 0.75
        )
        return [optimizer], [lr_scheduler]

    def initialize_clika(self, data_module: Pose300W_AFLW2000_DataModule) -> None:
        """Initialize CLIKAModule - Wrap `nn.Module` with `ClikaModule`"""
        train_dataloader = data_module.train_dataloader()
        example_inputs = next(iter(train_dataloader))[0]
        self._model.to(self._args.device)
        self._model: ClikaModule = torch.compile(
            self._model,
            backend="clika",
            options={
                "settings": self._args.clika_config,
                "example_inputs": example_inputs,
                "train_dataloader": train_dataloader,
                "discard_input_model": True,
                "logs_dir": os.path.join(self._args.output_dir, "logs"),
                "apply_on_data_fn": lambda x: x[0],
            },
        )
        self._model.clika_visualize(os.path.join(self._args.output_dir, f"sixdrepnet.svg"))
        state_dict = self._model.clika_serialize()
        if utils.is_main_process():
            torch.save(state_dict, os.path.join(self._args.output_dir, f"sixdrepent_init.pompom"))
        torch.onnx.export(
            model=self._model,
            args=example_inputs.cuda(),
            f=os.path.join(self._args.output_dir, f"sixdrepent_init.onnx"),
            input_names=["x"],
            output_names=["out"],
            dynamic_axes={
                "x": {
                    0: "batch_size",
                    2: "W",
                    3: "H",
                }
            },
        )


def evaluate_original(trainer: pl.Trainer, module: SixDRepNetModule, data_module: Pose300W_AFLW2000_DataModule) -> dict:
    results = trainer.test(model=module, datamodule=data_module, verbose=True if utils.is_main_process() else False)[0]
    results = {f"original/{k}": v for k, v in results.items()}
    return results


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="CLIKA 6dREPNET Example")

    ace_parser = parser.add_argument_group("General configuration")
    ace_parser.add_argument("--clika_config", type=Path, default="local/clika_config.yaml", help="ACE config yaml path")
    ace_parser.add_argument("--output_dir", type=Path, default=None, help="Path to save clika related files for the SDK")
    ace_parser.add_argument("--data_path", type=Path, default=BASE_DIR.joinpath("6DRepNet", "sixdrepnet", "datasets"), help="Dataset directory")
    ace_parser.add_argument("--resume", type=Path, default=None, help="Path to load the model checkpoints (e.g. .pth)")

    train_parser = parser.add_argument_group("Train configuration")
    train_parser.add_argument("--epochs", type=int, default=200, help="Number of epochs (default 200)")
    train_parser.add_argument("--steps_per_epoch", type=int, default=100, help="Steps per epoch (default 100)")
    train_parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Train, Evaluation batch size")
    train_parser.add_argument("--workers", type=int, default=3, help="Number of worker processes for data loading (default: 3)")
    train_parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for the optimizer")
    train_parser.add_argument("--amp", type=str, default=None, choices=["fp16", "bf16", None], help="autocast dtype")
    train_parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Max grad norm (0 to disable)")

    eval_parser = parser.add_argument_group("Evaluation configuration")
    # TODO: save eval visualization folder

    gpu_parser = parser.add_argument_group("GPU configuration")
    gpu_parser.add_argument("--gpu", type=int, default=0, help="GPU id to run on")
    gpu_parser.add_argument("--world_size", default=1, type=int, help="Number of distributed processes")
    gpu_parser.add_argument("--dist_url", default="env://", type=str, help="Url used to set up distributed training")

    etc_parser = parser.add_argument_group("ETC configuration")
    etc_parser.add_argument("--random_seed", type=int, default=373737, help="Random Seed")
    etc_parser.add_argument("--use_deterministic_algorithms", action="store_true", help="Whether or not to use deterministic algorithms. will slow down training")
    etc_parser.add_argument("--print_freq", type=int, default=1, help="Printing frequency")
    etc_parser.add_argument("--dry_run", action="store_true", help="Whether to run the initial calibration without further fine tuning")

    args = parser.parse_args()

    pl.seed_everything(args.random_seed)
    args.clip_grad_norm = None if args.clip_grad_norm == 0 else args.clip_grad_norm

    if not args.data_path.exists():
        raise FileNotFoundError(f"Cannot locate data directory at: `{args.data_path}`")
    if not args.clika_config.exists():
        raise FileNotFoundError(f"Cannot locate clika config at: `{args.clika_config}`")
    with open(args.clika_config, "r") as fp:
        args.clika_config_yaml = yaml.safe_load(fp)
    args.clika_config = str(args.clika_config)

    args.output_dir = args.output_dir or BASE_DIR.joinpath("outputs")
    args.output_dir = os.path.join(args.output_dir, args.clika_config_yaml["deployment_settings"]["target_framework"])
    utils.init_distributed_mode(args)
    if not utils.is_dist_avail_and_initialized():
        torch.cuda.set_device(args.gpu)
    if utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"

    return args


def main():
    args = parse_args()
    module = SixDRepNetModule(args=args)
    data_module = Pose300W_AFLW2000_DataModule(args)

    callbacks: list = [
        ModelCheckpoint(
            monitor=module.BEST_CKPT_METRIC_NAME,
            mode=module.BEST_CKPT_METRIC_NAME_MODE,
            dirpath=args.output_dir,
            save_on_train_epoch_end=True,
            filename="epoch-{epoch}-loss_total-{val/loss_total:3.5f}",
            save_top_k=1,
            auto_insert_metric_name=False,
            verbose=True,
        ),
        RichModelSummary(),
    ]
    tqdm_progress_bar_callback: Optional[TQDMProgressBar] = None
    if utils.is_main_process():
        tqdm_progress_bar_callback = TQDMProgressBar()
        callbacks.append(tqdm_progress_bar_callback)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        limit_train_batches=args.steps_per_epoch,
        precision=None,
        deterministic=args.use_deterministic_algorithms,
        accelerator="gpu" if "cuda" in args.device else args.device,
        devices=[args.gpu] if torch.cuda.is_available() else "auto",
        log_every_n_steps=args.print_freq,
        enable_progress_bar=True if utils.is_main_process() else False,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
    )
    trainer.print(f"Args {args}\n")
    original_test_results: Optional[dict] = None
    if not args.dry_run:
        original_test_results = evaluate_original(trainer=trainer, module=module, data_module=data_module)
    module.initialize_clika(data_module)
    if args.dry_run:
        return
    if tqdm_progress_bar_callback is not None and original_test_results is not None:
        tqdm_progress_bar_callback.original_test_results = original_test_results
    utils.override_pl_strategy(trainer, args)  # we override the strategy so Distributed Training works properly with PL
    trainer.fit(model=module, datamodule=data_module, ckpt_path=args.resume)
    if utils.is_dist_avail_and_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
