import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal, Optional, Union

import lightning as pl
import torch
import yaml
from example_utils.imdn_dataset import REDS
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
from torch import distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.optim import Optimizer
from torchmetrics.image import PeakSignalNoiseRatio

BASE_DIR = Path(__file__).parent

sys.path.insert(0, str(BASE_DIR.joinpath("IMDN")))
from clika_ace import ClikaModule
from example_utils.common import dist_utils as utils
from example_utils.common.pl_callbacks import RichModelSummary, TQDMProgressBar
from example_utils.common.pl_utils import tune_batch_size
from IMDN.data.DIV2K import div2k
from IMDN.data.Set5_val import DatasetFromFolderVal
from IMDN.model.architecture import IMDN
from IMDN.utils import load_state_dict


class IMDNModule(pl.LightningModule):
    _args: argparse.Namespace
    _model: torch.nn.Module
    _psnr_metric: PeakSignalNoiseRatio
    BEST_CKPT_METRIC_NAME: str = "val/psnr"
    BEST_CKPT_METRIC_NAME_MODE: str = "max"
    _grad_scaler: torch.cuda.amp.GradScaler
    _autocast_ctx: torch.cuda.amp.autocast
    train_batch_size: int  # for the tuner
    test_batch_size: int  # for the tuner
    _train_loader: Optional[torch.utils.data.DataLoader]
    _test_loader: Optional[torch.utils.data.DataLoader]

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.automatic_optimization = False
        self._args = args
        self.train_batch_size = self._args.batch_size
        self.test_batch_size = self._args.batch_size
        self._train_loader = None
        self._test_loader = None

        self._model = IMDN(upscale=args.scaling).to(self._args.device)
        self._psnr_metric = PeakSignalNoiseRatio()
        self.save_hyperparameters(vars(args))
        if utils.is_dist_avail_and_initialized():
            self._grad_scaler = ShardedGradScaler(enabled=True)
        else:
            self._grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        self._grad_scaler._enabled = (
            False  # we created the GradScaler with enabled=True so it initializes the internal vars
        )
        self._autocast_ctx = nullcontext()
        if self._args.amp is not None:
            if self._args.amp == "fp16":
                self._autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16, cache_enabled=False)
                self._grad_scaler._enabled = True
            else:  # bf16
                self._autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16, cache_enabled=False)

    @staticmethod
    def get_loader(args: argparse.Namespace, batch_size: int, train: bool = False) -> torch.utils.data.DataLoader:
        """Factory function that generates train/eval Dataloader."""
        if train:
            kwargs = {
                "scale": args.scaling,
                "root": str(args.data_path / "div2k"),
                "ext": ".png",
                "n_colors": 3,
                "rgb_range": 1,
                "n_train": 800,  # num of training images
                "patch_size": 192,
                "phase": "train",
                "test_every": 1000,
                "batch_size": args.batch_size,
            }
            opt = argparse.Namespace(**kwargs)
            dataset = div2k(opt)

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size,
                shuffle=True,
                num_workers=args.workers,
                drop_last=True,
            )
        else:
            if args.eval_data == "REDS":
                dataset = REDS(str(args.data_path / "REDS4"), args.scaling)
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size, shuffle=False, num_workers=args.workers, pin_memory=False
                )
            elif args.eval_data == "Set5":
                dataset = DatasetFromFolderVal(
                    "IMDN/Test_Datasets/Set5/", f"IMDN/Test_Datasets/Set5_LR/x{args.scaling}", args.scaling
                )
                loader = torch.utils.data.DataLoader(
                    dataset, 1, shuffle=False, num_workers=args.workers, pin_memory=False
                )
            else:
                raise ValueError(f"Not supported dataset `{args.eval_data}`")
        return loader

    def train_dataloader(self):
        if self._train_loader is None or self._train_loader.batch_size != self.train_batch_size:
            self._train_loader = self.get_loader(args=self._args, batch_size=self.train_batch_size, train=True)
        return self._train_loader

    def test_dataloader(self):
        if self._test_loader is None or self._test_loader.batch_size != self.test_batch_size:
            self._test_loader = self.get_loader(args=self._args, batch_size=self.test_batch_size, train=False)
        return self._test_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._model.zero_grad(set_to_none)

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        xs, ys = batch
        xs = xs.cuda()
        ys = ys.cuda()
        with self._autocast_ctx:
            y_hat = self._model(xs)
            loss = torch.nn.functional.l1_loss(y_hat, ys)

        self.log("train/loss", loss.detach(), prog_bar=True)
        self.manual_backward(loss)
        return loss

    def test_step(self, batch, batch_idx: int, prefix: str = "test") -> STEP_OUTPUT:
        xs, ys = batch
        xs = xs.cuda()
        ys = ys.cuda()
        with self._autocast_ctx:
            y_hat = self._model(xs)
            # loss = torch.nn.functional.l1_loss(y_hat, ys)
        self._psnr_metric.update(y_hat, ys)
        return None

    def validation_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return self.test_step(*args, **kwargs, prefix="val")

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        self.zero_grad()
        return None

    def manual_backward(self, loss: torch.Tensor, **kwargs) -> None:
        optimizer = self.optimizers(use_pl_optimizer=True)
        loss_scaled = self._grad_scaler.scale(loss)
        loss_scaled.backward()
        self._grad_scaler.unscale_(optimizer)
        self.clip_gradients(optimizer, None, None)
        self._grad_scaler.step(optimizer)
        self._grad_scaler.update()

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        optimizer = self.optimizers(use_pl_optimizer=True)
        scheduler = self.lr_schedulers()
        self.zero_grad(set_to_none=True)
        scheduler.step()
        lr: float = [g["lr"] for g in optimizer.param_groups][0]
        self.log("train/lr", lr, prog_bar=True)

    def on_train_epoch_start(self) -> None:
        pass

    def on_train_epoch_end(self) -> None:
        pass

    def on_test_epoch_start(self) -> None:
        self._psnr_metric.reset()

    def on_test_epoch_end(self, prefix: str = "test") -> None:
        self.log(f"{prefix}/psnr", self._psnr_metric.compute(), on_epoch=True)
        self._psnr_metric.reset()

    def on_validation_epoch_start(self) -> None:
        self.on_test_epoch_start()

    def on_validation_epoch_end(self):
        self.on_test_epoch_end("val")

    def clip_gradients(
        self,
        optimizer: Optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        clip_grad_norm: Optional[float] = getattr(self._args, "clip_grad_norm", None)
        if clip_grad_norm is None:
            return
        if hasattr(self._model, "clip_grad_norm_"):
            norm: torch.Tensor = self._model.clip_grad_norm_(max_norm=clip_grad_norm, norm_type=2.0)
        else:
            norm: torch.Tensor = torch.nn.utils.clip_grad_norm_(
                parameters=self._model.parameters(), max_norm=clip_grad_norm, norm_type=2.0
            )
        self.log("train/grad_norm", norm, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._args.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._args.steps_per_epoch // 2, eta_min=self._args.lr * 0.75
        )
        return [optimizer], [lr_scheduler]

    def initialize_clika(self) -> None:
        train_dataloader = self.train_dataloader()
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
        self._model.clika_visualize(os.path.join(self._args.output_dir, f"imdn.svg"))
        state_dict = self._model.clika_serialize()
        if utils.is_main_process():
            torch.save(state_dict, os.path.join(self._args.output_dir, f"imdn_init.pompom"))
        torch.onnx.export(
            model=self._model,
            args=example_inputs.cuda(),
            f=os.path.join(self._args.output_dir, f"imdn_init.onnx"),
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


def evaluate_original(trainer: pl.Trainer, module: IMDNModule) -> dict:
    results = trainer.test(model=module, verbose=True if utils.is_main_process() else False)[0]
    results = {f"original/{k}": v for k, v in results.items()}
    return results


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="CLIKA IMDN Example")
    parser.add_argument("--clika_config", type=Path, default="local/clika_config.yaml", help="Path to save clika related files for the SDK")
    parser.add_argument("--output_dir", type=Path, default=None, help="Path to save clika related files for the SDK")
    parser.add_argument("--data_path", type=Path, default="dataset", help="Dataset directory")
    parser.add_argument("--eval_data", type=str, choices=["REDS", "Set5"], default="REDS", help="Evaluation dataset to use (default: REDS)")
    parser.add_argument("--ckpt", type=Path, default="IMDN/checkpoints/IMDN_x4.pth", help="Path to load the model checkpoints (e.g. .pth, .pompom)")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Steps per epoch (default 100)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs (default 200)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size, will auto-tune for bigger batch size")
    parser.add_argument("--resume", type=Path, default=None, help="Path to load the model checkpoints (e.g. .pth)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for the optimizer")
    parser.add_argument("--workers", type=int, default=3, help="Number of worker processes for data loading (default: 3)")
    parser.add_argument("--random_seed", type=int, default=373737, help="Random Seed")
    parser.add_argument("--use_deterministic_algorithms", action="store_true", help="whether or not to use deterministic algorithms. will slow down training")
    parser.add_argument("--print_freq", type=int, default=1, help="printing frequency")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="max grad norm")
    parser.add_argument("--amp", type=str, default=None, choices=["fp16", "bf16", None], help="autocast dtype")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to run on")
    parser.add_argument("--dry_run", action="store_true", help="whether to run the initial calibration without further fine tuning")
    parser.add_argument("--lr_scheduler", type=str, default="cosineannealinglr", help="learning rate scheduler")
    parser.add_argument("--scaling", type=int, choices=[2, 3, 4], default=4, help="Super resolution scale size (default: 4)")

    args = parser.parse_args()

    if args.data_path.exists() is False:
        raise FileNotFoundError(f"Unknown directory: {args.data_path}")

    if args.output_dir is None:
        args.output_dir = BASE_DIR.joinpath("outputs", f"imdn_x{args.scaling}")

    if not args.clika_config.exists():
        if not BASE_DIR.joinpath(args.clika_config).exists():
            raise FileNotFoundError(f"Cannot find clika config at: {args.clika_config}")
        args.clika_config = BASE_DIR.joinpath(args.clika_config)

    args.clika_config = str(args.clika_config)

    with open(args.clika_config, "r") as fp:
        args.clika_config_yaml = yaml.safe_load(fp)

    utils.init_distributed_mode(args)
    args.output_dir = os.path.join(args.output_dir, args.clika_config_yaml["deployment_settings"]["target_framework"])
    if utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    pl.seed_everything(args.random_seed)
    if not utils.is_dist_avail_and_initialized():
        torch.cuda.set_device(args.gpu)

    if torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"

    return args


def main():
    args = parse_args()
    module = IMDNModule(args=args)
    if args.ckpt and args.ckpt.exists():
        print(f"[INFO] loading custom ckpt from {args.ckpt}")
        state_dict = load_state_dict(str(args.ckpt))
        module._model.load_state_dict(state_dict)

    callbacks: list = [
        ModelCheckpoint(
            monitor=module.BEST_CKPT_METRIC_NAME,
            mode=module.BEST_CKPT_METRIC_NAME_MODE,
            dirpath=args.output_dir,
            save_on_train_epoch_end=True,
            filename="epoch-{epoch}-psnr-{val/psnr:3.5f}",
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
    if utils.is_main_process():
        trainer.print(f"Args {args}\n")
    original_test_results: Optional[dict] = None
    if not args.dry_run:
        original_test_results = evaluate_original(trainer=trainer, module=module)
    module.initialize_clika()
    if args.dry_run:
        return
    if tqdm_progress_bar_callback is not None and original_test_results is not None:
        tqdm_progress_bar_callback.original_test_results = original_test_results
    utils.override_pl_strategy(trainer, args)  # we override the strategy so Distributed Training works properly with PL
    trainer.fit(model=module, ckpt_path=args.resume)
    if utils.is_dist_avail_and_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
