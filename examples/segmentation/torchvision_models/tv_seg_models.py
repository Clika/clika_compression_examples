import argparse
import os
import sys
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Optional, Union

import lightning as pl
import torch
import torchmetrics
import torchvision
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
from torch import distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.optim import Optimizer

BASE_DIR = Path(__file__).parent

sys.path.insert(0, str(BASE_DIR.joinpath("torchvision_reference", "segmentation")))
from clika_ace import ClikaModule
from example_utils.common import dist_utils
from example_utils.common.pl_callbacks import RichModelSummary, TQDMProgressBar
from example_utils.common.pl_utils import tune_batch_size
from torchvision_reference.segmentation import utils
from torchvision_reference.segmentation.train import get_args_parser, get_dataset


class ConfusionMatrix(torchmetrics.Metric):
    num_classes: int

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        n: int = self.num_classes
        self.add_state("mat", default=torch.zeros((n, n), dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        n = self.num_classes
        with torch.inference_mode():
            k = (labels >= 0) & (labels < n)
            inds = n * labels[k].to(torch.int64) + preds[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu


class COCODataModule(pl.LightningDataModule):
    _args: argparse.Namespace
    train_batch_size: int
    _train_loader: Optional[torch.utils.data.DataLoader]
    _test_loader: Optional[torch.utils.data.DataLoader]

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._args = args
        self.train_batch_size = self._args.batch_size  # for the Tuner of Batch Size
        self._train_loader = None
        self._test_loader = None

    @staticmethod
    def create_dataloaders(args: argparse.Namespace, train_batch_size: int) -> tuple:
        print("Creating data loaders")
        with redirect_stdout(new_target=None), redirect_stderr(new_target=None):
            dataset, num_classes = get_dataset(is_train=True, args=args)
            dataset_test, _ = get_dataset(is_train=False, args=args)

        if dist_utils.is_dist_avail_and_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            collate_fn=utils.collate_fn,
            drop_last=True,
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
        )

        return data_loader, data_loader_test

    def setup(self, stage: str):
        train_loader, test_loader = COCODataModule.create_dataloaders(self._args, self.train_batch_size)
        self._train_loader = train_loader
        self._test_loader = test_loader

    def train_dataloader(self):
        if self._train_loader is None or self._train_loader.batch_size != self.train_batch_size:
            self.teardown()
            self.setup("")
        return self._train_loader

    def test_dataloader(self):
        if self._test_loader is None or self._test_loader.batch_size != 1:  # must be one
            self.teardown()
            self.setup("")
        return self._test_loader

    def val_dataloader(self):
        return self.test_dataloader()

    def teardown(self, **kwargs):
        del self._train_loader
        del self._test_loader
        self._train_loader = None
        self._test_loader = None


class COCOSegmentationModels(pl.LightningModule):
    _args: argparse.Namespace
    _model: torch.nn.Module
    _metric: ConfusionMatrix
    _ignore_value: int
    BEST_CKPT_METRIC_NAME: str = "val/pixel_acc"
    BEST_CKPT_METRIC_NAME_MODE: str = "max"
    _grad_scaler: torch.cuda.amp.GradScaler
    _autocast_ctx: torch.cuda.amp.autocast

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._args = args
        self.automatic_optimization = False

        """MODEL"""
        paths = {
            "voc": 21,
            "voc_aug": 21,
            "coco": 21,
        }
        num_classes = paths[args.dataset]
        self._model = torchvision.models.get_model(
            args.model,
            weights=args.weights,
            weights_backbone=args.weights_backbone,
            num_classes=num_classes,
            aux_loss=args.aux_loss,
        )

        """METRIC"""
        self._metric = ConfusionMatrix(num_classes=num_classes)
        self._ignore_value = 255

        """ETC"""
        self.save_hyperparameters(vars(args))

        # we created the GradScaler with enabled=True so it initializes the internal vars
        if dist_utils.is_dist_avail_and_initialized():
            self._grad_scaler = ShardedGradScaler(enabled=True)
        else:
            self._grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        self._grad_scaler._enabled = False
        self._autocast_ctx = nullcontext()
        if self._args.amp:
            self._autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16, cache_enabled=False)
            self._grad_scaler._enabled = True

    @staticmethod
    def _pack_outputs(outputs: Union[dict, tuple, list, torch.Tensor]) -> dict:
        if isinstance(outputs, torch.Tensor):
            return {"out": outputs}
        elif isinstance(outputs, (tuple, list)):
            return {k: v for k, v in zip(["out", "aux"], outputs)}
        else:
            return outputs

    def _compute_loss(self, outputs: dict, target) -> torch.Tensor:
        losses = {}
        for name, x in outputs.items():
            losses[name] = torch.nn.functional.cross_entropy(x, target, ignore_index=self._ignore_value)

        if len(losses) == 1:
            return losses["out"]

        return losses["out"] + 0.5 * losses["aux"]

    def _process_for_accuracy(self, outputs, ys):
        out_argmax = outputs.argmax(1)
        ys = ys.flatten()
        out_argmax = out_argmax.flatten()
        mask = ys != self._ignore_value
        ys = ys[mask]
        out_argmax = out_argmax[mask]
        return out_argmax, ys

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._model.zero_grad(set_to_none)

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        """Return training loss"""
        xs, ys = batch
        xs = xs.cuda()
        ys = ys.cuda()
        with self._autocast_ctx:
            outputs = self._model(xs)
            outputs = COCOSegmentationModels._pack_outputs(outputs)
            loss = self._compute_loss(outputs, ys)
        out_argmax, new_ys = self._process_for_accuracy(outputs["out"].detach(), ys)
        pixel_acc = (out_argmax == new_ys).float().mean() * 100.0
        self.log("train/loss", loss.detach(), prog_bar=True)
        self.log("train/batch_pixel_acc", pixel_acc.detach(), prog_bar=True)
        self.manual_backward(loss)
        return loss

    def test_step(self, batch, batch_idx: int, prefix: str = "test") -> STEP_OUTPUT:
        """Collect logits and postprocess"""
        xs, ys = batch
        xs = xs.cuda()
        ys = ys.cuda()
        with self._autocast_ctx:
            outputs = self._model(xs)
            outputs = COCOSegmentationModels._pack_outputs(outputs)
            loss = self._compute_loss(outputs, ys)
        self._metric.update(outputs["out"].argmax(1).flatten(), ys.flatten())
        return loss

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
        """Backward call, if AMP enabled unscale gradients and clip"""
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
        self.zero_grad()
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
        self._metric.reset()

    def on_test_epoch_end(self, prefix: str = "test") -> None:
        """Compute metrics"""
        acc_global, acc, iu = self._metric.compute()
        self.log(f"{prefix}/pixel_acc", acc_global * 100, on_epoch=True)
        self.log(f"{prefix}/mIoU", iu.mean() * 100, on_epoch=True)
        self._metric.reset()

    def on_validation_epoch_start(self) -> None:
        """Called in the validation loop at the very beginning of the epoch"""
        self.on_test_epoch_start()

    def on_validation_epoch_end(self):
        """Called in the validation loop at the very end of the epoch"""
        self.on_test_epoch_end("val")

    def configure_optimizers(self):
        """Configure optimizer and LRScheduler used for training"""
        optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=self._args.lr,
            momentum=self._args.momentum,
            weight_decay=self._args.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._args.steps_per_epoch * 6, eta_min=self._args.lr * 0.01
        )
        return [optimizer], [lr_scheduler]

    def initialize_clika(self, data_module: COCODataModule) -> None:
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
            },
        )
        self._model.clika_visualize(os.path.join(self._args.output_dir, f"{self._args.model}.svg"))
        state_dict = self._model.clika_serialize()
        if dist_utils.is_main_process():
            torch.save(state_dict, os.path.join(self._args.output_dir, f"{self._args.model}_init.pompom"))
        torch.onnx.export(
            model=self._model,
            args=example_inputs.cuda(),
            f=os.path.join(self._args.output_dir, f"{self._args.model}_init.onnx"),
            input_names=["x"],
            dynamic_axes={"x": {0: "batch_size"}},
        )


def evaluate_original(
    trainer: pl.Trainer, module: COCOSegmentationModels, data_module: COCODataModule, args: argparse.Namespace
) -> dict:
    results = trainer.test(
        model=module, datamodule=data_module, verbose=True if dist_utils.is_main_process() else False
    )[0]
    results = {f"original/{k}": v for k, v in results.items()}
    return results


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="CLIKA TorchVision Segmentation Example")

    ace_parser = parser.add_argument_group("General configuration")
    ace_parser.add_argument("--clika_config", type=Path, default="local/clika_config.yaml", help="ACE config yaml path")
    ace_parser.add_argument("--output_dir", type=Path, default=None, help="Path to save clika related files for the SDK")
    ace_parser.add_argument("--data_path", type=Path, default="coco", help="COCO Dataset directory")
    ace_parser.add_argument("--resume", type=Path, default=None, help="Path to load the model checkpoints (e.g. .pth)")
    ace_parser.add_argument("--model", type=str, choices=[
        "fcn_resnet50",
        "fcn_resnet101",
        "deeplabv3_resnet50",
        "deeplabv3_resnet101",
        "deeplabv3_mobilenet_v3_large",
        "lraspp_mobilenet_v3_large",
    ], default="fcn_resnet50", help="")
    ace_parser.add_argument("--weights", type=str, default="DEFAULT", help="weights of Torchvision Model to use")

    train_parser = parser.add_argument_group("Train configuration")
    train_parser.add_argument("--epochs", type=int, default=200, help="Number of epochs (default 200)")
    train_parser.add_argument("--steps_per_epoch", type=int, default=100, help="Steps per epoch (default 100)")
    train_parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    train_parser.add_argument("--batch_size", type=int, default=8, help="Train, Evaluation batch size")
    train_parser.add_argument("--workers", type=int, default=3, help="Number of worker processes for data loading (default: 3)")
    train_parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for the optimizer")
    train_parser.add_argument("--clip_grad_norm", type=float, default=0, help="Random Seed")

    gpu_parser = parser.add_argument_group("GPU configuration")
    gpu_parser.add_argument("--gpu", type=int, default=0, help="GPU id to run on")
    gpu_parser.add_argument("--world_size", default=1, type=int, help="Number of distributed processes")
    gpu_parser.add_argument("--dist_url", default="env://", type=str, help="Url used to set up distributed training")

    etc_parser = parser.add_argument_group("ETC configuration")
    etc_parser.add_argument("--random_seed", type=int, default=373737, help="Random Seed")
    etc_parser.add_argument("--use_deterministic_algorithms", action="store_true", help="Whether or not to use deterministic algorithms. will slow down training")
    etc_parser.add_argument("--print_freq", type=int, default=1, help="Printing frequency")
    etc_parser.add_argument("--dry_run", action="store_true", help="Whether to run the initial calibration without further fine tuning")

    extra_args = parser.parse_args()
    base_args, tv_extra = get_args_parser().parse_known_args()
    # fmt: on

    pl.seed_everything(extra_args.random_seed)
    extra_args.clip_grad_norm = None if extra_args.clip_grad_norm == 0 else extra_args.clip_grad_norm

    # Load the diff for the args
    path_to_torchvision_config = list(BASE_DIR.joinpath("default_configs").rglob(f"**/{extra_args.model}.yaml"))
    if len(path_to_torchvision_config) != 1:
        raise ValueError(f"No match or too many matches for config of model: {extra_args.model}")
    with path_to_torchvision_config[0].open("r") as fp:
        vision_config = yaml.full_load(fp)
    if vision_config is not None:  # some configurations are empty because default args work for them
        for k, v in vision_config.items():  # from torchvision references for a specific model
            setattr(base_args, k, v)

    for k, v in vars(extra_args).items():  # From our parser to torchvision parser default args
        if not hasattr(base_args, k):  # Verifying
            if k not in {"clika_config", "random_seed", "steps_per_epoch", "gpu", "dry_run", "clip_grad_norm"}:
                raise ValueError(f"Invalid extra args. Key {k} does not exist.")
        setattr(base_args, k, v)
    args = base_args

    if not args.data_path.exists():
        raise FileNotFoundError(f"Cannot locate data directory at: `{args.data_path}`")
    if not args.clika_config.exists():
        raise FileNotFoundError(f"Cannot locate clika config at: `{args.clika_config}`")
    with open(args.clika_config, "r") as fp:
        args.clika_config_yaml = yaml.safe_load(fp)
    args.clika_config = str(args.clika_config)

    args.output_dir = args.output_dir or BASE_DIR.joinpath("outputs", args.model)
    args.output_dir = os.path.join(args.output_dir, args.clika_config_yaml["deployment_settings"]["target_framework"])
    dist_utils.init_distributed_mode(args)
    args.output_dir = os.path.join(args.output_dir, args.clika_config_yaml["deployment_settings"]["target_framework"])
    if not dist_utils.is_dist_avail_and_initialized():
        torch.cuda.set_device(args.gpu)
    if dist_utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"

    return args


def main():
    args = parse_args()
    data_module = COCODataModule(args)
    module = COCOSegmentationModels(args=args)
    callbacks: list = [
        ModelCheckpoint(
            monitor=module.BEST_CKPT_METRIC_NAME,
            mode=module.BEST_CKPT_METRIC_NAME_MODE,
            dirpath=args.output_dir,
            save_on_train_epoch_end=True,
            filename="epoch-{epoch}-pixel_acc-{val/pixel_acc:3.5f}-mIoU-{val/mIoU:3.5f}",
            save_top_k=1,
            auto_insert_metric_name=False,
            verbose=True,
        ),
        RichModelSummary(),
    ]
    tqdm_progress_bar_callback: Optional[TQDMProgressBar] = None
    if dist_utils.is_main_process():
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
        enable_progress_bar=True if dist_utils.is_main_process() else False,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
    )
    if dist_utils.is_main_process():
        trainer.print(f"Args {args}\n")
    original_test_results: Optional[dict] = None
    if not args.dry_run:
        original_test_results = evaluate_original(trainer=trainer, module=module, data_module=data_module, args=args)
    module.initialize_clika(data_module)
    if args.dry_run:
        return
    if dist_utils.is_dist_avail_and_initialized() is False:
        tune_batch_size(
            trainer=trainer,
            module=module,
            data_module=data_module,
            batch_arg_name="train_batch_size",
            method="fit",
            max_trials=3,
            steps_per_trial=5,
            init_val=args.batch_size,
            output_dir=args.output_dir,
        )
    if tqdm_progress_bar_callback is not None:
        tqdm_progress_bar_callback.original_test_results = original_test_results
    dist_utils.override_pl_strategy(
        trainer, args
    )  # we override the strategy so Distributed Training works properly with PL
    trainer.fit(model=module, datamodule=data_module, ckpt_path=args.resume)
    if dist_utils.is_dist_avail_and_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
