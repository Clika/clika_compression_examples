import argparse
import functools
import os
import sys
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Literal, Optional, Union

import lightning as pl
import torch
import torchvision
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
from torch import distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.optim import Optimizer
from torch.utils.data import default_collate
from torchmetrics.classification import MulticlassAccuracy

BASE_DIR = Path(__file__).parent

sys.path.insert(0, str(BASE_DIR.joinpath("torchvision_reference", "classification")))
from clika_ace import ClikaModule
from example_utils.common import dist_utils
from example_utils.common.pl_callbacks import RichModelSummary, TQDMProgressBar
from example_utils.common.pl_utils import tune_batch_size
from torchvision_reference.classification import utils
from torchvision_reference.classification.train import get_args_parser, get_mixup_cutmix, load_data


class ImagenetDataModule(pl.LightningDataModule):
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
    def create_dataloaders(args: argparse.Namespace, train_batch_size: int, test_batch_size: int) -> tuple:
        train_dir = os.path.join(args.data_path, "train")
        val_dir = os.path.join(args.data_path, "val")
        with redirect_stdout(new_target=None), redirect_stderr(new_target=None):
            with dist_utils.distributed_local_main_first():
                dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

        num_classes = len(dataset.classes)
        mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=num_classes, use_v2=args.use_v2
        )
        if mixup_cutmix is not None:

            def collate_fn(batch):
                return mixup_cutmix(*default_collate(batch))

        else:
            collate_fn = default_collate

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=test_batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
        )
        return data_loader, data_loader_test

    def setup(self, stage: str):
        print(
            f"Setting up new dataloaders, train_batch_size={self.train_batch_size}, test_batch_size={self.test_batch_size}"
        )
        train_loader, test_loader = self.create_dataloaders(self._args, self.train_batch_size, self.test_batch_size)
        self._train_loader = train_loader
        self._test_loader = test_loader

    def train_dataloader(self):
        if self._train_loader is None or self._train_loader.batch_size != self.train_batch_size:
            self.teardown()
            self.setup("")
        return self._train_loader

    def test_dataloader(self):
        if self._test_loader is None or self._test_loader.batch_size != self.test_batch_size:
            self.teardown()
            self.setup("")
        return self._test_loader

    def val_dataloader(self):
        return self.test_dataloader()

    def teardown(self, stage: Optional[str] = None):
        del self._train_loader
        del self._test_loader
        self._train_loader = None
        self._test_loader = None


class ImagenetClassificationModels(pl.LightningModule):
    _args: argparse.Namespace
    _model: torch.nn.Module
    _top1_acc: MulticlassAccuracy
    _top5_acc: MulticlassAccuracy
    BEST_CKPT_METRIC_NAME: str = "val/top1"
    BEST_CKPT_METRIC_NAME_MODE: str = "max"
    _grad_scaler: torch.cuda.amp.GradScaler
    _autocast_ctx: torch.cuda.amp.autocast

    def __init__(self, args: Optional[argparse.Namespace] = None):
        super().__init__()
        self._args = args
        self.automatic_optimization = False

        """MODEL"""
        self._model = torchvision.models.get_model(name=args.model, weights=args.weights).to(self._args.device)

        """METRIC"""
        self._top1_acc = MulticlassAccuracy(num_classes=1000, top_k=1).to(self._args.device)
        self._top5_acc = MulticlassAccuracy(num_classes=1000, top_k=5).to(self._args.device)

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

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._model.zero_grad(set_to_none)

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        """Return training loss"""
        xs, ys = batch
        xs = xs.to(self._args.device)
        ys = ys.to(self._args.device)
        with self._autocast_ctx:
            y_hat = self._model(xs)
            loss = torch.nn.functional.cross_entropy(y_hat, ys)
        batch_top1, batch_top5 = utils.accuracy(y_hat.detach(), ys, topk=(1, 5))
        self.log("train/loss", loss.detach(), prog_bar=True)
        self.log("train/batch_top1", batch_top1, prog_bar=True)
        self.log("train/batch_top5", batch_top5, prog_bar=True)
        self.manual_backward(loss)
        return loss

    def test_step(self, batch, batch_idx: int, prefix: str = "test") -> STEP_OUTPUT:
        """Collect logits and postprocess"""
        xs, ys = batch
        xs = xs.to(self._args.device)
        ys = ys.to(self._args.device)
        with self._autocast_ctx:
            y_hat = self._model(xs)
            loss = torch.nn.functional.cross_entropy(y_hat, ys)
        self._top1_acc.update(y_hat, ys)
        self._top5_acc.update(y_hat, ys)
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
        self.zero_grad()
        self._top1_acc.reset()
        self._top5_acc.reset()

    def on_test_epoch_end(self, prefix: str = "test") -> None:
        """Compute metrics"""
        self.log(f"{prefix}/top1", self._top1_acc.compute() * 100.0, on_epoch=True)
        self.log(f"{prefix}/top5", self._top5_acc.compute() * 100.0, on_epoch=True)
        self._top1_acc.reset()
        self._top5_acc.reset()

    def on_validation_epoch_start(self) -> None:
        """Called in the validation loop at the very beginning of the epoch"""
        self.on_test_epoch_start()

    def on_validation_epoch_end(self):
        """Called in the validation loop at the very end of the epoch"""
        self.on_test_epoch_end("val")

    def configure_optimizers(self):
        """Configure optimizer and LRScheduler used for training"""
        custom_keys_weight_decay = []
        if self._args.bias_weight_decay is not None:
            custom_keys_weight_decay.append(("bias", self._args.bias_weight_decay))
        if self._args.transformer_embedding_decay is not None:
            for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
                custom_keys_weight_decay.append((key, self._args.transformer_embedding_decay))
        parameters = utils.set_weight_decay(
            self._model,
            self._args.weight_decay,
            norm_weight_decay=None,
            custom_keys_weight_decay=None,
        )

        opt_name = self._args.opt.lower()
        if opt_name.startswith("sgd"):
            optimizer = torch.optim.SGD(
                parameters,
                lr=self._args.lr,
                momentum=self._args.momentum,
                weight_decay=self._args.weight_decay,
                nesterov="nesterov" in opt_name,
            )
        elif opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                parameters,
                lr=self._args.lr,
                momentum=self._args.momentum,
                weight_decay=self._args.weight_decay,
                eps=0.0316,
                alpha=0.9,
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=self._args.lr, weight_decay=self._args.weight_decay)
        else:
            raise RuntimeError(f"Invalid optimizer {self._args.opt}. Only SGD, RMSprop and AdamW are supported.")
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._args.steps_per_epoch * 6, eta_min=self._args.lr * 0.01
        )
        return [optimizer], [lr_scheduler]

    def initialize_clika(self, data_module: ImagenetDataModule) -> None:
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
        self._model.clika_visualize(os.path.join(self._args.output_dir, f"{self._args.model}.svg"))
        state_dict = self._model.clika_serialize()
        if dist_utils.is_main_process():
            torch.save(state_dict, os.path.join(self._args.output_dir, f"{self._args.model}_init.pompom"))
        torch.onnx.export(
            model=self._model,
            args=example_inputs.cuda(),
            f=os.path.join(self._args.output_dir, f"{self._args.model}_init.onnx"),
            input_names=["x"],
            output_names=["logits"],
            dynamic_axes={"x": {0: "batch_size"}},
        )


def evaluate_original(
    trainer: pl.Trainer, module: ImagenetClassificationModels, data_module: ImagenetDataModule
) -> dict:
    results = trainer.test(
        model=module, datamodule=data_module, verbose=True if dist_utils.is_main_process() else False
    )[0]
    results = {f"original/{k}": v for k, v in results.items()}
    return results


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="CLIKA TorchVision Classification Example")

    ace_parser = parser.add_argument_group("General configuration")
    ace_parser.add_argument("--clika_config", type=Path, default="local/clika_config.yaml", help="ACE config yaml path")
    ace_parser.add_argument("--output_dir", type=Path, default=None, help="Path to save clika related files for the SDK")
    ace_parser.add_argument("--data_path", type=Path, default="imagenet/ILSVRC/Data/CLS-LOC", help="Imagenet Dataset directory")
    ace_parser.add_argument("--resume", type=Path, default=None, help="Path to load the model checkpoints (e.g. .pth)")
    ace_parser.add_argument("--model", type=str, choices=[
        "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32",
        "vit_h_14",  # vit_h_14, unofficial recipe, may lead to mediocre results
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        "resnext50_32x4d", "resnext101_32x8d",
        "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
        "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l",
        "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",

        "densenet121", "densenet161", "densenet169", "densenet201",
        "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_y_3_2gf", "regnet_y_8gf",
        "regnet_y_16gf", "regnet_y_32gf", "regnet_y_128gf", "regnet_x_400mf", "regnet_x_800mf",
        "regnet_x_1_6gf", "regnet_x_3_2gf", "regnet_x_8gf", "regnet_x_16gf", "regnet_x_32gf",
        "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0",

        # TODO: following models with AdaptivePool(output_size>1) cannot be exported to ONNX
        #   "alexnet",
        #   "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
        # TODO:
        #  "maxvit_t", # has very long torch.compile time, unreasonable.
        # TODO: no official recipe for the following but can be made up with some patience.
        #   Recipe *may* be available under the Model Weights, i.e. torchvision.models.XXXXX_Weights objects
        #   "resnext101_64x4d",
        #   "wide_resnet50_2", "wide_resnet101_2",
        #   "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
        #   "inception_v3",
        #   "squeezenet1_0", "squeezenet1_1",
        #   "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",  # adaptive pool with output size != (1, 1, ...) not supported
    ], default="resnet18", help="torchvision model to run")
    ace_parser.add_argument("--weights", type=str, default="DEFAULT", help="weights of Torchvision Model to use")

    train_parser = parser.add_argument_group("Train configuration")
    train_parser.add_argument("--epochs", type=int, default=200, help="Number of epochs (default 200)")
    train_parser.add_argument("--steps_per_epoch", type=int, default=100, help="Steps per epoch (default 100)")
    train_parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    train_parser.add_argument("--batch_size", type=int, default=8, help="Train, Evaluation batch size")
    train_parser.add_argument("--workers", type=int, default=3, help="Number of worker processes for data loading (default: 3)")
    train_parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for the optimizer")
    train_parser.add_argument("--clip_grad_norm", type=float, default=0, help="Random Seed")
    # TODO: amp

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
            if k not in {"clika_config", "random_seed", "steps_per_epoch", "gpu", "dry_run"}:
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
    data_module = ImagenetDataModule(args)
    module = ImagenetClassificationModels(args=args)
    callbacks: list = [
        ModelCheckpoint(
            monitor=module.BEST_CKPT_METRIC_NAME,
            mode=module.BEST_CKPT_METRIC_NAME_MODE,
            dirpath=args.output_dir,
            save_on_train_epoch_end=True,
            filename="epoch-{epoch}-top1-{val/top1:3.5f}-top5-{val/top5:3.5f}",
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
        original_test_results = evaluate_original(trainer=trainer, module=module, data_module=data_module)
    module.initialize_clika(data_module)
    if args.dry_run:
        return
    if dist_utils.is_dist_avail_and_initialized() is False:
        # when singleGPU tune batch_size
        _ = functools.partial(
            tune_batch_size,
            trainer=trainer,
            module=module,
            data_module=data_module,
            max_trials=3,
            steps_per_trial=5,
            init_val=args.batch_size,
            output_dir=args.output_dir,
        )
        _(batch_arg_name="test_batch_size", method="test")
        _(batch_arg_name="train_batch_size", method="fit")

    if tqdm_progress_bar_callback is not None and original_test_results is not None:
        tqdm_progress_bar_callback.original_test_results = original_test_results
    dist_utils.override_pl_strategy(
        trainer, args
    )  # we override the strategy so Distributed Training works properly with PL
    trainer.fit(model=module, datamodule=data_module, ckpt_path=args.resume)
    if dist_utils.is_dist_avail_and_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
