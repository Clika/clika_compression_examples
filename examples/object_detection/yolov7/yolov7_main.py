import argparse
import functools
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

import lightning as pl
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
from torch import distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.optim import Optimizer
from torchmetrics.detection import MeanAveragePrecision

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.joinpath("yolov7")))
from clika_ace import ClikaModule
from example_utils.common import dist_utils as utils
from example_utils.common.pl_callbacks import RichModelSummary, TQDMProgressBar
from example_utils.common.pl_utils import tune_batch_size
from example_utils.yolov7_install_new_detect import install_fixed_detect
from models.common import RepConv
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import non_max_suppression, xywhn2xyxy
from utils.loss import ComputeLossOTA


class COCODataModule(pl.LightningDataModule):
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
    def create_dataloaders(args: argparse.Namespace, batch_size: int, is_train: bool) -> torch.utils.data.DataLoader:
        """Initialize dataloaders (train, eval)"""
        rank: int = utils.get_rank() if utils.is_dist_avail_and_initialized() else -1
        _ = functools.partial(
            create_dataloader,
            stride=args.cfg["grid_size"],
            opt=args,
            hyp=args.hyp_cfg,
            cache=False,
            image_weights=False,
            quad=False,
        )
        if is_train:
            return _(
                path=args.data_path.joinpath("train2017.txt"),
                imgsz=args.image_size,
                batch_size=batch_size,
                augment=True,
                pad=0.0,
                rect=False,
                rank=rank,
                world_size=utils.get_world_size(),
                workers=args.workers,
                prefix="Train: ",
            )
        else:
            batch_size = 1 if utils.is_dist_avail_and_initialized() else batch_size
            return _(
                path=args.data_path.joinpath("val2017.txt"),
                imgsz=args.eval_image_size,
                batch_size=batch_size,
                augment=False,
                pad=0.5,
                rect=True,
                rank=rank,
                world_size=utils.get_world_size(),
                workers=args.workers,
                prefix="Val: ",
            )

    def setup(self, stage: str):
        """Preprocess data (called at the beginning of fit)"""
        pass

    def train_dataloader(self):
        if self._train_loader is None or self._train_loader.batch_size != self.train_batch_size:
            if self._train_loader is not None:
                del self._train_loader
            self._train_loader, _ = COCODataModule.create_dataloaders(self._args, self.train_batch_size, True)
        return self._train_loader

    def test_dataloader(self):
        if self._test_loader is None or self._test_loader.batch_size != self.test_batch_size:
            if self._test_loader is not None:
                del self._test_loader
            self._test_loader, _ = COCODataModule.create_dataloaders(self._args, self.test_batch_size, False)
        return self._test_loader

    def val_dataloader(self):
        return self.test_dataloader()

    def teardown(self, stage: Optional[str] = None):
        del self._train_loader
        del self._test_loader
        self._train_loader = None
        self._test_loader = None


class YoloV7Module(pl.LightningModule):
    _args: argparse.Namespace
    _model: torch.nn.Module
    _mAP_metric: MeanAveragePrecision
    _loss_fn: ComputeLossOTA
    BEST_CKPT_METRIC_NAME: str = "val/map_50"
    BEST_CKPT_METRIC_NAME_MODE: str = "max"
    _anchors: torch.Tensor
    _strides: torch.Tensor
    _metrics_to_print: List[str] = ["map", "map_50", "map_75", "map_small", "map_medium", "map_large"]
    _grad_scaler: torch.cuda.amp.GradScaler
    _autocast_ctx: torch.cuda.amp.autocast

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._args = args
        self.automatic_optimization = False

        """MODEL"""
        self._model = Model(cfg=self._args.cfg)
        ckpt_dir = BASE_DIR.joinpath("checkpoints")
        relevant_ckpt_path = list(BASE_DIR.joinpath("checkpoints").glob(f"{self._args.cfg_path.stem}*"))
        if len(relevant_ckpt_path) == 0:
            raise RuntimeError(f"No checkpoint found at {ckpt_dir}")
        state_dict = torch.load(relevant_ckpt_path[0])["model"].state_dict()
        self._model.load_state_dict(state_dict)
        self._model.to(self._args.device)
        # fuse RepConv blocks before we start training
        for m in self._model.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()

        """ETC"""
        # NOTE: grid size (max stride) - required for datamoudle to be initialized
        self._args.cfg["grid_size"] = max(int(self._model.stride.max()), 32)

        self._strides = self._model.stride.clone().to(self._args.device)
        self._anchors = torch.tensor(self._args.cfg["anchors"], dtype=torch.float32, device=self._args.device)

        # follow how original repo sets variables
        setattr(self._model, "hyp", self._args.hyp_cfg)
        setattr(self._model, "nc", self._args.data_cfg["nc"])
        setattr(self._model, "gr", 1.0)

        self.save_hyperparameters(vars(args))

        # we create the GradScaler with enabled=True so that it initializes the internal vars
        if utils.is_dist_avail_and_initialized():
            self._grad_scaler = ShardedGradScaler(enabled=True)
        else:
            self._grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        self._grad_scaler._enabled = False
        self._autocast_ctx = nullcontext()
        if self._args.amp:
            self._autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16, cache_enabled=False)
            self._grad_scaler._enabled = True

        """LOSS"""
        self._loss_fn = ComputeLossOTA(self._model)

        """METRIC"""
        self._mAP_metric = MeanAveragePrecision()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._model.zero_grad(set_to_none)

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        """Return training loss"""
        xs, ys, _, _ = batch
        xs = xs.to(self._args.device).float() / 255.0
        ys = ys.to(self._args.device)
        with self._autocast_ctx:
            predictions = self._model(xs)
            total_loss, loss_items = self._loss_fn(predictions, ys, xs)
            loss_box, loss_obj, loss_cls, _ = loss_items.unbind(0)
        self.log("train/total_loss", total_loss.detach(), prog_bar=True)
        self.log("train/loss_boxes", loss_box.detach(), prog_bar=True)
        self.log("train/loss_objs", loss_obj.detach(), prog_bar=True)
        self.log("train/loss_cls", loss_cls.detach(), prog_bar=True)
        self.manual_backward(total_loss)
        return total_loss

    def _decode_outputs(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Post processing fn for evaluating mAP (self._mAP_metric)"""
        device: torch.device = predictions[0].device
        anchor_grid = self._anchors.view(len(predictions), 1, -1, 1, 1, 2)
        grid = [torch.zeros(1, device=device)] * len(predictions)
        z = []
        for i, output in enumerate(predictions):
            (
                bs,
                num_anchors,
                ny,
                nx,
                out_channels,
            ) = output.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

            if grid[i].shape[2:4] != output.shape[2:4]:
                yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
                grid[i] = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().to(device)

            y = output.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + grid[i]) * self._strides[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1, out_channels))
        detections = torch.cat(z, dim=1)
        detections = non_max_suppression(
            detections,
            multi_label=False,
            conf_thres=self._args.confidence_threshold,
            iou_thres=self._args.nms_threshold,
        )
        return detections

    def test_step(self, batch, batch_idx: int, prefix: str = "test") -> STEP_OUTPUT:
        """Collect logits and postprocess"""
        xs, ys, _, shapes = batch
        xs = xs.to(self._args.device).float() / 255.0
        ys = ys.to(self._args.device)
        with self._autocast_ctx:
            predictions = self._model(xs)
        detections = self._decode_outputs(predictions)
        for i, det in enumerate(detections):
            if det is None:
                continue
            cur_img_shape: tuple = xs[i].shape[1:]
            _ts = ys[ys[:, 0] == i]  # filter targets
            _ts[:, 2:] = xywhn2xyxy(_ts[:, 2:], h=cur_img_shape[0], w=cur_img_shape[1], padh=0, padw=0)
            self._mAP_metric.update(
                [
                    {
                        "labels": det[:, -1].ravel().long(),
                        "scores": det[:, -2].ravel(),
                        "boxes": det[:, 0:4],
                    }
                ],
                [{"labels": _ts[:, 1].ravel().long(), "boxes": _ts[:, 2:]}],
            )
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
        self._mAP_metric.reset()

    def on_test_epoch_end(self, prefix: str = "test") -> None:
        """Compute metrics"""
        bbox_map_results = self._mAP_metric.compute()
        for met in self._metrics_to_print:
            if met in bbox_map_results:
                self.log(f"{prefix}/{met}", value=bbox_map_results[met], on_epoch=True)
        self._mAP_metric.reset()

    def on_validation_epoch_start(self) -> None:
        """Called in the validation loop at the very beginning of the epoch"""
        self.on_test_epoch_start()

    def on_validation_epoch_end(self) -> None:
        """Called in the validation loop at the very end of the epoch"""
        self.on_test_epoch_end("val")

    def configure_optimizers(self):
        """Configure optimizer and LRScheduler used for training"""
        optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=self._args.lr,
            momentum=0.0,
            weight_decay=self._args.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._args.steps_per_epoch * 6, eta_min=self._args.lr * 0.01
        )
        return [optimizer], [lr_scheduler]

    def initialize_clika(self, data_module: COCODataModule) -> None:
        """Initialize CLIKAModule - Wrap `nn.Module` with `ClikaModule`"""
        train_dataloader = data_module.train_dataloader()
        self._model.to(self._args.device)
        self._model: ClikaModule = torch.compile(
            self._model,
            backend="clika",
            options={
                "settings": self._args.clika_config,
                "train_dataloader": train_dataloader,
                "discard_input_model": True,
                "logs_dir": os.path.join(self._args.output_dir, "logs"),
                "apply_on_data_fn": lambda x: x[0].float() / 255.0,  # yolov7 dataloader returns uint8 data
            },
        )
        self._model.clika_visualize(os.path.join(self._args.output_dir, f"{self._args.cfg_path.stem}.svg"))
        state_dict = self._model.clika_serialize()
        if utils.is_main_process():
            torch.save(state_dict, os.path.join(self._args.output_dir, f"{self._args.cfg_path.stem}_init.pompom"))
        torch.onnx.export(
            model=self._model,
            args=self._model.clika_generate_dummy_inputs(),
            f=os.path.join(self._args.output_dir, f"{self._args.cfg_path.stem}_init.onnx"),
            input_names=["x"],
            dynamic_axes={"x": {0: "batch_size", 2: "H", 3: "W"}},
        )


def evaluate_original(trainer: pl.Trainer, module: YoloV7Module, data_module: COCODataModule) -> dict:
    results = trainer.test(model=module, datamodule=data_module, verbose=True if utils.is_main_process() else False)[0]
    results = {f"original/{k}": v for k, v in results.items()}
    return results


def override_pl_strategy(trainer: pl.Trainer, args: argparse.Namespace) -> None:
    from pytorch_lightning.strategies import FSDPStrategy

    if utils.is_dist_avail_and_initialized():
        trainer.strategy.global_rank = utils.get_rank()
        trainer.strategy.local_rank = args.gpu
        trainer.strategy.world_size = utils.get_world_size()
        type(trainer.strategy).is_global_zero = property(fget=lambda self: utils.is_main_process())
        type(trainer.strategy)._determine_device_ids = lambda x: [args.gpu]
        type(trainer.strategy).barrier = FSDPStrategy.barrier  # could be DDPStrategy as well, doesn't matter.
        type(trainer.strategy).broadcast = FSDPStrategy.broadcast


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="CLIKA YOLOv7 Example")

    ace_parser = parser.add_argument_group("General configuration")
    ace_parser.add_argument("--clika_config", type=Path, default="local/clika_config.yaml", help="ACE config yaml path")
    ace_parser.add_argument("--output_dir", type=Path, default=None, help="Path to save clika related files for the SDK")
    ace_parser.add_argument("--data_path", type=Path, default="coco", help="COCO Dataset directory")
    ace_parser.add_argument("--resume", type=Path, default=None, help="Path to load the model checkpoints (e.g. .pth)")

    train_parser = parser.add_argument_group("Train configuration")
    train_parser.add_argument("--cfg_path", default=None, type=Path, help="[YOLOv7] model yaml file e.g. `yolov7/cfg/deploy/yolov7.yaml`")
    train_parser.add_argument("--hyp_cfg_path", default=None, type=Path, help="[YOLOv7] hyperparameter yaml file e.g. `yolov7/data/hyp.scratch.p5`")
    train_parser.add_argument("--data_cfg_path", default=None, type=Path, help="[YOLOv7] data hyperparameter yaml file e.g. `yolov7/data/coco.yaml`")
    train_parser.add_argument("--epochs", type=int, default=200, help="Number of epochs (default 200)")
    train_parser.add_argument("--steps_per_epoch", type=int, default=100, help="Steps per epoch (default 100)")
    train_parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Train, Evaluation batch size")
    train_parser.add_argument("--workers", type=int, default=3, help="Number of worker processes for data loading (default: 3)")
    train_parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for the optimizer")
    train_parser.add_argument("--no_amp", action='store_true', help="Whether to use auto mixed precision")
    train_parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Max grad norm (0 to disable)")
    train_parser.add_argument("--image_size", type=int, default=640, help="Train image size (default: 640)")
    train_parser.add_argument("--single_cls", action='store_true', help="Train multi-class data as single-class")

    eval_parser = parser.add_argument_group("Evaluation configuration")
    eval_parser.add_argument("--nms_threshold", type=float, default=0.65, help="NMS threshold to evaluate mAP (default: 0.65)")
    eval_parser.add_argument("--confidence_threshold", type=float, default=0.001, help="Confidence threshold to evaluate mAP (default: 0.001)")
    eval_parser.add_argument("--eval_image_size", type=int, default=640, help="Evaluation image size (default: 640)")

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
    args.amp = not args.no_amp
    args.clip_grad_norm = None if args.clip_grad_norm == 0 else args.clip_grad_norm

    # locate and read yolov7 repository config yaml files
    args.cfg_path = args.cfg_path or BASE_DIR.joinpath("yolov7", "cfg", "deploy", "yolov7.yaml")
    args.hyp_cfg_path = args.hyp_cfg_path or BASE_DIR.joinpath("yolov7", "data", "hyp.scratch.p5.yaml")
    args.data_cfg_path = args.data_cfg_path or BASE_DIR.joinpath("yolov7", "data", "coco.yaml")
    with open(args.cfg_path) as fp:
        args.cfg = yaml.load(fp, Loader=yaml.SafeLoader)
    with open(args.hyp_cfg_path) as fp:
        args.hyp_cfg = yaml.load(fp, Loader=yaml.SafeLoader)
    with open(args.data_cfg_path) as fp:
        args.data_cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    if not args.data_path.exists():
        raise FileNotFoundError(f"Cannot locate data directory at: `{args.data_path}`")
    if not args.clika_config.exists():
        raise FileNotFoundError(f"Cannot locate clika config at: `{args.clika_config}`")
    with open(args.clika_config, "r") as fp:
        args.clika_config_yaml = yaml.safe_load(fp)
    args.clika_config = str(args.clika_config)

    args.output_dir = args.output_dir or BASE_DIR.joinpath("outputs", args.cfg_path.stem)
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

    args.num_detection_heads = len(args.cfg["head"][-1][0])
    args.hyp_cfg["box"] *= 3.0 / args.num_detection_heads  # scale to layers
    args.hyp_cfg["cls"] *= args.data_cfg["nc"] / 80.0 * 3.0 / args.num_detection_heads  # scale to classes and layers
    args.hyp_cfg["obj"] *= (args.image_size / 640) ** 2 * 3.0 / args.num_detection_heads  # scale to image size and layers
    args.hyp_cfg["label_smoothing"] = 0
    args.cfg["grid_size"] = 32  # not sure how to calculate it but should be `max(int(model.stride.max()), 32)`, grid size = max stride

    # labels_to_class_weights(args.data_cfg["names"], args.data_cfg["nc"])
    return args


def main():
    args = parse_args()
    install_fixed_detect()
    module = YoloV7Module(args=args)
    data_module = COCODataModule(args)
    callbacks: list = [
        ModelCheckpoint(
            monitor=module.BEST_CKPT_METRIC_NAME,
            mode=module.BEST_CKPT_METRIC_NAME_MODE,
            dirpath=args.output_dir,
            save_on_train_epoch_end=True,
            filename="epoch-{epoch}-map-{val/map:3.5f}-map_50-{val/map_50:3.5f}",
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
        # keep track of original evaluation results
        original_test_results = evaluate_original(trainer=trainer, module=module, data_module=data_module)
    module.initialize_clika(data_module)
    if args.dry_run:
        return

    if utils.is_dist_avail_and_initialized() is False:
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
    utils.override_pl_strategy(trainer, args)  # we override the strategy so Distributed Training works properly with PL
    trainer.fit(model=module, datamodule=data_module, ckpt_path=args.resume)
    if utils.is_dist_avail_and_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
