import argparse
import functools
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import lightning as pl
import torch
import torchvision
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
from torch import distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.optim import Optimizer
from torchmetrics.detection import MeanAveragePrecision

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "Pytorch_Retinaface"))
from clika_ace import ClikaModule
from data import WiderFaceDetection, cfg_mnet, cfg_re50, detection_collate, preproc
from example_utils.common import dist_utils as utils
from example_utils.common.pl_callbacks import RichModelSummary, TQDMProgressBar
from example_utils.common.pl_utils import tune_batch_size
from example_utils.retinaface_eval_dataset import WiderFaceEvalDataset
from layers.functions.prior_box import PriorBox
from layers.modules import MultiBoxLoss
from models.retinaface import RetinaFace
from utils.box_utils import decode


class WiderFaceDataModule(pl.LightningDataModule):
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
        sampler = None
        if is_train:
            data_path = str(args.data_path.joinpath("train", "label.txt"))
            dataset = WiderFaceDetection(data_path, preproc(args.image_size, rgb_means=(104, 117, 123)))  # BGR order
            sampler = torch.utils.data.RandomSampler(data_source=dataset)
        else:
            data_path = str(args.data_path.joinpath("val", "label.txt"))
            size = args.image_size
            if args.eval_origin_img:
                batch_size = 1
                size = None
            dataset = WiderFaceEvalDataset(data_path, size=size)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size,
            sampler=torch.utils.data.DistributedSampler(dataset, shuffle=is_train) if args.distributed else sampler,
            num_workers=args.workers,
            collate_fn=detection_collate,
            pin_memory=False,
        )
        return loader

    def setup(self, stage: str):
        """Preprocess data (called at the beginning of fit)"""
        pass

    def train_dataloader(self):
        if self._train_loader is None or self._train_loader.batch_size != self.train_batch_size:
            if self._train_loader is not None:
                del self._train_loader
            self._train_loader = WiderFaceDataModule.create_dataloaders(self._args, self.train_batch_size, True)
        return self._train_loader

    def test_dataloader(self):
        if self._test_loader is None or self._test_loader.batch_size != self.test_batch_size:
            if self._test_loader is not None:
                del self._test_loader
            self._test_loader = WiderFaceDataModule.create_dataloaders(self._args, self.test_batch_size, False)
        return self._test_loader

    def val_dataloader(self):
        return self.test_dataloader()

    def teardown(self, stage: Optional[str] = None):
        del self._train_loader
        del self._test_loader
        self._train_loader = None
        self._test_loader = None


class RetinaFaceModule(pl.LightningModule):
    _args: argparse.Namespace
    _model: torch.nn.Module
    _mAP_metric: MeanAveragePrecision
    _loss_fn: MultiBoxLoss
    BEST_CKPT_METRIC_NAME: str = "val/map_50"
    BEST_CKPT_METRIC_NAME_MODE: str = "max"
    _priors: torch.Tensor
    _metrics_to_print: List[str] = ["map", "map_50", "map_75", "map_small", "map_medium", "map_large"]
    _grad_scaler: torch.cuda.amp.GradScaler
    _autocast_ctx: torch.cuda.amp.autocast

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._args = args
        self.automatic_optimization = False

        """MODEL"""
        self._model = RetinaFace(cfg=self._args.cfg)
        ckpt_dir = BASE_DIR.joinpath("checkpoints")
        relevant_ckpt_path = list(BASE_DIR.joinpath("checkpoints").glob(f"{self._args.cfg['name']}*"))
        if len(relevant_ckpt_path) == 0:
            raise RuntimeError(f"No checkpoint found at {ckpt_dir}")
        # TODO: which ckpt? -> make an argparse
        state_dict = {k.replace("module.", ""): v for k, v in torch.load(relevant_ckpt_path[0]).items()}
        self._model.load_state_dict(state_dict)
        self._model.to(self._args.device)

        """LOSS"""
        _num_classes = 2  # [face, not_face]
        self.priorbox = PriorBox(args.cfg, (args.image_size, args.image_size))
        self._prior_cache: Dict[tuple, torch.Tensor] = {
            (args.image_size,) * 2: self.priorbox.forward().to(self._args.device)
        }
        self._loss_fn = MultiBoxLoss(
            num_classes=_num_classes,
            overlap_thresh=0.35,
            prior_for_matching=True,
            bkg_label=0,
            neg_mining=True,
            neg_pos=7,
            neg_overlap=0.35,
            encode_target=False,
        )

        """METRIC"""
        self._mAP_metric = MeanAveragePrecision()

        """ETC"""
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

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._model.zero_grad(set_to_none)

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        """Return training loss"""
        xs, ys = batch
        xs = xs.to(self._args.device)
        ys = [y.to(self._args.device) for y in ys]
        with self._autocast_ctx:
            loc, conf, landms = self._model(xs)
            loss_l, loss_c, loss_landm = self._loss_fn(
                (loc, conf, landms), self._prior_cache[(self._args.image_size, self._args.image_size)], ys
            )
            total_loss = loss_l + loss_c + loss_landm
        self.log("train/total_loss", total_loss.detach(), prog_bar=True)
        self.log("train/loss_localization", loss_l.detach(), prog_bar=True)
        self.log("train/loss_confidence", loss_c.detach(), prog_bar=True)
        self.log("train/loss_landmarks", loss_landm.detach(), prog_bar=True)
        self.manual_backward(total_loss)
        return total_loss

    def _decode_outputs(
        self,
        batch_bbox_regressions: List[torch.Tensor],
        batch_confidence: List[torch.Tensor],
        labels: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Post processing fn for evaluating mAP (self._mAP_metric)"""
        batch_size: int = batch_bbox_regressions.shape[0]
        batch_scores = torch.softmax(batch_confidence, -1)[..., 1][..., None]
        batch_inds = batch_scores >= self._args.confidence_threshold
        filtered_batch_scores = [batch_scores[i][batch_inds[i]] for i in range(batch_size)]
        detections: List[torch.Tensor] = []
        for i in range(batch_size):
            scores = filtered_batch_scores[i]
            if len(scores) == 0:
                detections.append(None)
                continue
            inds = batch_inds[i]
            bbox_regressions = batch_bbox_regressions[i]

            # when evaluating on original image size -> read prior cache || compute prior again
            if image_shapes[i] in self._prior_cache:
                _priors = self._prior_cache[image_shapes[i]]
            else:
                self.priorbox.image_size = image_shapes[i]
                self.priorbox.feature_maps = [
                    [math.ceil(self.priorbox.image_size[0] / step), math.ceil(self.priorbox.image_size[1] / step)]
                    for step in self.priorbox.steps
                ]
                _priors = self._prior_cache.setdefault(image_shapes[i], self.priorbox.forward())

            _priors = _priors.to(bbox_regressions.device)[inds.ravel()]
            bbox_regressions = bbox_regressions[inds.ravel()]
            bboxes = decode(loc=bbox_regressions, priors=_priors, variances=self._args.cfg["variance"])

            # scale predicted&target bboxes by WHWH (?, 4) * (4,)
            whwh = torch.flip(torch.tensor(image_shapes[i], device=bbox_regressions.device), (-1,)).repeat(2)
            bboxes = bboxes * whwh
            labels[i] = labels[i] * whwh

            order = torch.argsort(scores, descending=True)
            bboxes = bboxes[order]
            scores = scores[order]

            dets = torch.hstack((bboxes, scores[:, None])).to(torch.float32)
            keep = torchvision.ops.nms(bboxes, scores, iou_threshold=self._args.nms_threshold)
            dets = dets[keep, :]
            detections.append(dets)
        return detections, labels

    def test_step(self, batch, batch_idx: int, prefix: str = "test") -> STEP_OUTPUT:
        """Collect logits and postprocess"""
        xs, ys = batch

        xs = xs.to(self._args.device)
        with self._autocast_ctx:
            loc, conf, landms = self._model(xs)
        image_shapes = [tuple(_.shape[1:]) for _ in xs]  # type: List[Tuple[int, int]]
        detections, ys = self._decode_outputs(
            batch_bbox_regressions=loc, batch_confidence=conf, labels=ys, image_shapes=image_shapes
        )
        for i, det in enumerate(detections):
            if det is None:
                continue
            pred_boxes, pred_score = torch.split(det, [4, 1], -1)
            pred_labels = torch.ones(pred_boxes.shape[0], device=self._args.device)
            pred_boxes.clamp_(min=0)
            targets_boxes = ys[i].to(det.device)
            target_labels = torch.ones(targets_boxes.shape[0], device=self._args.device)
            self._mAP_metric.update(
                [{"labels": pred_labels.ravel().long(), "scores": pred_score.ravel(), "boxes": pred_boxes}],
                [{"labels": target_labels.ravel().long(), "boxes": targets_boxes}],
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

    def on_validation_epoch_end(self):
        """Called in the validation loop at the very end of the epoch"""
        self.on_test_epoch_end("val")

    def configure_optimizers(self):
        """Configure optimizer and LRScheduler used for training"""
        optimizer = torch.optim.SGD(
            self._model.parameters(), lr=self._args.lr, momentum=0.0, weight_decay=self._args.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._args.steps_per_epoch * 6, eta_min=self._args.lr * 0.01
        )
        return [optimizer], [lr_scheduler]

    def initialize_clika(self, data_module: WiderFaceDataModule) -> None:
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
        if utils.is_main_process():
            torch.save(state_dict, os.path.join(self._args.output_dir, f"{self._args.model}_init.pompom"))
        torch.onnx.export(
            model=self._model,
            args=example_inputs.cuda(),
            f=os.path.join(self._args.output_dir, f"{self._args.model}_init.onnx"),
            input_names=["x"],
            dynamic_axes={"x": {0: "batch_size", 2: "W", 3: "H"}},
        )


def evaluate_original(trainer: pl.Trainer, module: RetinaFaceModule, data_module: WiderFaceDataModule) -> dict:
    results = trainer.test(model=module, datamodule=data_module, verbose=True if utils.is_main_process() else False)[0]
    results = {f"original/{k}": v for k, v in results.items()}
    return results


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="CLIKA RetinaFace Example")

    ace_parser = parser.add_argument_group("General configuration")
    ace_parser.add_argument("--clika_config", type=Path, default="local/clika_config.yaml", help="ACE config yaml path")
    ace_parser.add_argument("--output_dir", type=Path, default=None, help="Path to save clika related files for the SDK")
    ace_parser.add_argument("--data_path", type=Path, default="widerface", help="Widerface Dataset directory")
    ace_parser.add_argument("--resume", type=Path, default=None, help="Path to load the model checkpoints (e.g. .pth)")

    train_parser = parser.add_argument_group("Train configuration")
    train_parser.add_argument("--model", type=str, default="resnet50", choices=["mobilenetv1", "resnet50"], help="[retinaface] Model backbone")
    train_parser.add_argument("--epochs", type=int, default=200, help="Number of epochs (default 200)")
    train_parser.add_argument("--steps_per_epoch", type=int, default=100, help="Steps per epoch (default 100)")
    train_parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Train, Evaluation batch size")
    train_parser.add_argument("--workers", type=int, default=3, help="Number of worker processes for data loading (default: 3)")
    train_parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for the optimizer")
    train_parser.add_argument("--amp", action='store_true', help="Whether to use auto mixed precision")
    train_parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="max grad norm")
    train_parser.add_argument("--lr_scheduler", type=str, default="cosineannealinglr", help="learning rate scheduler")

    eval_parser = parser.add_argument_group("Evaluation configuration")
    eval_parser.add_argument("--nms_threshold", type=float, default=0.4, help="NMS threshold to evaluate mAP (default: 0.4)")
    eval_parser.add_argument("--confidence_threshold", type=float, default=0.02, help="Confidence threshold to evaluate mAP (default: 0.02)")
    eval_parser.add_argument("--eval_origin_img", action="store_true", help="Use origin image size to evaluate (Evaluation Loader batch size will be fixed to 1)")

    gpu_parser = parser.add_argument_group("GPU configuration")
    gpu_parser.add_argument("--gpu", type=int, default=0, help="GPU id to run on")
    gpu_parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    gpu_parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")

    etc_parser = parser.add_argument_group("ETC configuration")
    etc_parser.add_argument("--random_seed", type=int, default=373737, help="Random Seed")
    etc_parser.add_argument("--use_deterministic_algorithms", action="store_true", help="whether or not to use deterministic algorithms. will slow down training")
    etc_parser.add_argument("--print_freq", type=int, default=1, help="printing frequency")
    etc_parser.add_argument("--dry_run", action="store_true", help="whether to run the initial calibration without further fine tuning")

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

    args.output_dir = args.output_dir or BASE_DIR.joinpath("outputs", args.model)
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

    if args.model == "mobilenetv1":
        args.cfg = cfg_mnet
    else:
        args.cfg = cfg_re50
    args.image_size = args.cfg["image_size"]

    return args


def main():
    args = parse_args()
    module = RetinaFaceModule(args=args)
    data_module = WiderFaceDataModule(args)
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
