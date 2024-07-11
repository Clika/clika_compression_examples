import argparse
import sys
import warnings
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Any, List, Tuple

import torch
import torchvision
from clika_compression import Settings, TensorBoardCallback, clika_compress, clika_resume
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision import transforms

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "pytorch-retinanet"))
from retinanet.anchors import Anchors
from retinanet.dataloader import AspectRatioBasedSampler, Augmenter, CocoDataset, Normalizer, Resizer, collater
from retinanet.losses import FocalLoss
from retinanet.model import ResNet
from retinanet.utils import BBoxTransform, Bottleneck, ClipBoxes


# Define Class/Function Wrappers
# ==================================================================================================================== #
class RetinaNet(ResNet):
    """
    Original Retinanet code calculates final loss within a single forward call when self.training=True
    Remove loss calculation and return logit values for classification & regression heads + anchors based on input image

    REFERENCE:
    https://github.com/yhenon/pytorch-retinanet/blob/0348a9d57b279e3b5b235461b472d37da5feec3d/retinanet/model.py#L233-L257
    """

    def __init__(self, num_classes, block, layers):
        super().__init__(num_classes, block, layers)

        # Anchors related field variables
        self.pyramid_levels = [3, 4, 5, 6, 7]
        self.strides = [2**x for x in self.pyramid_levels]
        self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        self.ratios = torch.tensor([0.5, 1, 2])
        self.scales = torch.tensor([2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image_batch) -> Tuple[Tensor, Tensor]:
        x = self.conv1(image_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        return classification, regression


def _collate_fn(data) -> Tuple[Tensor, tuple]:
    """Parse dataloader output to satisfy CLIKA SDK input requirements.
    (https://docs.clika.io/docs/compression-constraints/cco_input_requirements#Dataset-Dataloader)
    """
    batch: dict = collater(data)

    # `batch["img"].shape` is required to call forward on ClipBoxes (used inside MetricWrapper)
    return batch["img"], (batch["annot"], batch["scale"], batch["img"].shape)


def get_loader(data_dir: Path, batch_size: int, workers: int, train=True) -> DataLoader:
    """Factory function that generates train/eval Dataloader.

    REFERENCE:
    https://github.com/yhenon/pytorch-retinanet/blob/0348a9d57b279e3b5b235461b472d37da5feec3d/train.py#L43-L46
    https://github.com/yhenon/pytorch-retinanet/blob/0348a9d57b279e3b5b235461b472d37da5feec3d/train.py#L69-L74
    """
    if train is True:
        dataset = CocoDataset(
            str(data_dir),
            set_name="train2017",
            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]),
        )
        sampler = AspectRatioBasedSampler(dataset, batch_size=batch_size, drop_last=False)
        dataloader = DataLoader(
            dataset,
            num_workers=workers,
            collate_fn=_collate_fn,
            batch_sampler=sampler,
            pin_memory=False,
        )

    else:
        dataset = CocoDataset(
            str(data_dir),
            set_name="val2017",
            transform=transforms.Compose([Normalizer(), Resizer()]),
        )
        sampler = AspectRatioBasedSampler(dataset, batch_size=batch_size, drop_last=False)
        dataloader = DataLoader(
            dataset,
            num_workers=workers,
            collate_fn=_collate_fn,
            batch_sampler=sampler,
            pin_memory=False,
        )

    return dataloader


class CriterionWrapper:
    """Wrapper around the `FocalLoss` & `Anchors` to satisfy CLIKA SDK Loss restriction.
    (https://docs.clika.io/docs/compression-constraints/cco_input_requirements#Loss-Function)

    REFERENCE:
    https://github.com/yhenon/pytorch-retinanet/blob/0348a9d57b279e3b5b235461b472d37da5feec3d/retinanet/model.py#L254-L257
    """

    def __init__(self):
        self.loss_fn = FocalLoss()
        self.anchors = Anchors()

    def __call__(self, predictions: Tuple[Tensor, Tensor], targets: Tuple[Tensor, tuple]) -> Tensor:
        classification, regression = predictions
        annotations, _, shape = targets
        anchors = self.anchors(torch.ones(shape))
        return self.loss_fn.forward(classification, regression, anchors, annotations)


class MetricWrapper(MeanAveragePrecision):
    """MeanAveragePrecision class wrapper that postprocess model's logit outputs and compute mAP"""

    def __init__(
        self,
        nms_thresh: float,
        conf_thresh: float,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh

    def update(self, outputs: List[Tensor], targets: tuple):
        classification, regression = outputs
        annotations, scale, shape = targets

        anchors = self.anchors(torch.ones(shape))
        _device = anchors.device
        regression = regression.to(_device)
        annotations = annotations.to(_device)

        """
        postprocess model logits

        REFERENCE:
        https://github.com/yhenon/pytorch-retinanet/blob/0348a9d57b279e3b5b235461b472d37da5feec3d/retinanet/model.py#L259-L297
        """

        transformed_anchors = self.regressBoxes(anchors, regression.to(_device))
        transformed_anchors = self.clipBoxes(transformed_anchors, torch.ones(shape))

        finalScores = torch.Tensor([]).to(_device)
        finalAnchorBoxesIndexes = torch.Tensor([]).long().to(_device)
        finalAnchorBoxesCoordinates = torch.Tensor([]).to(_device)

        for i in range(classification.shape[2]):
            scores = torch.squeeze(classification[:, :, i])
            scores_over_thresh = scores >= self.conf_thresh
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just continue
                continue

            scores = scores[scores_over_thresh].to(_device)
            anchorBoxes = torch.squeeze(transformed_anchors)
            anchorBoxes = anchorBoxes[scores_over_thresh]
            anchors_nms_idx = torchvision.ops.nms(anchorBoxes, scores, self.nms_thresh)

            finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
            finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0]).to(_device)

            finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
            finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

        scores, classification, transformed_anchors = (
            finalScores,
            finalAnchorBoxesIndexes,
            finalAnchorBoxesCoordinates,
        )
        transformed_anchors = transformed_anchors / scale[0]

        super().update(
            [
                {
                    "labels": classification,
                    "scores": scores,
                    "boxes": transformed_anchors,  # xyxy
                }
            ],
            [
                {
                    "labels": annotations[0][:, 4].long(),
                    "boxes": annotations[0][:, :4] / scale[0],
                }
            ],
        )

    def compute(self) -> dict:
        results: dict = super().compute()
        results.pop("classes", None)
        return results


# ==================================================================================================================== #
def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="CLIKA RetinaNet Example")

    ace_parser = parser.add_argument_group("CLIKA ACE configuration")
    ace_parser.add_argument("--config", type=str, default="config.yml", help="ACE config yaml path")

    model_parser = parser.add_argument_group("Model configuration")
    model_parser.add_argument("--output_dir", type=Path, default="outputs", help="Path to save clika related files for the SDK")
    model_parser.add_argument("--data", type=Path, default="coco", help="Dataset directory")
    model_parser.add_argument("--ckpt", type=Path, default="coco_resnet_50_map_0_335_state_dict.pt", help="Path to load the model checkpoints (e.g. .pth, .pompom)")
    model_parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation (default: 8)")
    model_parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    model_parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for data loading (default: 1)")
    model_parser = parser.add_argument_group("Additional arguments")
    model_parser.add_argument("--nms_threshold", type=float, default=0.5, help="NMS threshold to evaluate mAP (default: 0.5)")
    model_parser.add_argument("--confidence_threshold", type=float, default=0.05, help="Confidence threshold to evaluate mAP (default: 0.05)")

    args = parser.parse_args()

    if args.data.exists() is False:
        raise FileNotFoundError(f"Unknown directory: {args.data}")
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
    model = RetinaNet(80, Bottleneck, [3, 4, 6, 3])

    _resume = False
    if (_train_from_scratch is False) and (args.ckpt is not None):
        if args.ckpt.suffix == ".pompom":
            print("[INFO] .pompom file provided as ckpt, resuming compression ...")
            _resume = True
        else:
            print(f"[INFO] loading custom ckpt from {args.ckpt}")
            import torch.utils.model_zoo as model_zoo

            model.load_state_dict(
                model_zoo.load_url(
                    "https://download.pytorch.org/models/resnet50-19c8e357.pth",
                    model_dir=str(BASE_DIR),
                ),
                strict=False,
            )
            model.load_state_dict(torch.load(str(args.ckpt)))
    torch.fx.symbolic_trace(model)

    """
    Define Loss Function
    ====================================================================================================================
    """
    train_losses = {"loss_sum": CriterionWrapper()}
    eval_losses = None

    """
    Define Optimizer
    ====================================================================================================================
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    """
    Define Dataloaders
    ====================================================================================================================
    """
    get_train_loader = partial(
        get_loader,
        data_dir=args.data,
        batch_size=args.batch_size,
        workers=args.workers,
        train=True,
    )
    get_eval_loader = partial(get_loader, data_dir=args.data, batch_size=1, workers=args.workers, train=False)

    """
    Define Metric Wrapper
    ====================================================================================================================
    """
    train_metrics = None
    eval_metrics = MetricWrapper(nms_thresh=args.nms_threshold, conf_thresh=args.confidence_threshold)
    eval_metrics = {"mAP": eval_metrics}

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
