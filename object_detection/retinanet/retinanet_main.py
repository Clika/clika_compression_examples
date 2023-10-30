import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from typing import Optional, List, Any, Dict, Callable, Union
import argparse
from functools import partial
import warnings
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchmetrics.detection import MeanAveragePrecision

from clika_compression import PyTorchCompressionEngine, QATQuantizationSettings, DeploymentSettings_TensorRT_ONNX, \
    DeploymentSettings_TFLite
from clika_compression.settings import (
    generate_default_settings, LayerQuantizationSettings, ModelCompileSettings
)

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "pytorch-retinanet"))
from retinanet.model import ResNet, resnet50
from retinanet.dataloader import collater, CocoDataset, Normalizer, Resizer, Augmenter, AspectRatioBasedSampler
from retinanet.losses import FocalLoss
from retinanet.utils import Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors


IOU_THRESHOLD = 0.65
COMPDTYPE = Union[Dict[str, Union[Callable, torch.nn.Module]], None]

DEPLOYMENT_DICT = {
    "trt": DeploymentSettings_TensorRT_ONNX(graph_author="CLIKA",
                                            graph_description=None,
                                            input_shapes_for_deployment=[(None, 3, None, None)]),

    "tflite": DeploymentSettings_TFLite(graph_author="CLIKA",
                                        graph_description=None,
                                        input_shapes_for_deployment=[(None, 3, None, None)]),
}

# Define Class/Function Wrappers
# ==================================================================================================================== #
class Retinanet(ResNet):
    """
    Original Retinanet code calculates final loss within a single forward call when self.training=True

    Remove loss calculation and return logit values for classification & regression heads + anchors based on input image
    https://github.com/yhenon/pytorch-retinanet/blob/0348a9d57b279e3b5b235461b472d37da5feec3d/retinanet/model.py#L233-L257
    """

    def __init__(self, num_classes, block, layers):
        super().__init__(num_classes, block, layers)

        # Anchors related field variables
        self.pyramid_levels = [3, 4, 5, 6, 7]
        self.strides = [2 ** x for x in self.pyramid_levels]
        self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        self.ratios = torch.tensor([0.5, 1, 2])
        self.scales = torch.tensor([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image_batch):
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


class ClipBoxesWrapper(ClipBoxes):

    def forward(self, boxes, shape):
        img = torch.ones(*shape)  # dummy tensor to pass tensor shape information to actual ClipBoxes
        return super().forward(boxes, img)


def collater_wrapper(data):
    """
    original collater returns image batch in dictionary
    change return type to tuple =>(img, (annotation, scale, img.shape))
    img.shape required to call forward on ClipBoxes (later used inside metric function)

    https://github.com/yhenon/pytorch-retinanet/blob/0348a9d57b279e3b5b235461b472d37da5feec3d/retinanet/dataloader.py#L301
    """
    batch = collater(data)
    return batch["img"], (batch["annot"], batch["scale"], batch["img"].shape)


def get_loader(path, batch_size, workers, train=True) -> torch.utils.data.DataLoader:
    """
    Return train/eval Dataloader
    https://github.com/yhenon/pytorch-retinanet/blob/0348a9d57b279e3b5b235461b472d37da5feec3d/train.py#L43-L46
    https://github.com/yhenon/pytorch-retinanet/blob/0348a9d57b279e3b5b235461b472d37da5feec3d/train.py#L69-L74
    """
    if train is True:
        dataset = CocoDataset(path, set_name='train2017',
                              transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        sampler = AspectRatioBasedSampler(dataset, batch_size=batch_size, drop_last=False)
        dataloader = DataLoader(dataset, num_workers=workers, collate_fn=collater_wrapper, batch_sampler=sampler)

    else:
        dataset = CocoDataset(path, set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer()]))
        sampler = AspectRatioBasedSampler(dataset, batch_size=batch_size, drop_last=False)
        dataloader = DataLoader(dataset, num_workers=workers, collate_fn=collater_wrapper, batch_sampler=sampler)

    return dataloader


class CRITERION_WRAPPER(object):
    """
    Since we have overwritten Retinanet forward call
    Change loss_fn's parameters accordingly
    """
    def __init__(self):
        self.loss_fn = FocalLoss()
        self.anchors = Anchors()

    def __call__(self, predictions, targets):
        classification, regression = predictions
        annotations, _, shape = targets
        anchors = self.anchors(torch.ones(shape))
        return self.loss_fn.forward(classification, regression, anchors, annotations)


class MeanAveragePrecisionWrapper(MeanAveragePrecision):
    """
    A custom metric class that inherits from a torchmetrics object.
    We use this class to preform postprocessing to the model's outputs before calculating the MeanAveragePrecision
    """
    def __init__(self,
                 iou_thres: float,
                 box_format: str = "xyxy",
                 iou_type: str = "bbox",
                 iou_thresholds: Optional[List[float]] = None,
                 rec_thresholds: Optional[List[float]] = None,
                 max_detection_thresholds: Optional[List[int]] = None,
                 class_metrics: bool = False,
                 **kwargs: Any):
        super().__init__(box_format, iou_type, iou_thresholds, rec_thresholds, max_detection_thresholds, class_metrics,
                         **kwargs)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxesWrapper()

        self.iou_thres: float = iou_thres

    def update(self, outputs: List[Tensor], targets: Tensor):
        classification, regression = outputs
        annotations, scale, shape = targets
        anchors = self.anchors(torch.ones(shape))

        """
        postprocess model logits
        https://github.com/yhenon/pytorch-retinanet/blob/0348a9d57b279e3b5b235461b472d37da5feec3d/retinanet/model.py#L259-L297
        """
        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, shape)

        finalResult = [[], [], []]

        finalScores = torch.Tensor([])
        finalAnchorBoxesIndexes = torch.Tensor([]).long()
        finalAnchorBoxesCoordinates = torch.Tensor([])

        if torch.cuda.is_available():
            finalScores = finalScores.cuda()
            finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
            finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

        for i in range(classification.shape[2]):
            scores = torch.squeeze(classification[:, :, i])
            scores_over_thresh = (scores >= 0.05)  # TODO: if only interested in mAP@50 or mAP@75 can set it higher. will make your code faster
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just continue
                continue

            scores = scores[scores_over_thresh]
            anchorBoxes = torch.squeeze(transformed_anchors)
            anchorBoxes = anchorBoxes[scores_over_thresh]
            anchors_nms_idx = torchvision.ops.nms(anchorBoxes, scores, IOU_THRESHOLD)

            finalResult[0].extend(scores[anchors_nms_idx])
            finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
            finalResult[2].extend(anchorBoxes[anchors_nms_idx])

            finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
            finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
            if torch.cuda.is_available():
                finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

            finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
            finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

        scores, classification, transformed_anchors = finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates
        transformed_anchors = transformed_anchors / scale[0]

        super().update(
            [{
                "labels": classification,
                "scores": scores,
                "boxes": transformed_anchors,  # xyxy
            }],
            [{
                "labels": annotations[0][:, 4].long(),
                "boxes": annotations[0][:, :4] / scale[0]
            }])

    def compute(self) -> dict:
        results: dict = super().compute()
        results.pop("classes", None)
        return results

# ==================================================================================================================== #


def resume_compression(
        config: argparse.Namespace,
        get_train_loader: Callable,
        get_eval_loader: Callable,
        train_losses: COMPDTYPE,
        train_metrics: COMPDTYPE,
        eval_losses: COMPDTYPE = None,
        eval_metrics: COMPDTYPE = None
):
    engine = PyTorchCompressionEngine()

    mcs = ModelCompileSettings(
        optimizer=None,
        training_losses=train_losses,
        training_metrics=train_metrics,
        evaluation_losses=eval_losses,
        evaluation_metrics=eval_metrics,
    )
    final = engine.resume(
        clika_state_path=config.ckpt,
        model_compile_settings=mcs,
        init_training_dataset_fn=get_train_loader,
        init_evaluation_dataset_fn=get_eval_loader,
        settings=None
    )
    engine.deploy(
        clika_state_path=final,
        output_dir_path=config.output_dir,
        file_suffix=None,
        input_shapes=None,
        graph_author="CLIKA",
        graph_description="Demo - Created by clika.io",
    )


def run_compression(
        config: argparse.Namespace,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        get_train_loader: Callable,
        get_eval_loader: Callable,
        train_losses: COMPDTYPE,
        train_metrics: COMPDTYPE,
        eval_losses: COMPDTYPE = None,
        eval_metrics: COMPDTYPE = None
):
    global DEPLOYMENT_DICT

    engine = PyTorchCompressionEngine()
    settings = generate_default_settings()

    settings.deployment_settings = DEPLOYMENT_DICT[config.target_framework]
    settings.global_quantization_settings = QATQuantizationSettings()
    settings.global_quantization_settings.weights_num_bits = config.weights_num_bits
    settings.global_quantization_settings.activations_num_bits = config.activations_num_bits

    settings.training_settings.num_epochs = config.epochs

    settings.training_settings.steps_per_epoch = config.steps_per_epoch
    settings.training_settings.evaluation_steps = config.evaluation_steps
    settings.training_settings.stats_steps = config.stats_steps
    settings.training_settings.print_interval = config.print_interval
    settings.training_settings.print_num_steps_averaging = config.ma_window_size
    settings.training_settings.save_interval = config.save_interval
    settings.training_settings.reset_training_dataset_between_epochs = config.reset_train_data
    settings.training_settings.reset_evaluation_dataset_between_epochs = config.reset_eval_data
    settings.training_settings.grads_accumulation_steps = config.grads_acc_steps
    settings.training_settings.mixed_precision = config.mixed_precision
    settings.training_settings.lr_warmup_epochs = config.lr_warmup_epochs
    settings.training_settings.lr_warmup_steps_per_epoch = config.lr_warmup_steps_per_epoch
    settings.training_settings.use_fp16_weights = config.fp16_weights
    settings.training_settings.use_gradients_checkpoint = config.gradients_checkpoint

    layer_names_to_skip = {
        "concat_1",
        "conv_105", "sigmoid_3", "permute_8", "reshape_11", "reshape_12",
        "conv_110", "sigmoid_4", "permute_9", "reshape_13", "reshape_14",
        "conv_100", "sigmoid_2", "permute_7", "reshape_9", "reshape_10",
        "conv_90", "sigmoid", "permute_5", "reshape_5", "reshape_6",
        "conv_95", "sigmoid_1", "permute_6", "reshape_7", "reshape_8",
        "concat",
        "conv_85", "permute_4", "reshape_4",
        "conv_80", "permute_3", "reshape_3",
        "conv_75", "permute_2", "reshape_2",
        "conv_70", "permute_1", "reshape_1",
        "conv_65", "permute", "reshape",
    }
    for x in layer_names_to_skip:
        settings.set_quantization_settings_for_layer(x, LayerQuantizationSettings(skip_quantization=True))

    mcs = ModelCompileSettings(
        optimizer=optimizer,
        training_losses=train_losses,
        training_metrics=train_metrics,
        evaluation_losses=eval_losses,
        evaluation_metrics=eval_metrics,
    )
    final = engine.optimize(
        output_path=config.output_dir,
        settings=settings,
        model=model,
        model_compile_settings=mcs,
        init_training_dataset_fn=get_train_loader,
        init_evaluation_dataset_fn=get_eval_loader
    )
    engine.deploy(
        clika_state_path=final,
        output_dir_path=config.output_dir,
        file_suffix=None,
        input_shapes=None,
        graph_author="CLIKA",
        graph_description="Demo - Created by clika.io",
    )


def main(config):
    global BASE_DIR, IOU_THRESHOLD

    print("\n".join(f"{k}={v}" for k, v in vars(config).items()))  # pretty print argparse

    resume_compression_flag = False

    config.data = config.data if os.path.isabs(config.data) else str(BASE_DIR / config.data)
    if os.path.exists(config.data) is False:
        raise FileNotFoundError("Could not find default dataset please check `--data`")

    config.output_dir = config.output_dir if os.path.isabs(config.output_dir) else str(BASE_DIR / config.output_dir)

    """
    Define Model
    ====================================================================================================================
    """
    model = Retinanet(80, Bottleneck, [3, 4, 6, 3])

    if config.train_from_scratch is False:
        config.ckpt = config.ckpt if os.path.isabs(config.ckpt) else str(BASE_DIR / config.ckpt)

        if config.ckpt.rsplit(".", 1)[-1] == "pompom":
            warnings.warn(".pompom file provided, resuming compression (argparse attributes ignored)")
            resume_compression_flag = True
        else:
            print(f"loading ckpt from {config.ckpt}")

            import torch.utils.model_zoo as model_zoo
            model.load_state_dict(model_zoo.load_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", model_dir='.'), strict=False)
            model.load_state_dict(torch.load(config.ckpt))

    """
    Define Loss Function
    ====================================================================================================================
    """
    train_losses = CRITERION_WRAPPER()
    train_losses = {"loss_sum": train_losses}
    eval_losses = None

    """
    Define Optimizer
    ====================================================================================================================
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    """
    Define Dataloaders
    ====================================================================================================================
    """
    get_train_loader = partial(get_loader, batch_size=config.batch_size, workers=config.workers, path=config.data, train=True)
    get_eval_loader = partial(get_loader, batch_size=1, workers=config.workers, path=config.data, train=False)

    """
    Define Metric Wrapper
    ====================================================================================================================
    """
    train_metrics = None
    eval_metrics = MeanAveragePrecisionWrapper(iou_thres=IOU_THRESHOLD)
    eval_metrics = {"mAP": eval_metrics}

    """
    RUN Compression
    ====================================================================================================================    
    """
    if resume_compression_flag is True:
        resume_compression(
            config=config,
            get_train_loader=get_train_loader,
            get_eval_loader=get_eval_loader,
            train_losses=train_losses,
            train_metrics=train_metrics,
            eval_losses=eval_losses,
            eval_metrics=eval_metrics
        )
    else:
        run_compression(
            config=config,
            model=model,
            optimizer=optimizer,
            get_train_loader=get_train_loader,
            get_eval_loader=get_eval_loader,
            train_losses=train_losses,
            train_metrics=train_metrics,
            eval_losses=eval_losses,
            eval_metrics=eval_metrics
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIKA RetinaNet Example")
    parser.add_argument("--target_framework", type=str, default="trt", choices=["tflite", "trt"], help="choose the target framework TensorFlow Lite or TensorRT")
    parser.add_argument("--data", type=str, default="coco", help="Dataset directory")

    # CLIKA Engine Training Settings
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Number of steps per epoch")
    parser.add_argument("--evaluation_steps", type=int, default=None, help="Number of steps for evaluation")
    parser.add_argument("--stats_steps", type=int, default=50, help="Number of steps for scans")
    parser.add_argument("--print_interval", type=int, default=50, help="COE print log interval")
    parser.add_argument("--ma_window_size", type=int, default=20, help="Moving average window size (default: 20)")
    parser.add_argument("--save_interval", action="store_true", default=None, help="Save interval")
    parser.add_argument("--reset_train_data", action="store_true", default=False, help="Reset training dataset between epochs")
    parser.add_argument("--reset_eval_data", action="store_true", default=False, help="Reset evaluation dataset between epochs")
    parser.add_argument("--grads_acc_steps", type=int, default=4, help="gradient accumulation steps")
    parser.add_argument("--no_mixed_precision", action="store_false", default=True, dest="mixed_precision", help="Not using Mixed Precision")
    parser.add_argument("--lr_warmup_epochs", type=int, default=1, help="Learning Rate used in the Learning Rate Warmup stage (default: 1)")
    parser.add_argument("--lr_warmup_steps_per_epoch", type=int, default=500, help="Number of steps per epoch used in the Learning Rate Warmup stage")
    parser.add_argument("--fp16_weights", action="store_true", default=False, help="Use FP16 weight (can reduce VRAM usage)")
    parser.add_argument("--gradients_checkpoint", action="store_true", default=False, help="Use gradient checkpointing")

    # Model Training Setting
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model (default: 100)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training and evaluation (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer (default: 1e-5)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for data loading (default: 4)")
    parser.add_argument("--ckpt", type=str, default="coco_resnet_50_map_0_335_state_dict.pt", help="Path to load the model checkpoints (e.g. .pth, .pompom)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for saving results and checkpoints (default: outputs)")
    parser.add_argument("--train_from_scratch", action="store_true", help="Train the model from scratch")

    # Quantization Config
    parser.add_argument("--weights_num_bits", type=int, default=8, help="How many bits to use for the Weights for Quantization")
    parser.add_argument("--activations_num_bits", type=int, default=8, help="How many bits to use for the Activation for Quantization")

    args = parser.parse_args()

    main(config=args)
