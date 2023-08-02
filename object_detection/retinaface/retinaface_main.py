import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from typing import Optional, List, Any, Dict, Callable, Union
import argparse
from functools import partial
from pathlib import Path
import warnings

import numpy as np
import cv2
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torchvision
from torchmetrics.detection import MeanAveragePrecision

from clika_compression import PyTorchCompressionEngine, QATQuantizationSettings, DeploymentSettings_TensorRT_ONNX, \
    DeploymentSettings_TFLite
from clika_compression.settings import (
    generate_default_settings, LayerQuantizationSettings, ModelCompileSettings
)

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "Pytorch_Retinaface"))
from models.retinaface import RetinaFace
from models import net
from data import WiderFaceDetection, detection_collate, preproc, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode

CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.35

COMPDTYPE = Union[Dict[str, Union[Callable, torch.nn.Module]], None]

DEPLOYMENT_DICT = {
    'trt': DeploymentSettings_TensorRT_ONNX(graph_author="CLIKA",
                                            graph_description=None,
                                            input_shapes_for_deployment=[(None, 3, None, None)]),

    'tflite': DeploymentSettings_TFLite(graph_author="CLIKA",
                                        graph_description=None,
                                        input_shapes_for_deployment=[(None, 3, None, None)]),
}


def load_checkpoints(config, model):
    """
    loading the model from checkpoint using code from the original RetinaFace Repository
    https://github.com/biubug6/Pytorch_Retinaface/blob/b984b4b775b2c4dced95c1eadd195a5c7d32a60b/train.py#L59-L71
    """
    state_dict = torch.load(config.ckpt)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model


# Define Class/Function Wrappers
# ==================================================================================================================== #
def FPN_forward_replacement(self, x):
    """
    CLIKA Compression SDK currently does not support 'size' method on tensors
    replace `size` with 'shape' by overwrite the original forward funtion of the model
    https://github.com/biubug6/Pytorch_Retinaface/blob/b984b4b775b2c4dced95c1eadd195a5c7d32a60b/models/net.py#L81C8-L81C8
    """

    x = list(x.values())

    output1 = self.output1(x[0])
    output2 = self.output2(x[1])
    output3 = self.output3(x[2])

    up3 = torch.nn.functional.interpolate(output3, size=[output2.shape[2], output2.shape[3]], mode="nearest")
    output2 = output2 + up3
    output2 = self.merge2(output2)

    up2 = torch.nn.functional.interpolate(output2, size=[output1.shape[2], output1.shape[3]], mode="nearest")
    output1 = output1 + up2
    output1 = self.merge1(output1)

    out = [output1, output2, output3]
    return out


net.FPN.forward = FPN_forward_replacement


class CRITERION_WRAPPER(MultiBoxLoss):
    """
    Making the loss funtion conform with CLIKA Compression constrains by returning a dict
    This way we can see the name of each loss during the logs of the training process
    https://github.com/biubug6/Pytorch_Retinaface/blob/b984b4b775b2c4dced95c1eadd195a5c7d32a60b/layers/modules/multibox_loss.py#L44
    """
    def __init__(self, *attrs, priors):
        self.priors = priors
        super().__init__(*attrs)

    def forward(self, predictions, targets):
        loss_l, loss_c, loss_landm = super().forward(predictions, self.priors, targets)
        return {"loss_l": loss_l, "loss_c": loss_c, "loss_landm": loss_landm}


def get_train_loader_(config, image_size):
    data_path = str(Path(config.data).joinpath("train", "label.txt"))
    dataset = WiderFaceDetection(data_path, preproc(image_size, rgb_means=(104, 117, 123)))  # BGR order
    loader = DataLoader(dataset, config.batch_size, shuffle=True, num_workers=config.workers,
                        collate_fn=detection_collate)
    return loader


def get_eval_loader_(config, image_size):
    data_path = str(Path(config.data).joinpath("val", "label.txt"))
    dataset = WiderFaceEvalDataset(data_path, size=image_size)
    loader = DataLoader(dataset, config.batch_size, shuffle=True, num_workers=config.workers,
                        collate_fn=detection_collate)
    return loader


class MeanAveragePrecisionWrapper(MeanAveragePrecision):
    """
    A custom metric class that inherits from a torchmetrics object.
    we use this class to preform postprocessing to the model's outputs before calculating the MeanAveragePrecision
    """
    def __init__(self,
                 priors: torch.Tensor,
                 conf_thres: float,
                 iou_thres: float,
                 img_shape: tuple,
                 box_format: str = "xyxy",
                 iou_type: str = "bbox",
                 iou_thresholds: Optional[List[float]] = None,
                 rec_thresholds: Optional[List[float]] = None,
                 max_detection_thresholds: Optional[List[int]] = None,
                 class_metrics: bool = False,
                 **kwargs: Any):
        super().__init__(box_format, iou_type, iou_thresholds, rec_thresholds, max_detection_thresholds, class_metrics,
                         **kwargs)
        self.variances = [0.1, 0.2]
        img_sizes = torch.tensor((img_shape[0], img_shape[1]))
        img_sizes = torch.flip(img_sizes, (-1,))
        img_sizes = torch.cat((img_sizes, img_sizes), dim=-1)
        self.add_state("prior_data", default=priors, persistent=False)
        self.add_state("img_sizes", default=img_sizes, persistent=False)
        self.conf_thres: float = conf_thres
        self.iou_thres: float = iou_thres
        self.img_shape = img_shape

    def decode_outputs(self, batch_bbox_regressions: torch.Tensor, batch_classifications: torch.Tensor):
        """post process the logit outputs in order to calculate MeanAveragePrecision"""
        BATCH_SIZE: int = batch_bbox_regressions.size()[0]
        batch_scores = torch.softmax(batch_classifications, -1)[..., 1][..., None]
        batch_inds = batch_scores > self.conf_thres
        filtered_batch_scores = [batch_scores[i][batch_inds[i]] for i in range(BATCH_SIZE)]
        detections: list = []
        for i in range(BATCH_SIZE):
            scores = filtered_batch_scores[i]
            if len(scores) == 0:
                detections.append(None)
                continue
            inds = batch_inds[i]
            bbox_regressions = batch_bbox_regressions[i]
            priors = self.prior_data[inds.ravel()]
            bbox_regressions = bbox_regressions[inds.ravel()]
            bboxes = decode(loc=bbox_regressions, priors=priors, variances=self.variances)
            bboxes = bboxes * self.img_sizes

            # keep top-K before NMS
            order = torch.argsort(scores, descending=True)
            bboxes = bboxes[order]
            scores = scores[order]

            # do NMS
            dets = torch.hstack((bboxes, scores[:, None])).to(torch.float32)
            keep = torchvision.ops.nms(bboxes, scores, iou_threshold=self.iou_thres)
            dets = dets[keep, :]
            detections.append(dets)
        return detections

    def update(self, output: List[Tensor], targets: List[Tensor]):
        """
        detections = model output, [(N, num_anchors, S, S, num_classes + 5),] * num_predictions
        targets = target label, (idx, cls, cxcywh)
        """
        if len(output) == 2:
            bbox_regressions, classifications = output
        else:
            bbox_regressions, classifications, _ = output
        targets = [target.to(bbox_regressions.device) for target in targets]

        detections = self.decode_outputs(bbox_regressions, classifications)
        for i in range(len(detections)):
            if detections[i] is None:
                continue
            pred_boxes, pred_score = torch.split(detections[i], (4, 1), -1)
            pred_labels = (pred_score > self.conf_thres).long()
            pred_boxes.clamp_(min=0)

            targets_boxes = targets[i] * torch.tensor(
                [self.img_shape[0], self.img_shape[1], self.img_shape[0], self.img_shape[1]],
                device=bbox_regressions.device)
            target_labels = torch.ones(targets_boxes.shape[0]).to(targets_boxes.device)
            # calculating MeanAveragePrecision
            super().update(
                [{
                    "labels": pred_labels.ravel().long(),
                    "scores": pred_score.ravel(),
                    "boxes": pred_boxes,  # xyxy
                }],
                [{
                    "labels": target_labels.ravel().long(),
                    "boxes": targets_boxes
                }])


class WiderFaceEvalDataset(WiderFaceDetection):
    """
    No eval dataloader defined inside original REPO
    Create custom one based on ...
    https://github.com/biubug6/Pytorch_Retinaface/blob/b984b4b775b2c4dced95c1eadd195a5c7d32a60b/data/wider_face.py#L9
    """
    def __init__(self, txt_path: str, size: int):
        super().__init__(txt_path, None)
        self.size = size

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        # preprocess without augmentation
        # https://github.com/biubug6/Pytorch_Retinaface/blob/b984b4b775b2c4dced95c1eadd195a5c7d32a60b/data/data_augment.py#L203-L206
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        img = img.astype("float32") - np.array([104, 117, 123], dtype="float32")
        img = img.transpose(2, 0, 1)  # HWC --> CHW

        labels = self.words[index]
        annotations = np.zeros((0, 4))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 4))
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        target[:, 0::2] /= width
        target[:, 1::2] /= height

        return torch.from_numpy(img), target


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
    mcs = ModelCompileSettings(
        optimizer=None,
        training_losses=train_losses,
        training_metrics=train_metrics,
        evaluation_losses=eval_losses,
        evaluation_metrics=eval_metrics,
    )
    engine = PyTorchCompressionEngine()

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

    # Skip quantization for last layers
    layer_names_to_skip = [
        "conv_75", "conv_77", "conv_81", "conv_80", "conv_76", "conv_75", "conv_74", "conv_79", "conv_73",
        "permute_5", "permute_4", "permute_3", "permute_8", "permute_7", "permute_6", "permute_2", "permute_1",
        "permute",
        "shape_9", "shape_8", "shape_7", "shape_12", "shape_11", "shape_10", "shape_6", "shape_5", "shape_4",
        "reshape_5", "reshape_4", "reshape_3", "reshape_8", "reshape_7", "reshape_6", "reshape_2", "reshape_1",
        "reshape",
        "concat_3", "concat_4", "concat_5"
    ]
    for x in layer_names_to_skip:
        settings.set_quantization_settings_for_layer(x, LayerQuantizationSettings(skip_quantization=True))

    mcs = ModelCompileSettings(
        optimizer=optimizer,
        training_losses=train_losses,
        training_metrics=train_metrics,
        evaluation_losses=eval_losses,
        evaluation_metrics=eval_metrics,
    )
    engine = PyTorchCompressionEngine()

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
    global BASE_DIR, CONF_THRESHOLD, IOU_THRESHOLD

    print("\n".join(f"{k}={v}" for k, v in vars(config).items()))  # pretty print argparse

    resume_compression_flag = False

    config.data = config.data if os.path.isabs(config.data) else str(BASE_DIR / config.data)
    if os.path.exists(config.data) is False:
        raise FileNotFoundError('Could not find default dataset please check `--data`')

    config.output_dir = config.output_dir if os.path.isabs(config.output_dir) else str(BASE_DIR / config.output_dir)

    """
    Define Model
    ====================================================================================================================
    """
    cfg = cfg_re50  # cfg_mnet for mobilenet backbone
    image_size = cfg['image_size']
    model = RetinaFace(cfg=cfg)

    if config.train_from_scratch is False:
        config.ckpt = config.ckpt if os.path.isabs(config.ckpt) else str(BASE_DIR / config.ckpt)

        if config.ckpt.rsplit(".", 1)[-1] == "pompom":
            warnings.warn(".pompom file provided, resuming compression (argparse attributes ignored)")
            resume_compression_flag = True
        else:
            print(f'loading ckpt from {config.ckpt}')
            model = load_checkpoints(config, model)

    """
    Define Loss Function
    ====================================================================================================================
    """
    _num_classes = 2  # face or not face

    # https://github.com/biubug6/Pytorch_Retinaface/blob/b984b4b775b2c4dced95c1eadd195a5c7d32a60b/train.py#L82-L87
    priorbox = PriorBox(cfg, image_size=(image_size, image_size))
    with torch.no_grad():
        _priors = priorbox.forward()
        _priors = _priors.cuda()

    train_losses = CRITERION_WRAPPER(_num_classes, 0.35, True, 0, True, 7, 0.35, False, priors=_priors)
    train_losses = {"loss_sum": train_losses}
    eval_losses = None

    """
    Define Optimizer
    ====================================================================================================================
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)

    """
    Define Dataloaders
    ====================================================================================================================
    """
    get_train_loader = partial(get_train_loader_, config=config, image_size=image_size)
    get_eval_loader = partial(get_eval_loader_, config=config, image_size=image_size)

    """
    Define Metric Wrapper
    ====================================================================================================================
    """
    train_metrics = None
    eval_metrics = MeanAveragePrecisionWrapper(_priors,
                                               img_shape=(image_size, image_size),
                                               conf_thres=CONF_THRESHOLD,
                                               iou_thres=IOU_THRESHOLD)
    eval_metrics = {"mAP": eval_metrics}

    """
    RUN COE
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIKA RetinaFace Example')
    parser.add_argument('--target_framework', type=str, default='trt', choices=["tflite", "trt"], help='choose the targe frame work TensorFlow Lite or TensorRT')
    parser.add_argument("--data", type=str, default="widerface", help="Dataset directory")

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
    parser.add_argument("--mixed_precision", action="store_true", default=False, help="Use Mixed Precision")
    parser.add_argument("--lr_warmup_epochs", type=int, default=1, help="Learning Rate used in the Learning Rate Warmup stage (default: 1)")
    parser.add_argument("--lr_warmup_steps_per_epoch", type=int, default=500, help="Number of steps per epoch used in the Learning Rate Warmup stage")
    parser.add_argument("--fp16_weights", action="store_true", default=False, help="Use FP16 weight (can reduce VRAM usage)")
    parser.add_argument("--gradients_checkpoint", action="store_true", default=False, help="Use gradient checkpointing")

    # Model Training Setting
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model (default: 100)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training and evaluation (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer (default: 1e-3)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for data loading (default: 4)")
    parser.add_argument("--ckpt", type=str, default="Resnet50_Final.pth", help="Path to load the model checkpoints (e.g. .pth, .pompom)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for saving results and checkpoints (default: outputs)")
    parser.add_argument("--train_from_scratch", action="store_true", help="Train the model from scratch")

    # Quantization Config
    parser.add_argument("--weights_num_bits", type=int, default=8, help="How many bits to use for the Weights for Quantization")
    parser.add_argument("--activations_num_bits", type=int, default=8, help="How many bits to use for the Activation for Quantization")

    args = parser.parse_args()

    main(config=args)
