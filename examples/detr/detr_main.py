import copy

from PIL import Image
from typing import List, Optional, Union

import PIL.Image
import datasets
import torch
import os
from pathlib import Path
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrForSegmentation
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput
from clika_ace import (
    clika_compile,
    ClikaModule, QuantizationSettings, BaseDeploymentSettings,
    DeploymentSettings_TensorRT_ONNX, DeploymentSettings_OpenVINO_ONNX,
    PruningSettings,
)
from detr_utils import draw_bounding_boxes


def visualize_debug(
        images: List[PIL.Image.Image], processor: DetrImageProcessor, model: torch.nn.Module,
        is_segmentation: bool
):
    if is_segmentation:
        return  # TODO
    with torch.no_grad():
        for img in images:
            xs = processor(images=img, return_tensors="pt").to("cuda")
            outputs = model(xs)
            detr_output = DetrObjectDetectionOutput(logits=outputs[0], pred_boxes=outputs[1])
            if is_segmentation:
                pass  # TODO
            else:
                predictions = processor.post_process_object_detection(detr_output, target_sizes=[img.size[::-1]])
            result = copy.deepcopy(img)
            for pred in predictions:
                box = pred["boxes"]
                scores = pred["scores"]
                labels = pred["labels"]
                result = draw_bounding_boxes(result, box.numpy(), scores.numpy(), labels.numpy())
            result.show()


def run(
        model_name: str,
        images: List[PIL.Image.Image],
        deployment_settings: BaseDeploymentSettings,
        quantization_settings: Optional[QuantizationSettings],
        pruning_settings: Optional[PruningSettings],
        output_dir: Path,
        cache_dir: Path) -> None:
    file_friendly_model_name = model_name.replace("/", "_").replace(".", "_").replace("-", "_")
    file_friendly_model_name = f"{file_friendly_model_name}_{deployment_settings}"

    revision: Optional[str] = "no_timm"
    if any([x in model_name for x in ["-panoptic", "-dc5"]]):
        revision = None
    is_segmentation_model: bool = "-panoptic" in model_name
    processor: DetrImageProcessor = DetrImageProcessor.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=str(cache_dir), revision=revision
    )
    model: Optional[Union[DetrForObjectDetection, DetrForSegmentation]] = None
    if is_segmentation_model:
        model = DetrForSegmentation.from_pretrained(
            pretrained_model_name_or_path=model_name,
            cache_dir=str(cache_dir),
            return_dict=False, revision=revision,
            output_attentions=False, output_hidden_states=False
        )
    else:
        model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=model_name,
            cache_dir=str(cache_dir),
            return_dict=False, revision=revision,
            output_attentions=False, output_hidden_states=False
        )
    model.cuda()

    tracing_inputs: List[dict] = [
        dict(processor(images=images[:4], size=(720, 1280), return_tensors="pt").to(model.device)),
        dict(processor(images=images[:8], size=(640, 640), return_tensors="pt").to(model.device)),
        dict(processor(images=images[:2], return_tensors="pt").to(model.device)),
    ]
    assert len(images) > 12
    _images_to_pop = list(images)  # so we dont modify the external one
    calibration_inputs: List[dict] = [  # splitting it off to smaller batch sizes so we dont OOM.
        dict(processor(images=[_images_to_pop.pop(0) for _ in range(2)], return_tensors="pt").to(model.device)),
        dict(processor(images=[_images_to_pop.pop(0) for _ in range(2)], return_tensors="pt").to(model.device)),
        dict(processor(images=[_images_to_pop.pop(0) for _ in range(4)], return_tensors="pt").to(model.device)),
        dict(processor(images=[_images_to_pop.pop(0) for _ in range(4)], return_tensors="pt").to(model.device)),
        dict(processor(images=_images_to_pop, return_tensors="pt").to(model.device)),
    ]
    with torch.no_grad():
        model(**tracing_inputs[0])  # sanity test

    clika_model: ClikaModule = clika_compile(
        model=model,
        deployment_settings=deployment_settings,
        quantization_settings=quantization_settings,
        pruning_settings=pruning_settings,
        tracing_inputs=tracing_inputs,
        calibration_inputs=calibration_inputs,
        forward_fn=lambda model, xs: model(**xs),
        logs_dir=output_dir.joinpath("logs"),
    )

    clika_model.clika_save(output_dir.joinpath(file_friendly_model_name))
    # clika_model = ClikaModule.clika_load(output_dir.joinpath(file_friendly_model_name))

    # if not is_segmentation_model:  # TODO Uncomment for visualization
    #     visualize_debug(images=images, processor=processor, model=clika_model, is_segmentation=is_segmentation_model)

    output_names = ["logits", "pred_boxes"]
    keep_outputs_at_idx = [0, 1]
    if is_segmentation_model:
        output_names.append("pred_masks")
        keep_outputs_at_idx.append(2)
    with clika_model.export_configuration(keep_outputs_at_idx=keep_outputs_at_idx):
        clika_model.clika_export(
            file=str(output_dir.joinpath(file_friendly_model_name))+".onnx",
            input_names=["pixel_values", "pixel_mask"],
            output_names=output_names,
            dynamic_axes={
                "pixel_values": {0: "bsz", 2: "H", 3: "W"},
                "pixel_mask": {0: "bsz", 1: "H", 2: "W"},
            },
        )


def main():
    torch.cuda.set_device(0)
    base_dir = Path(__file__).parent
    cache_dir = base_dir.joinpath("cache")
    output_dir = base_dir.joinpath("files")
    output_dir.mkdir(exist_ok=True)

    num_examples_to_calibrate: int = 16

    dataset = datasets.load_dataset(
        path="detection-datasets/coco",
        split="train",
        streaming=True,
        cache_dir=str(cache_dir),
        token=os.environ.get("HF_TOKEN", None)
    ).take(num_examples_to_calibrate)
    images: list = [x["image"].convert("RGB") for x in dataset]

    all_deployment_settings = [
        DeploymentSettings_OpenVINO_ONNX(),
        # DeploymentSettings_TensorRT_ONNX(),  # there is a bug in TensorRT - the topology not supported.
    ]
    # Regarding the note above, there is some numerical error that gets accumulated.
    #   OpenVINO handles the model perfectly. TensorRT can't handle. Even without Quantization.
    quantization_settings = QuantizationSettings(
        weights_utilize_full_int_range=True,
        prefer_weights_only_quantization=False,
        quantization_sensitivity_threshold=0.005,
        weights_num_bits=[8, 4]
    )
    pruning_settings = PruningSettings(mask_type="1:1", pruning_sensitivity_threshold=0.001)

    all_model_names: list = [
        "facebook/detr-resnet-50",
        "facebook/detr-resnet-101",
        "facebook/detr-resnet-50-dc5",
        "facebook/detr-resnet-101-dc5",
        # "facebook/detr-resnet-50-panoptic",  # Will be supported next minor release - has unsupported op
        # "facebook/detr-resnet-101-panoptic",  # Will be supported next minor release - has unsupported op
        # "facebook/detr-resnet-50-dc5-panoptic",  # Will be supported next minor release - has unsupported op
    ]
    for deployment_settings in all_deployment_settings:
        for model_name in all_model_names:
            print(f"Running {model_name} - {deployment_settings}")
            run(
                model_name=model_name,
                images=images,
                deployment_settings=deployment_settings,
                quantization_settings=quantization_settings,
                pruning_settings=pruning_settings,
                output_dir=output_dir,
                cache_dir=cache_dir,
            )


if __name__ == '__main__':
    main()
