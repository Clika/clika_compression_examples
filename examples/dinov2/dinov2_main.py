import os

from PIL import Image
from typing import List, Optional

import PIL.Image
import datasets
import torch
from pathlib import Path
from transformers import (
    AutoModel, AutoImageProcessor, Dinov2Model, BaseImageProcessor
)
from clika_ace import (
    clika_compile,
    ClikaModule, QuantizationSettings, BaseDeploymentSettings,
    DeploymentSettings_TensorRT_ONNX,
    DeploymentSettings_TensorRT_LLM_ONNX,
    DeploymentSettings_OpenVINO_ONNX,
    PruningSettings,
    clika_move_to_device
)


def run(
        model_name: str,
        images: List[PIL.Image.Image],
        deployment_settings: BaseDeploymentSettings,
        quantization_settings: Optional[QuantizationSettings],
        pruning_settings: Optional[PruningSettings],
        output_dir: Path,
        cache_dir: Path) -> None:
    file_friendly_model_name = model_name.replace("/", "_").replace(".", "_")
    file_friendly_model_name = f"{file_friendly_model_name}_{deployment_settings}"
    save_path_prefix = output_dir.joinpath(file_friendly_model_name)

    processor: BaseImageProcessor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=str(cache_dir)
    )
    model: Dinov2Model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=str(cache_dir),
        return_dict=False,
        output_attentions=False,
        attn_implementation="sdpa",  # if output_attentions True it will throw a Warning if it's "sdpa"
    ).cuda()
    if deployment_settings.is_TensorRT_LLM_ONNX():
        model.bfloat16()

    image_size: int = model.config.image_size

    images_for_tracing = images[:16]
    tracing_inputs: List[dict] = [
        dict(processor(images=images_for_tracing[0:1], size=(224, 224), return_tensors="pt")),
        dict(processor(images=images_for_tracing[0:2], size=(640, 640), return_tensors="pt")),
        dict(processor(images=images_for_tracing, size=(779, 520), return_tensors="pt")),
        dict(processor(images=images_for_tracing, size=(480, 480), return_tensors="pt")),
        dict(processor(images=images_for_tracing[:16], size=(320, 720), return_tensors="pt")),
    ]
    calibration_inputs: List[dict] = [
        dict(processor(images=images[:len(images) // 2], size=(image_size, image_size), return_tensors="pt")),
        dict(processor(images=images[len(images) // 2:], size=(image_size, image_size), return_tensors="pt")),
    ]
    calibration_inputs, tracing_inputs = clika_move_to_device((calibration_inputs, tracing_inputs), model.device)
    with torch.no_grad():
        model(**tracing_inputs[0])  # sanity test

    clika_model: ClikaModule = clika_compile(
        model=model,
        calibration_inputs=calibration_inputs,
        tracing_inputs=tracing_inputs,
        deployment_settings=deployment_settings,
        quantization_settings=quantization_settings,
        pruning_settings=pruning_settings,
    )

    clika_model.clika_save(save_path_prefix)
    # clika_model = ClikaModule.clika_load(save_path_prefix)

    export_bf16: bool = False
    if deployment_settings.is_TensorRT_LLM_ONNX():
        export_bf16 = True
    with clika_model.export_configuration(
        export_int4=True, export_bf16=export_bf16,
        # keep_outputs_at_idx=[0]
    ):
        clika_model.clika_export(
            file=str(save_path_prefix) + f".onnx",
            example_inputs=clika_model.clika_generate_dummy_inputs(),
            input_names=["pixel_values"],
            output_names=["sequence_output", "pooled_output"],
            dynamic_axes={
                "pixel_values": {0: "bsz", 2: "H", 3: "W"}
            },
        )


def main():
    torch.cuda.set_device(0)
    base_dir = Path(__file__).parent
    cache_dir = base_dir.joinpath("cache")
    output_dir = base_dir.joinpath("files")
    output_dir.mkdir(exist_ok=True)

    num_examples_to_calibrate: int = 32

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
        DeploymentSettings_TensorRT_ONNX(),
        DeploymentSettings_TensorRT_LLM_ONNX(),
    ]
    pruning_settings = PruningSettings(mask_type="1:1", pruning_sensitivity_threshold=0.001)

    all_model_names: list = [
        "facebook/dinov2-small",  # 22.1M
        "facebook/dinov2-base",  # 86.6M
        "facebook/dinov2-large",  # 304M
        "facebook/dinov2-giant",  # 1.14B
    ]

    for deployment_settings in all_deployment_settings:
        if deployment_settings.is_TensorRT_ONNX() or deployment_settings.is_OpenVINO_ONNX():
            quantization_settings = QuantizationSettings(
                weights_utilize_full_int_range=True,
                weights_num_bits=[8, 4],
                quantization_sensitivity_threshold=0.1,
                prefer_weights_only_quantization=False,
            )
        else:
            quantization_settings = QuantizationSettings(
                weights_utilize_full_int_range=True,
                weights_num_bits=[8, 4],
                quantization_sensitivity_threshold=0.1,
                prefer_weights_only_quantization=True,
            )
        for model_name in all_model_names:
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
