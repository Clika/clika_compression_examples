import itertools
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import Optional, List, Tuple
import datasets
import torch
from pathlib import Path
from PIL import Image
from clika_ace import (
    clika_compile,
    ClikaModule, QuantizationSettings, PruningSettings,
    DeploymentSettings_TensorRT_ONNX, DeploymentSettings_OpenVINO_ONNX, DeploymentSettings_TensorRT_LLM_ONNX,
    BaseDeploymentSettings,
    clika_move_to_device,
)
from transformers import (
    AutoImageProcessor, AutoModelForDepthEstimation,
    DPTImageProcessor, DepthAnythingForDepthEstimation
)


def generate_data(images: List[Image.Image], processor: DPTImageProcessor, cache_dir: Path) -> Tuple[List[dict], List[dict]]:
    calibration_inputs: list = [dict(processor(x, return_tensors="pt")) for x in images]
    tracing_inputs = list()
    _unique_shapes: set = set()
    for img in images:
        xs = dict(processor([img]*2, return_tensors="pt"))  # for batch specialization as well
        pixel_values_shape: tuple = tuple(xs["pixel_values"].shape)
        if pixel_values_shape in _unique_shapes:
            continue
        else:
            tracing_inputs.append(xs)
            _unique_shapes.add(pixel_values_shape)
    return calibration_inputs, tracing_inputs


def main():
    torch.cuda.set_device(0)
    base_dir = Path(__file__).parent
    files_dir = base_dir.joinpath("files")
    cache_dir = base_dir.joinpath("cache")
    files_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    all_deployment_settings = [
        DeploymentSettings_OpenVINO_ONNX(),
        DeploymentSettings_TensorRT_ONNX(),
        DeploymentSettings_TensorRT_LLM_ONNX(),
    ]
    quantization_settings = QuantizationSettings(
        weights_num_bits=[8, 4],
        weights_utilize_full_int_range=True,
        prefer_weights_only_quantization=False,
        quantization_sensitivity_threshold=0.005
    )

    pruning_settings = None

    num_examples: int = 32
    dataset = datasets.load_dataset("detection-datasets/coco", split="train", streaming=True, cache_dir=str(cache_dir))
    dataset = list(dataset.shuffle(seed=69420).take(num_examples))
    images: List[Image.Image] = [x["image"].convert("RGB") for x in dataset]

    all_model_names = [
        "depth-anything/Depth-Anything-V2-Small-hf",  # 24.8M
        "depth-anything/Depth-Anything-V2-Base-hf",   # 97.5M
        "depth-anything/Depth-Anything-V2-Large-hf",  # 335M
    ]
    for deployment_settings in all_deployment_settings:
        for model_name in all_model_names:
            file_friendly_model_name = model_name.replace("/", "_").replace(".", "_")
            file_friendly_model_name = f"{file_friendly_model_name}_{deployment_settings}"

            processor: DPTImageProcessor = AutoImageProcessor.from_pretrained(model_name, cache_dir=str(cache_dir))
            model: DepthAnythingForDepthEstimation = AutoModelForDepthEstimation.from_pretrained(
                model_name, return_dict=False, cache_dir=str(cache_dir)
            ).cuda()

            calibration_inputs, tracing_inputs = generate_data(images=images, processor=processor, cache_dir=cache_dir)
            calibration_inputs, tracing_inputs = clika_move_to_device((calibration_inputs, tracing_inputs), "cuda")

            with torch.no_grad():
                model(**calibration_inputs[0])
            if deployment_settings.is_TensorRT_LLM_ONNX():
                model.half()

            clika_model: ClikaModule = clika_compile(
                model=model,
                deployment_settings=deployment_settings,
                quantization_settings=quantization_settings,
                pruning_settings=pruning_settings,
                calibration_inputs=calibration_inputs,
                tracing_inputs=tracing_inputs,
            )
            clika_model.clika_save(files_dir.joinpath(file_friendly_model_name))

            # clika_model = ClikaModule.clika_load(files_dir.joinpath(file_friendly_model_name))

            export_fp16: bool = False
            if deployment_settings.is_TensorRT_LLM_ONNX():
                export_fp16 = True
            with clika_model.export_configuration(export_int4=True, export_fp16=export_fp16):
                clika_model.clika_export(
                    file=str(files_dir.joinpath(file_friendly_model_name + ".onnx")),
                    example_inputs=clika_model.clika_generate_dummy_inputs(),
                    input_names=["pixel_values"],
                    output_names=["depth"],
                    dynamic_axes={
                        "pixel_values": {0: "bsz", 2: "H", 3: "W"}
                    },
                )


if __name__ == '__main__':
    main()
