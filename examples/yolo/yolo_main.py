import os
from pathlib import Path
import datasets
from typing import Tuple, List, Optional
import numpy as np
import torch
from clika_ace import (
    clika_compile,
    ClikaModule, QuantizationSettings,
    DeploymentSettings_TensorRT_ONNX, DeploymentSettings_OpenVINO_ONNX,
    BaseDeploymentSettings,
    PruningSettings,
    clika_move_to_device,
)
from ultralytics import YOLO
from ultralytics import download
from ultralytics.data.augment import LetterBox

from install_yolo_fixes import install_fixes
from yolo_utils import merge_post_processing_to_onnx


def generate_data(num_examples: int, cache_dir: Path) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    dataset = datasets.load_dataset("detection-datasets/coco", split="train", streaming=True,
                                    token=os.environ.get("HF_TOKEN", None),
                                    cache_dir=str(cache_dir)).take(num_examples)
    samples = list(dataset)
    transform = LetterBox(new_shape=(640, 640))
    # TODO Input images must be divisible by 32, architecture limitation because of strides.
    calibration_inputs = [
        torch.from_numpy(transform(image=np.array(x["image"].convert("RGB")))).permute(2, 0, 1).unsqueeze(0)
        for x in samples
    ]
    calibration_inputs.append(torch.cat([calibration_inputs.pop(-1) for _ in range(8)]))
    tracing_inputs = list(calibration_inputs)[-4:]
    tracing_inputs.extend([
        torch.from_numpy(
            LetterBox(new_shape=(736, 1280))(image=np.array(x["image"].convert("RGB")))).permute(2, 0, 1).unsqueeze(0)
        for x in samples[:2]
    ])
    tracing_inputs.append(torch.cat([tracing_inputs.pop(-1) for _ in range(2)]))

    calibration_inputs = [x.float() * 1 / 255.0 for x in calibration_inputs]
    tracing_inputs = [x.float() * 1 / 255.0 for x in tracing_inputs]
    return calibration_inputs, tracing_inputs


def run(
        model_name: str,
        deployment_settings: BaseDeploymentSettings,
        quantization_settings: Optional[QuantizationSettings],
        pruning_settings: Optional[PruningSettings],
        calibration_inputs: List[torch.Tensor],
        tracing_inputs: List[torch.Tensor],
        checkpoints_dir: Path,
        output_dir: Path,
        is_detection_model: bool = False,
):
    model_name_for_saving: str = f"{deployment_settings}_{model_name}"

    download(f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}.pt", checkpoints_dir, exist_ok=True)
    model = YOLO(model=str(checkpoints_dir.joinpath(f"{model_name}.pt")), task="detect").cuda()
    model.fuse()
    model.model.eval()
    with torch.no_grad():
        model.model(calibration_inputs[0])  # sanity test
    clika_model: ClikaModule = clika_compile(
        model=model.model,
        deployment_settings=deployment_settings,
        quantization_settings=quantization_settings,
        pruning_settings=pruning_settings,
        calibration_inputs=calibration_inputs,
        tracing_inputs=tracing_inputs,
    )

    clika_model.clika_save(output_dir.joinpath(model_name_for_saving))
    # clika_model = ClikaModule.clika_load(output_dir.joinpath(model_name_for_saving))

    onnx_save_path = output_dir.joinpath(model_name_for_saving + ".onnx")
    final_merged_onnx_save_path = onnx_save_path
    with clika_model.export_configuration():
        clika_model.clika_export(
            file=onnx_save_path,
            input_names=["xs"],
            dynamic_axes={
                "xs": {0: "bsz", 2: "H", 3: "W"}
            }
        )
    # merge postprocessing
    merge_post_processing_to_onnx(
        model_onnx_path=onnx_save_path, final_save_path=final_merged_onnx_save_path,
        is_detection_model=is_detection_model,
        is_export_mode=True  # returns just the processed outputs for detections
    )


def main():
    torch.cuda.set_device(0)
    install_fixes()
    base_dir = Path(__file__).parent
    checkpoints_dir = base_dir.joinpath("checkpoints")
    cache_dir = base_dir.joinpath("cache")
    output_dir = base_dir.joinpath("files")
    checkpoints_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    # Deployment Settings:
    all_deployment_settings = [
        DeploymentSettings_TensorRT_ONNX(),
        DeploymentSettings_OpenVINO_ONNX(),
    ]

    quantization_settings = QuantizationSettings(
        weights_utilize_full_int_range=True,
        prefer_weights_only_quantization=False,
        quantization_sensitivity_threshold=0.005,
        weights_num_bits=[8]
    )
    pruning_settings = PruningSettings()

    # Data preparation
    num_examples: int = 16
    calibration_inputs, tracing_inputs = generate_data(num_examples, cache_dir=cache_dir)
    calibration_inputs, tracing_inputs = clika_move_to_device((calibration_inputs, tracing_inputs), "cuda")

    # Compress all yolo models.
    detection_model_names: list = [
        "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
        "yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x",
        "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    ]
    segmentation_model_names: list = [
        "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg"
    ]
    for deployment_settings in all_deployment_settings:
        for model_name in detection_model_names:
            print(f"Running '{model_name}' - {deployment_settings}")
            run(
                model_name=model_name,
                deployment_settings=deployment_settings,
                quantization_settings=quantization_settings,
                pruning_settings=pruning_settings,
                calibration_inputs=calibration_inputs,
                tracing_inputs=tracing_inputs,
                checkpoints_dir=checkpoints_dir,
                output_dir=output_dir,
                is_detection_model=True
            )
    for deployment_settings in all_deployment_settings:
        for model_name in segmentation_model_names:
            print(f"Running '{model_name}' - {deployment_settings}")
            run(
                model_name=model_name,
                deployment_settings=deployment_settings,
                quantization_settings=quantization_settings,
                pruning_settings=pruning_settings,
                calibration_inputs=calibration_inputs,
                tracing_inputs=tracing_inputs,
                checkpoints_dir=checkpoints_dir,
                output_dir=output_dir,
            )


if __name__ == '__main__':
    main()
