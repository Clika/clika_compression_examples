import os
from typing import List, Optional

import PIL.Image
import datasets
import torch
import torchvision
from pathlib import Path
from torchvision.transforms import functional as TVF
from clika_ace import (
    clika_compile,
    ClikaModule, QuantizationSettings, BaseDeploymentSettings,
    DeploymentSettings_TensorRT_ONNX, DeploymentSettings_OpenVINO_ONNX,
    DeploymentSettings_TensorRT_LLM_ONNX,
    PruningSettings,
)


def run(
        model_name: str, images: List[PIL.Image.Image],
        deployment_settings: BaseDeploymentSettings,
        quantization_settings: Optional[QuantizationSettings],
        pruning_settings: Optional[PruningSettings],
        output_dir: Path) -> None:

    weights_enum: torchvision.models.WeightsEnum = torchvision.models.get_model_weights(model_name)["DEFAULT"]
    model = torchvision.models.get_model(name=model_name, weights=weights_enum)
    model.cuda()
    transform_fn = weights_enum.transforms()
    example_inputs = torch.cat(
        [
            transform_fn(
                TVF.pil_to_tensor(
                    x
                ).clone()  # cloning just in-case something operates in-place and still refers to the original memory
            ).unsqueeze(0) for x in images
        ], dim=0
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        model(example_inputs)  # sanity test

    clika_model: ClikaModule = clika_compile(
        model=model,
        deployment_settings=deployment_settings,
        quantization_settings=quantization_settings,
        pruning_settings=pruning_settings,
        calibration_inputs=list(example_inputs.split(16, 0)),
        tracing_inputs=[
            example_inputs[0:1],
            example_inputs[0:4],
        ]
    )
    clika_model.clika_save(output_dir.joinpath(f"{deployment_settings}_{model_name}"))

    # clika_model = ClikaModule.clika_load(output_dir.joinpath(f"{deployment_settings}_{model_name}"))

    with clika_model.export_configuration(export_int4=True):
        clika_model.clika_export(
            file=str(output_dir.joinpath(f"{model_name}_{deployment_settings}.onnx")),
            input_names=["xs"],
            output_names=["logits"],
            dynamic_axes={
                "xs": {0: "bsz"}
            },
        )


def main():
    torch.cuda.set_device(0)
    num_examples = 64
    base_dir = Path(__file__).parent
    cache_dir = base_dir.joinpath("cache")
    output_dir = base_dir.joinpath("files")
    output_dir.mkdir(exist_ok=True)
    dataset = datasets.load_dataset(
        # path="ILSVRC/imagenet-1k",  # requires request for that Dataset on HF, hence the HF_TOKEN
        path="timm/mini-imagenet",
        split="train",
        cache_dir=str(cache_dir),
        token=os.environ.get("HF_TOKEN", None),
        verification_mode=datasets.VerificationMode.NO_CHECKS,
        data_dir="data",  # applies to the mini-imagenet
        data_files=["train-00000-of-00013.parquet"]  # applies to the mini-imagenet
    ).shuffle(seed=69420).take(num_examples)
    images: List[PIL.Image.Image] = [x["image"].convert("RGB") for x in dataset]

    all_deployment_settings = [
        DeploymentSettings_TensorRT_ONNX(),
        DeploymentSettings_TensorRT_LLM_ONNX(),
        DeploymentSettings_OpenVINO_ONNX(),
    ]

    quantization_settings = QuantizationSettings(
        weights_utilize_full_int_range=True,
        prefer_weights_only_quantization=False,
        quantization_sensitivity_threshold=0.01,
        weights_only_quantization_block_size=[0, 32, 64, 128, 256],
        weights_num_bits=[8, 4]
    )
    pruning_settings = PruningSettings(mask_type="1:1")

    models: List[str] = [
        "resnet18",
        "resnet34", "resnet50",
        "resnet101", "resnet152",
        "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
        "vit_b_16",
        "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14",

        "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d", "wide_resnet50_2", "wide_resnet101_2",
        "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
        "densenet121",
        "densenet161", "densenet169", "densenet201",
        "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_y_3_2gf",
        "regnet_y_8gf", "regnet_y_16gf", "regnet_y_32gf", "regnet_y_128gf",
        "regnet_x_400mf", "regnet_x_800mf",
        "regnet_x_1_6gf", "regnet_x_3_2gf", "regnet_x_8gf", "regnet_x_16gf", "regnet_x_32gf",

        "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l",
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
        "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0",
        "inception_v3",
        "squeezenet1_0", "squeezenet1_1",
    ]
    for deployment_settings in all_deployment_settings:
        for model_name in models:
            print(f"Running '{model_name=}' - {deployment_settings=}")
            run(model_name=model_name, images=images,
                deployment_settings=deployment_settings,
                quantization_settings=quantization_settings,
                pruning_settings=pruning_settings,
                output_dir=output_dir)


if __name__ == '__main__':
    main()

