import gc
import os
import sys
from typing import List, Optional, Tuple, Callable, Dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import (
    CLIPModel, CLIPProcessor
)
from PIL import Image
import requests
import datasets
from pathlib import Path
from clika_ace import (
    clika_compile,
    ClikaModule, QuantizationSettings, PruningSettings,
    BaseDeploymentSettings,
    DeploymentSettings_TensorRT_ONNX,
    DeploymentSettings_OpenVINO_ONNX,
    DeploymentSettings_TensorRT_LLM_ONNX,
)
from clip_model_wrappers import CLIPVisionModel, CLIPTextModel


def _free_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_datacomp_dataset_samples(cache_dir: Path, num_examples: int) -> List[dict]:
    dataset = iter(datasets.load_dataset(
        path="apple/DataCompDR-12M",
        split="train",
        cache_dir=str(cache_dir),
        streaming=True))
    samples: List[dict] = []
    while len(samples) <= num_examples:
        item = next(dataset)
        try:
            img_url: str = item["url.txt"]
            texts: List[str] = item["syn.json"]["syn_text"]
            image = Image.open(requests.get(img_url, stream=True).raw)
            samples.append({"images": [image], "texts": texts})
        except:
            pass

    return samples


def _compress_vision_model(
        model: CLIPModel,
        vision_inputs: List[dict],
        deployment_settings: BaseDeploymentSettings,
        quantization_settings: QuantizationSettings,
        pruning_settings: Optional[PruningSettings],
        save_path_prefix: Path,
):
    vision_model = CLIPVisionModel(model)
    with torch.no_grad():
        vision_model(**vision_inputs[0])  # sanity test
    clika_vision_model: ClikaModule = clika_compile(
        model=vision_model,
        deployment_settings=deployment_settings,
        quantization_settings=quantization_settings,
        pruning_settings=pruning_settings,
        calibration_inputs=vision_inputs,
        forward_fn=lambda model, xs: model(**xs),
    )
    clika_vision_model.clika_save(save_path_prefix)
    # clika_vision_model = ClikaModule.clika_load(save_path_prefix)
    output_names: list = ["image_features"]
    export_bf16: bool = False
    if deployment_settings.is_TensorRT_LLM_ONNX():
        export_bf16 = True
    with clika_vision_model.export_configuration(export_int4=True, export_bf16=export_bf16):
        clika_vision_model.clika_export(
            file=str(save_path_prefix)+".onnx",
            example_inputs=clika_vision_model.clika_generate_dummy_inputs(),
            input_names=["pixel_values"],
            output_names=output_names,
            dynamic_axes={
                "pixel_values": {0: "bsz"},
            },
        )


def _compress_text_model(
        model: CLIPModel,
        text_inputs: List[dict],
        deployment_settings: BaseDeploymentSettings,
        quantization_settings: QuantizationSettings,
        pruning_settings: Optional[PruningSettings],
        save_path_prefix: Path,
):
    text_model = CLIPTextModel(model)
    with torch.no_grad():
        text_model(**text_inputs[0])  # sanity test
    clika_text_model: ClikaModule = clika_compile(
        model=text_model,
        deployment_settings=deployment_settings,
        quantization_settings=quantization_settings,
        pruning_settings=pruning_settings,
        calibration_inputs=text_inputs,
        forward_fn=lambda model, xs: model(**xs),
    )
    clika_text_model.clika_save(save_path_prefix)
    # clika_text_model = ClikaModule.clika_load(save_path_prefix)

    new_input_names: list = []
    new_output_names: list = ["text_features"]
    dynamic_axes: dict = {
        "input_ids": {0: "bsz", 1: "seq_length"},
        "attention_mask": {0: "bsz", 1: "seq_length"},
    }
    export_bf16: bool = False
    if deployment_settings.is_TensorRT_LLM_ONNX():
        export_bf16 = True
    with clika_text_model.export_configuration(export_int4=True, export_bf16=export_bf16):
        clika_text_model.clika_export(
            file=str(save_path_prefix) + ".onnx",
            example_inputs=clika_text_model.clika_generate_dummy_inputs(),
            input_names=new_input_names,
            output_names=new_output_names,
            dynamic_axes=dynamic_axes
        )


def run(
        model_name: str,
        save_path_dir: Path,
        cache_dir: Path,
        raw_samples: List[dict],
        deployment_settings: BaseDeploymentSettings,
        quantization_settings: Optional[QuantizationSettings],
        pruning_settings: Optional[PruningSettings],
) -> None:
    # We need to be very mindful about how, where we create the model and how long the references to the model are made.
    # It will occupy GPU memory
    file_friendly_model_name = model_name
    if quantization_settings is not None:
        file_friendly_model_name = f"{file_friendly_model_name}_{quantization_settings.quantization_sensitivity_threshold:3.3f}"
    file_friendly_model_name = file_friendly_model_name.replace("/", "_").replace(".", "_").replace("-", "_")

    target_dtype = torch.float32
    if deployment_settings.is_TensorRT_LLM_ONNX():
        target_dtype = torch.bfloat16
    processor: CLIPProcessor = CLIPProcessor.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=str(cache_dir),
        padding_side="left",
    )
    raw_samples: list = list(raw_samples)
    raw_samples.append({
        "images": raw_samples[-1]["images"] + raw_samples[-2]["images"] + raw_samples[-3]["images"],
        "texts": raw_samples[-1]["texts"] + raw_samples[-2]["texts"] + raw_samples[-3]["texts"],
    })

    vision_inputs: List[dict] = []
    text_inputs: List[dict] = []
    for item in raw_samples:
        samples = dict(processor(text=item["texts"], images=item["images"], padding=True, return_tensors="pt").to("cuda"))
        vision_inputs.append({
            "pixel_values": samples.pop("pixel_values")
        })
        text_inputs.append(samples)
    clip_model_init: Callable[[], CLIPModel] = lambda : CLIPModel.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=str(cache_dir),
        return_dict=False,
        device_map="cpu").cuda().to(target_dtype)

    # this will also deallocate the 'clip_model', we will need to reinitialize.
    _compress_vision_model(
        model=clip_model_init(),
        vision_inputs=vision_inputs,
        deployment_settings=deployment_settings,
        quantization_settings=quantization_settings,
        pruning_settings=pruning_settings,
        save_path_prefix=save_path_dir.joinpath(f"{file_friendly_model_name}_vision_model_{deployment_settings}")
    )
    _free_memory()
    _compress_text_model(
        model=clip_model_init(),
        text_inputs=text_inputs,
        deployment_settings=deployment_settings,
        quantization_settings=quantization_settings,
        pruning_settings=pruning_settings,
        save_path_prefix=save_path_dir.joinpath(f"{file_friendly_model_name}_text_model_{deployment_settings}")
    )
    _free_memory()


def main():
    torch.cuda.set_device(0)
    base_dir = Path(__file__).parent
    files_dir = base_dir.joinpath("files")
    cache_dir = base_dir.joinpath("cache")
    files_dir.mkdir(exist_ok=True)

    all_deployment_settings = [
        DeploymentSettings_TensorRT_ONNX(),
        DeploymentSettings_TensorRT_LLM_ONNX(),
        DeploymentSettings_OpenVINO_ONNX(),
    ]

    pruning_settings = PruningSettings(pruning_sensitivity_threshold=0.001)

    raw_samples: list = get_datacomp_dataset_samples(cache_dir=cache_dir, num_examples=32)

    all_model_names: List[str] = [
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14",
        "openai/clip-vit-large-patch14-336",
        "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",

        # TODO Not possible to initialize the following models directly from Transformers Huggingface,
        #  maybe we can find more that do work.
        # "laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg",
        # "laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup",
        # "laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k",
        # "laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k",
        # "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    ]

    tested_out_quantization_sensitivities: dict = {
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": 0.35,
        "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K": 0.35,
        "openai/clip-vit-large-patch14": 0.015,
        "openai/clip-vit-large-patch14-336": 0.020,

        # These need some q_sens tuning:
        "openai/clip-vit-base-patch16": 0.015,
        "openai/clip-vit-base-patch32": 0.015,
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k": 0.015,
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": 0.015,
    }

    for deployment_settings in all_deployment_settings:
        if deployment_settings.is_TensorRT_ONNX() or deployment_settings.is_OpenVINO_ONNX():
            quantization_settings = QuantizationSettings(
                weights_utilize_full_int_range=True,
                weights_num_bits=[8, 4],
                quantization_sensitivity_threshold=0.015,
                prefer_weights_only_quantization=False,
            )
        else:
            quantization_settings = QuantizationSettings(
                weights_utilize_full_int_range=True,
                weights_num_bits=[8, 4],
                quantization_sensitivity_threshold=0.015,
                prefer_weights_only_quantization=True,
            )
        for model_name in all_model_names:
            quantization_settings.quantization_sensitivity_threshold = tested_out_quantization_sensitivities.get(model_name, 0.015)
            try:
                run(
                    model_name=model_name,
                    save_path_dir=files_dir,
                    cache_dir=cache_dir,
                    raw_samples=raw_samples,
                    deployment_settings=deployment_settings,
                    quantization_settings=quantization_settings,
                    pruning_settings=pruning_settings,
                )
            except Exception as e:
                print(f"Failed to run: {model_name} - {deployment_settings}", file=sys.stderr)
                raise e


if __name__ == '__main__':
    main()
