import os
from typing import List, Optional, Dict, Tuple

import datasets
import torch
from pathlib import Path
from diffusers import (
    StableDiffusionPipeline, DiffusionPipeline
)
from itertools import product
from clika_ace import SimpleProgressBar
from clika_ace import (
    clika_compile,
    clika_install_inputs_catcher, clika_move_to_device,
    ClikaModule, QuantizationSettings, BaseDeploymentSettings,
    DeploymentSettings_TensorRT_ONNX, DeploymentSettings_OpenVINO_ONNX,
    DeploymentSettings_TensorRT_LLM_ONNX,
    DeploymentSettings_ONNXRuntime_ONNX,
    LayerQuantizationSettings,
    PruningSettings,
    LayerPruningSettings
)


def get_prompts(
        num_prompts_to_fetch: int,
        prompt_min_length: int,
        cache_dir: Path
) -> List[str]:
    assert num_prompts_to_fetch > 0
    assert prompt_min_length > 0
    _dataset = list(datasets.load_dataset(
        "Gustavosta/Stable-Diffusion-Prompts", split="train", cache_dir=str(cache_dir)
    ))
    _dataset = list(sorted(_dataset, key=lambda item: len(item["Prompt"])))
    _dataset = list(filter(lambda item: len(item["Prompt"]) > prompt_min_length, _dataset))
    unique_hashes: set = set()
    selected_prompts: List[str] = []
    while len(selected_prompts) < num_prompts_to_fetch:
        prompt: str = _dataset.pop(0)["Prompt"]
        _prompt_hash: int = hash(prompt.lower().replace(",", "_").replace("-", "_").replace(" ", "_"))
        if _prompt_hash in unique_hashes:
            continue
        unique_hashes.add(_prompt_hash)
        selected_prompts.append(prompt)

    return selected_prompts


def collect_unet_sample_data(
        pipeline: StableDiffusionPipeline,
        sample_texts: List[str],
        resolutions: List[Dict[str, int]],
        cache_results_path: Path,
        seed: int = 1234321,
) -> Tuple[List[dict], List[dict]]:
    cached_samples_path = cache_results_path
    if cached_samples_path.exists():
        unet_inputs, unet_tracing_inputs = torch.load(str(cached_samples_path))
    else:
        assert len(sample_texts) >= 2
        some_sample_texts = list(sample_texts)[:32]
        modified_text_samples: List[List[str]] = [[text] for text in some_sample_texts]
        last_items = sample_texts[-6:]
        modified_text_samples = modified_text_samples[:-2]
        modified_text_samples.append(last_items)  # we want a batched example
        print("Generating sample data")
        num_inference_timesteps: Optional[int] = 50
        all_combinations = list(product(modified_text_samples, resolutions))
        # debug_images: List[Tuple[str, PIL.Image.Image]] = []
        inputs_collected: List[List[dict]] = []
        for idx in range(len(all_combinations)):
            print(f"Generating sample: {idx + 1}/{len(all_combinations)}")
            prompt_batch, res = all_combinations[idx]
            # Need to reinitialize, the Pipeline modifies the seed somehow
            unet_callback = clika_install_inputs_catcher(pipeline.unet)
            generator = torch.Generator(device=pipeline.device).manual_seed(seed)
            result = pipeline(prompt=prompt_batch, **res, generator=generator, num_inference_steps=num_inference_timesteps)
            # debug_images.extend(list(zip(prompt_batch, result.images)))
            _collected_inputs: list = unet_callback()
            inputs_collected.append(_collected_inputs)

        unet_inputs = inputs_collected[-1]
        unet_tracing_inputs = sum(inputs_collected[:-1], [])
        torch.save((unet_inputs, unet_tracing_inputs), f=cached_samples_path)

    unet_tracing_inputs = unet_tracing_inputs[-64:]
    unet_inputs = unet_inputs[:16]
    return unet_inputs, unet_tracing_inputs


def collect_vae_sample_data(
        pipeline: StableDiffusionPipeline,
        sample_texts: List[str],
        resolutions: List[Dict[str, int]],
        cache_results_path: Path,
        seed: int = 1234321,
) -> Tuple[List[dict], List[dict]]:
    cached_samples_path = cache_results_path
    if cached_samples_path.exists():
        vae_inputs, vae_tracing_inputs = torch.load(str(cached_samples_path))
    else:
        assert len(sample_texts) >= 2
        some_sample_texts = list(sample_texts)[:32]
        modified_text_samples: List[List[str]] = [[text] for text in some_sample_texts]
        last_items = sample_texts[-6:]
        modified_text_samples = modified_text_samples[:-2]
        modified_text_samples.append(last_items)  # we want a batched example
        print("Generating sample data")
        num_inference_timesteps: Optional[int] = 50
        resolutions.append({"height": 64, "width": 64})  # to diversify inputs tracing
        all_combinations = list(product(modified_text_samples, resolutions))
        # debug_images: List[Tuple[str, PIL.Image.Image]] = []
        inputs_collected: List[List[dict]] = []
        for idx in range(len(all_combinations)):
            print(f"Generating sample: {idx + 1}/{len(all_combinations)}")
            prompt_batch, res = all_combinations[idx]
            # Need to reinitialize, the Pipeline modifies the seed somehow
            vae_callback = clika_install_inputs_catcher(pipeline.vae.decode)
            generator = torch.Generator(device=pipeline.device).manual_seed(seed)
            result = pipeline(prompt=prompt_batch, **res, generator=generator, num_inference_steps=num_inference_timesteps)
            # debug_images.extend(list(zip(prompt_batch, result.images)))
            _collected_inputs: list = vae_callback()
            inputs_collected.append(_collected_inputs)

        vae_inputs = inputs_collected[-1]
        vae_tracing_inputs = sum(inputs_collected, [])
        torch.save((vae_inputs, vae_tracing_inputs), f=cached_samples_path)
    vae_tracing_inputs = vae_tracing_inputs[-64:]
    vae_inputs = vae_inputs[:16]
    return vae_inputs, vae_tracing_inputs


def collect_text_encoder_sample_data(
        pipeline: StableDiffusionPipeline,
        sample_texts: List[str],
        seed: int = 1234321,
) -> List[dict]:
    assert len(sample_texts) >= 2
    num_inference_timesteps: Optional[int] = 1
    text_encoder_callback = clika_install_inputs_catcher(pipeline.text_encoder)
    modified_text_samples: List[List[str]] = [sample_texts[0]]
    modified_text_samples.append(list(sample_texts))  # we want a batched example
    for batched_prompts in modified_text_samples:
        # Need to reinitialize, the Pipeline modifies the seed somehow
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        # we dont really care about the height,width, we just want the Embeddings of the Text Encoder
        pipeline(prompt=batched_prompts, height=32, width=32, generator=generator, num_inference_steps=num_inference_timesteps)
    text_encoder_inputs: list = text_encoder_callback()  # grabbing the inputs to the Text Encoder
    assert len(text_encoder_inputs) > 0
    return text_encoder_inputs


def _compress_text_encoder(
        model: torch.nn.Module,
        tracing_inputs: list,
        deployment_settings: BaseDeploymentSettings,
        quantization_settings: Optional[QuantizationSettings],
        pruning_settings: Optional[PruningSettings],
        save_path_prefix: Path
):
    tracing_inputs = clika_move_to_device(tracing_inputs, "cuda")
    # Just export
    clika_text_encoder = clika_compile(
        model,
        calibration_inputs=tracing_inputs,
        deployment_settings=deployment_settings,
        quantization_settings=quantization_settings,
        pruning_settings=pruning_settings,
        logs_dir=save_path_prefix.parent.joinpath("logs"),
    )
    clika_text_encoder.clika_save(save_path_prefix)
    # clika_text_encoder = ClikaModule.clika_load(save_path_prefix)

    export_bf16: bool = False
    if deployment_settings.is_TensorRT_LLM_ONNX():
        export_bf16 = True
    with clika_text_encoder.export_configuration(export_int4=True, export_bf16=export_bf16):
        clika_text_encoder.clika_export(
            file=str(save_path_prefix) + ".onnx",
            example_inputs=clika_text_encoder.clika_generate_dummy_inputs(),
            input_names=["input_ids"],
            dynamic_axes={
                "input_ids": {0: "bsz", 1: "seq_length"},
            },
        )


def _compress_unet(
        model: torch.nn.Module,
        calibration_inputs: list,
        tracing_inputs: list,
        deployment_settings: BaseDeploymentSettings,
        quantization_settings: Optional[QuantizationSettings],
        pruning_settings: Optional[PruningSettings],
        save_path_prefix: Path
):
    calibration_inputs = clika_move_to_device(calibration_inputs, "cuda")
    clika_unet = clika_compile(
        model,
        calibration_inputs=calibration_inputs,
        tracing_inputs=tracing_inputs,
        deployment_settings=deployment_settings,
        quantization_settings=quantization_settings,
        pruning_settings=pruning_settings,
        logs_dir=save_path_prefix.parent.joinpath("logs"),
    )
    clika_unet.clika_save(save_path_prefix)
    # clika_unet = ClikaModule.clika_load(save_path_prefix)

    export_bf16: bool = False
    if deployment_settings.is_TensorRT_LLM_ONNX():
        export_bf16 = True
    with clika_unet.export_configuration(export_int4=True, export_bf16=export_bf16):
        clika_unet.clika_export(
            file=str(save_path_prefix) + ".onnx",
            example_inputs=clika_unet.clika_generate_dummy_inputs(),
            input_names=None,
            output_names=["out"],
            dynamic_axes={
                "sample": {0: "bsz", 2: "H", 3: "W"},
                "encoder_hidden_states": {0: "bsz"},
            },
        )


def _compress_vae(
        model: torch.nn.Module,
        tracing_inputs: list,
        calibration_inputs: list,
        deployment_settings: BaseDeploymentSettings,
        quantization_settings: Optional[QuantizationSettings],
        pruning_settings: Optional[PruningSettings],
        save_path_prefix: Path
):
    clika_vae = clika_compile(
        model.decode,
        tracing_inputs=tracing_inputs,
        calibration_inputs=calibration_inputs,
        deployment_settings=deployment_settings,
        quantization_settings=quantization_settings,
        pruning_settings=pruning_settings,
        logs_dir=save_path_prefix.parent.joinpath("logs"),
    )
    clika_vae.clika_save(save_path_prefix)
    # clika_vae = ClikaModule.clika_load(save_path_prefix)
    export_bf16: bool = False
    if deployment_settings.is_TensorRT_LLM_ONNX():
        export_bf16 = True
    with clika_vae.export_configuration(export_int4=True, export_bf16=export_bf16):
        clika_vae.clika_export(
            file=str(save_path_prefix) + ".onnx",
            example_inputs=clika_vae.clika_generate_dummy_inputs(),
            input_names=["z"],
            output_names=["sample"],
            dynamic_axes={
                "z": {
                    0: "bsz", 2: "H", 3: "W"
                },
            },
        )


def run(
        model_name: str,
        sample_texts: List[str],
        resolutions: List[Dict[str, int]],
        random_seed: int,
        deployment_settings: BaseDeploymentSettings,
        quantization_settings: Optional[QuantizationSettings],
        pruning_settings: Optional[PruningSettings],
        output_dir: Path,
        cache_dir: Path, ) -> None:
    file_friendly_model_name = model_name.replace("/", "_").replace(".", "_")

    # We have to disable the safety_checker. It kills images it if suspects there is NSFW in the image and returns zeroed out images (completley black)
    #   It has False Positives and it can ruin the calibration.
    pipeline: StableDiffusionPipeline = DiffusionPipeline.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        token=os.environ.get("HF_TOKEN", None)
    )
    pipeline.safety_checker = None

    pipeline.to("cuda")
    text_encoder_inputs: List[dict] = collect_text_encoder_sample_data(
        pipeline=pipeline, sample_texts=sample_texts,
        seed=random_seed
    )
    unet_inputs, unet_tracing_inputs = collect_unet_sample_data(
        pipeline=pipeline,
        sample_texts=sample_texts,
        resolutions=resolutions,
        seed=random_seed,
        cache_results_path=cache_dir.joinpath(f"{file_friendly_model_name}_unet_cached_data.pt")
    )
    vae_inputs, vae_tracing_inputs = collect_vae_sample_data(
        pipeline=pipeline,
        sample_texts=sample_texts,
        resolutions=resolutions,
        seed=random_seed,
        cache_results_path=cache_dir.joinpath(f"{file_friendly_model_name}_vae_cached_data.pt")
    )
    pipeline.to("cpu")  # just to save some space

    _compress_text_encoder(
        model=pipeline.text_encoder.cuda(),
        tracing_inputs=text_encoder_inputs,
        deployment_settings=deployment_settings,
        quantization_settings=None,
        pruning_settings=None,
        save_path_prefix=output_dir.joinpath(f"{file_friendly_model_name}_{deployment_settings}_text_encoder")
    )
    _compress_unet(
        model=pipeline.unet.cuda(),
        calibration_inputs=unet_inputs,
        tracing_inputs=unet_tracing_inputs,
        deployment_settings=deployment_settings,
        quantization_settings=quantization_settings,
        pruning_settings=pruning_settings,
        save_path_prefix=output_dir.joinpath(f"{file_friendly_model_name}_{deployment_settings}_unet")
    )
    _compress_vae(
        model=pipeline.vae.cuda(),
        calibration_inputs=vae_inputs,
        tracing_inputs=vae_tracing_inputs,
        deployment_settings=deployment_settings,
        quantization_settings=None,
        pruning_settings=None,
        save_path_prefix=output_dir.joinpath(f"{file_friendly_model_name}_{deployment_settings}_vae")
    )


def main():
    torch.cuda.set_device(0)
    base_dir = Path(__file__).parent
    cache_dir = base_dir.joinpath("cache")
    output_dir = base_dir.joinpath("files")
    output_dir.mkdir(exist_ok=True)
    random_seed: int = 12399321
    num_prompts_to_fetch: int = 1024
    prompt_min_length: int = 70

    all_deployment_settings = [
        DeploymentSettings_TensorRT_LLM_ONNX(),
        DeploymentSettings_TensorRT_ONNX(),
        DeploymentSettings_OpenVINO_ONNX(),
    ]
    pruning_settings = PruningSettings(mask_type="1:1", pruning_sensitivity_threshold=0.001)

    sample_inputs: List[str] = get_prompts(
        num_prompts_to_fetch=num_prompts_to_fetch,
        prompt_min_length=prompt_min_length, cache_dir=cache_dir
    )

    resolutions: List[Dict[str, int]] = [
        {"height": 512, "width": 512},
    ]

    all_model_names: list = [
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
    ]

    for deployment_settings in all_deployment_settings:
        if deployment_settings.is_TensorRT_ONNX() or deployment_settings.is_OpenVINO_ONNX():
            quantization_settings = QuantizationSettings(
                weights_utilize_full_int_range=True,
                weights_num_bits=[8, 4],
                quantization_sensitivity_threshold=0.00025,
                prefer_weights_only_quantization=False,
            )
        else:
            quantization_settings = QuantizationSettings(
                weights_utilize_full_int_range=True,
                weights_num_bits=[8, 4],
                quantization_sensitivity_threshold=0.00025,
                prefer_weights_only_quantization=True,
            )
        for model_name in all_model_names:
            run(
                model_name=model_name,
                sample_texts=sample_inputs,
                resolutions=resolutions,
                random_seed=random_seed,
                deployment_settings=deployment_settings,
                quantization_settings=quantization_settings,
                pruning_settings=pruning_settings,
                output_dir=output_dir,
                cache_dir=cache_dir,
            )


if __name__ == '__main__':
    main()
