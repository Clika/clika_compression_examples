# Segmentation Compression Example
<!--TOC-->

- [Requirements](#requirements)
- [Pre-Requisite](#pre-requisite)
- [Run Compression](#run-compression)
- [Deploy Compressed checkpoint](#deploy-compressed-checkpoint)
- [Supported Models](#supported-models)
- [References](#references)

<!--TOC-->

**CLIKA SDK** example to compress Segmentation models on `COCO` dataset

## Requirements

- 3.11 >= Python >= 3.8
- CLIKA SDK (<https://docs.clika.io/docs/installation>)
- `pip install -r ./requirements.txt`
- COCO dataset (<https://cocodataset.org/#home>)

## Pre-Requisite

```shell
sh prepare_code.sh
```

If the above script completed successfully. You should be able to see 1 new directory under the current directory.
(`torchvision_reference`)

```text
torchvision_models/
# cloned repository
└── torchvision_reference/
    ├── classification
    ├── detection
    ├── __init__.py
    ├── optical_flow
    ├── references
    ├── segmentation
    ├── similarity
    └── video_classification
```

## Run Compression

```shell
# single gpu
python3 tv_seg_models.py --model {model name}

# multi gpu
torchrun --nproc-per-node={num gpus} tv_seg_models.py --model {model name}
```

## Deploy Compressed checkpoint

```shell
python3 tv_seg_models_deploy.py {saved chkpt}
```

## Supported Models

```text
"fcn_resnet50",
"fcn_resnet101",
"deeplabv3_resnet50",
"deeplabv3_resnet101",
"deeplabv3_mobilenet_v3_large",
"lraspp_mobilenet_v3_large",
```

## References

<https://github.com/pytorch/vision/tree/main/references/segmentation>
