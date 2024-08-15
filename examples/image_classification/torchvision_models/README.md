# Classification Compression Example
<!--TOC-->

- [Requirements](#requirements)
- [Pre-Requisite](#pre-requisite)
- [Run Compression](#run-compression)
- [Deploy Compressed checkpoint](#deploy-compressed-checkpoint)
- [Known Issues](#known-issues)
- [Supported Models](#supported-models)
- [References](#references)

<!--TOC-->

**CLIKA SDK** example to compress Classification models on `ImageNet` dataset

## Requirements

- 3.11 >= Python >= 3.8
- CLIKA SDK (<https://docs.clika.io/docs/installation>)
- `pip install -r ./requirements.txt`
- ImageNet dataset (<https://www.image-net.org/index.php>)

```text
# Expected ImageNet folder hierarchy
imagenet
├── ILSVRC/
│   ├── Annotations/
    ...
│   ├── Data/
│   │   ├── CLS-LOC/
│   │   │   ├── train/
│   │   │   │   ├── n01440764/
                ...
│   │   │   ├── test/
            ...
│   │   │   ├── val/
            ...
```

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
python3 tv_cls_models.py --model {model name}
```

## Deploy Compressed checkpoint

```shell
python3 tv_cls_models_deploy.py {saved chkpt}
```

## Known Issues

- MaxViT Pre-Compiling time takes long.

## Supported Models

```text
"vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "maxvit_t",
"resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
"resnext50_32x4d", "resnext101_32x8d",
"mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
"efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l",
"convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
"alexnet",
"densenet121", "densenet161", "densenet169", "densenet201",
"vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
"regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_y_3_2gf", "regnet_y_8gf", "regnet_y_16gf", "regnet_y_32gf", "regnet_y_128gf", "regnet_x_400mf", "regnet_x_800mf", "regnet_x_1_6gf", "regnet_x_3_2gf", "regnet_x_8gf", "regnet_x_16gf", "regnet_x_32gf",
"shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0",
```

## References

<https://github.com/pytorch/vision/tree/main/references/classification>
