# CLIKA SDK

<!--TOC-->

- [Prerequisites](#prerequisites)
- [Datasets](#datasets)
- [Repository Structure](#repository-structure)
- [CLIKA Compression Examples Table](#clika-compression-examples-table)
- [Run Examples](#run-examples)
  - [Command line arguments](#command-line-arguments)
- [Docker Image](#docker-image)
  - [Usage](#usage)

<!--TOC-->

## Prerequisites

- Install CLIKA SDK (<https://docs.clika.io/docs/installation>)

## Datasets

All examples have a `README.md` file with instructions on how to prepare your environment and dataset and run the example.

Datasets will be downloaded to each example direct folder `examples/<task-name>/<model-name>/<dataset-name>`.

## Repository Structure

CLIKA Compression Examples

```text
clika_compression_examples/
├── examples
│   ├── common/ # common scripts to prepare datasets
│   ├── <task-1>/ # for example image_classification
│   │   ├── <model-1>/ # for example image_classification/mobilenet
│   │   │   ├── README.md # instruction on how to setup environment, prepare dataset, and run the example
│   │   │   ├── <model-1>_main.py # the file that contains the usage example
│   │   │   ├── prepare_<dataset-name>_dataset.sh # shell scrip used to prepare the dataset (see example's README.md)
│   │   │   ├── <dataset-name>/ # folder containing the dataset (needs to be downloaded as instructed in example's README.md)
│   │   │   ├── config.yml/ # a configuration file that contain the training and compression parameters
│   │   │   ├── requirements.txt # python requirements for the specific example
│   │   ├── <model-2>/
...
│   │   ├── <task-2>/
...
├── template\ # a template of how to create a new example
...
```

## CLIKA Compression Examples Table

- Model - The model's name
- Task - The model's objective
- Dataset - The default dataset used in the example
- Domain - The field of application
  - CV - Computer Vision

| Model                                                          | Task                                                    | Example Dataset       | Domain |
|:---------------------------------------------------------------|:--------------------------------------------------------|:----------------------|:-------|
| [MNIST](examples%2Fimage_classification%2Fmnist)               | [image_classification](examples%2Fimage_classification) | MNIST                 | CV     |
| [EfficientNet](examples%2Fimage_classification%2Fefficientnet) | [image_classification](examples%2Fimage_classification) | ImageNet / ImageNette | CV     |
| [MobileNet](examples%2Fimage_classification%2Fmobilenet)       | [image_classification](examples%2Fimage_classification) | ImageNet / ImageNette | CV     |
| [ResNet](examples%2Fimage_classification%2Fresnet)             | [image_classification](examples%2Fimage_classification) | ImageNet / ImageNette | CV     |
| [Visual Transformer (ViT)](examples%2Fimage_classification%2Fvit) | [image_classification](examples%2Fimage_classification) | ImageNet / ImageNette | CV     |
| [RetinaFace](examples%2Fobject_detection%2Fretinaface)         | [object_detection](examples%2Fobject_detection)         | WIDER FACE            | CV     |
| [RetinaNet](examples%2Fobject_detection%2Fretinanet)           | [object_detection](examples%2Fobject_detection)         | COCO                  | CV     |
| [YoloV7](examples%2Fobject_detection%2Fyolov7)                 | [object_detection](examples%2Fobject_detection)         | COCO                  | CV     |
| [YOLOX](examples%2Fobject_detection%2Fyolox)                   | [object_detection](examples%2Fobject_detection)         | COCO                  | CV     |
| [IMDN](examples%2Fsuper_resolution%2Fimdn)                     | [super_resolution](examples%2Fsuper_resolution)         | DIV2K and REDS4       | CV     |
| [6DRepNet](examples%2Fpose_estimation%2Fsixdrepnet)          | [pose_estimation](examples%2Fpose_estimation)           | 300W_LP and AFLW2000  | CV     |

## Run Examples

See [`README.md`](template%2FREADME.md) inside each example folder.
There are two main ways to change the configuration of the example,
via command line arguments when running [`main_<model_name>.py`](template%2Fmain.py),
and the example's [configuration yaml file](template%2Fconfig.yml).

For more information about the configuration file see the [docs](https://docs.clika.io/docs/conf_file)

### Command line arguments

All examples have some common command line arguments:

- **output_dir** - Output directory for saving checkpoints (`.pompom` files), logs, model architecture and deployed models
- **data** - Path to the dataset directory
- **ckpt** - Path to load the model checkpoints (e.g. `.pth` or `.pompom`)
- **batch_size** - Batch size for training and evaluation
- **lr** - Learning rate for the optimizer
- **workers** - Number of worker processes for dataloader

And some additional ones that are example specific (see examples arguments parser).

## Docker Image

We provide a simple [`clika_examples.Dockerfile`](%2Fclika_examples.Dockerfile) to set up an environment with PyTorch and CLIKA Compression.
It is based on the official [PyTorch `.Dockerfile`](https://hub.docker.com/layers/pytorch/pytorch/2.1.2-cuda11.8-cudnn8-devel/images/sha256-66b41f1755d9644f6341cf4053cf2beaf3948e2573acf24c3b4c49f55e82f578?context=explore)

### Usage

Requirements:

- [Docker](https://www.docker.com/) >= 24.0.6
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/overview.html) (since `clika-compression` requires CUDA)

To build the Docker Image

```shell
# pwd: clika_compression_examples/
export CC_LICENSE_KEY=<your-license-key>
docker build --build-arg CC_LICENSE_KEY=$CC_LICENSE_KEY --tag "clika_compression:latest" -f clika_examples.Dockerfile .
```

To run a container and the [MNIST](examples%2Fimage_classification%2Fmnist) example:

```shell
# pwd: clika_compression_examples/
docker run -it --shm-size 8G --gpus all  --entrypoint /bin/bash -v $PWD:/workspace:rw clika_compression
# install clika-compression (for more information see https://docs.clika.io/docs/installation)
export $CC_LICENSE_KEY={your_license_key}
pip install "clika-compression" --extra-index-url \
https://license:$CC_LICENSE_KEY@license.clika.io/simple
# cd to the specific example you are interested in
cd examples/image_classification/mnist
# follow the instructions on the specific `README.`md file for the example to prepare dataset and prerequisites
# for MNIST we only need to install torchmetrics
pip install torchmetrics==1.3.2
# run the compression example
python mnist_main.py
```
