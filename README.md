# Prerequisites

- Install CLIKA SDK (https://docs.clika.io/docs/installation)

# Datasets

All examples have a `README.md` file with instructions on how to prepare your environment and dataset and run the example

Datasets will be downloaded to `<task-name>/<model-name>/<dataset-name>`

# Repository Structure

CLIKA Compression Examples

```
clika_compression_examples/
├── <task-1>/ # for example image_classification
│   ├── <model-1>/ # for example image_classification/mobilenet
│   │   ├── <model-1>_main.py # the file that contine the usage example
│   │   ├── prepare_<dataset-name>_dataset.sh # shell scrip used to prerare the dataset (see specific example README.md)
│   │   ├── <dataset-name>/ # folder containing the dataset (needs to be downloaded as insructed in the specific README.md)
│   │   ├── example_utils/ # small repository with neccessary utilities for the example
│   │   ├── requirements.txt # python requirements for the specific example
│   │   ├── README.md # instruction on how to setup enviroment prepare dataset and run the example
│   ├── <model-2>/
...
├── <task-2>/
...
```

# CLIKA Compression Examples Table

- Example Script - link to example python file
- Model - The model's name
- Task - The model's objective
- Dataset - The default dataset used in the example
- Domain - The field of application
    - CV - Computer Vision

| Model                                                  | Task                                         | Example Dataset       | Domain |
|:-------------------------------------------------------|:---------------------------------------------|:----------------------|:-------|
| [MNIST](image_classification%2Fmnist)                  | [image_classification](image_classification) | MNIST                 | CV     |
| [EfficientNet](image_classification%2Fefficientnet)    | [image_classification](image_classification) | ImageNet / ImageNette | CV     |
| [MobileNet](image_classification%2Fmobilenet)          | [image_classification](image_classification) | ImageNet / ImageNette | CV     |
| [ResNet](image_classification%2Fresnet)                | [image_classification](image_classification) | ImageNet / ImageNette | CV     |
| [Visual Transformer (ViT)](image_classification%2Fvit) | [image_classification](image_classification) | ImageNet / ImageNette | CV     |
| [RetinaFace](object_detection%2Fretinaface)            | [object_detection](object_detection)         | WIDER FACE            | CV     |
| [RetinaNet](object_detection%2Fretinanet)              | [object_detection](object_detection)         | COCO                  | CV     |
| [YoloV7](object_detection%2Fyolov7)                    | [object_detection](object_detection)         | COCO                  | CV     |
| [YOLOX](object_detection%2Fyolox)                      | [object_detection](object_detection)         | COCO                  | CV     |
| [U2-Net](saliency_detection%2Fu2net)                   | [saliency_detection](saliency_detection)     | DUTS                  | CV     |
| [IMDN](super_resolution%2Fimdn)                        | [super_resolution](super_resolution)         | DIV2K and REDS4       | CV     |

# Run Examples

See `README.md` inside each model folder
all examples has the same command line argument which are:

### CLIKA Engine Training Settings

- **target_framework** - Choose the targe framework TensorFlow Lite or TensorRT for deployment
- **data** - Path to the dataset directory
- **steps_per_epoch** - Steps per epoch during the compression
- **evaluation_steps** - Steps per epoch during the model evaluation for the initial model and for the compressed model after each epoch
- **scans_steps** - Number of steps to collect model Outputs to determine initial quantization parameters
- **print_interval** - Each how many steps to print out the information about the compression process
- **ma_window_size** - The logs contains the running average of the loss in addition to the current one, this will set the window size for the running average
- **save_interval** - Each how many steps the compressed model's checkpoint will be saved as `.pompom` files
- **reset_train_data** - Reset training dataset between epochs
- **reset_eval_data** - Reset evaluation dataset between epochs
- **grads_acc_steps** - (useful for larger model that must run with a smaller batch size)
- **no_mixed_precision** - turn off mixed precision when compressing the model, mixed precision uses FP16 for the weights, FP32 for the gradients. Activations can be either (all examples are using mixed precision by default)
- **lr_warmup_epochs** - Number of epochs of the Learning Rate warmup stage
- **lr_warmup_steps_per_epoch** - Number of steps each epoch of the Learning Rate warmup stage
- **fp16_weights** - Use FP16 weights when compressing  (can reduce VRAM usage)
- **gradients_checkpoint** - Use gradient checkpointing that offloads the activations to CPU after the forward pass of each layer and reload them when preforming backpropagation

### Model Training Setting

- **epochs** - Number of epochs to train the model
- **batch_size** - Batch size for training and evaluation
- **lr** - Learning rate for the optimizer
- **workers** - Number of worker processes for dataloader
- **ckpt** - Path to load the model checkpoints (e.g. .pth or .pompom)
- **output_dir** - Output directory for saving checkpoints (`.pompom` files), logs, model architecture and deployed models
- **train_from_scratch** - Ignoring `--ckpt` (if given) and training the model from scratch

### Quantization Config

- **weights_num_bits** - How many bits to use for the Weights for Quantization
- **activations_num_bits** - How many bits to use for the Activation for Quantization

## Docker Image

We provide a simple [`clika_examples.Dockerfile`](..%2Fclika_examples.Dockerfile) to set up an environment with PyTorch and `CLIKA Compression`.
It is based on the official [PyTorch `.Dockerfile`](https://hub.docker.com/layers/pytorch/pytorch/2.0.1-cuda11.7-cudnn8-devel/images/sha256-4f66166dd757752a6a6a9284686b4078e92337cd9d12d2e14d2d46274dfa9048?context=explore)

### Usage

Requirements:

- [Docker](https://www.docker.com/) > 19.03
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/overview.html) (`clika-compression` requires CUDA)

To build the Docker Image

```commandline
# pwd: clika_compression_examples/
export CC_LICENSE_KEY=<your-license-key> 
docker build --build-arg CC_LICENSE_KEY=$CC_LICENSE_KEY --tag "clika_compression:latest" -f clika_examples.Dockerfile .
```

To run a container and the [MNIST](image_classification%2Fmnist) example:

```commandline
docker run -it --shm-size 8G --gpus all  --entrypoint /bin/bash -v $PWD:/workspace:rw clika_compression
# folow the instructions on the specific `README.`md file for the example
# for MNIST simply install `requirements.txt` and run example  
pip install -r image_classification/mnist/requirements.txt 
python image_classification/mnist/mnist_main.py
```