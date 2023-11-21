# ViT Compressing Example
<!--TOC-->

- [Requirements](#requirements)
- [Prepare Dataset](#prepare-dataset)
  - [Option 1 - _ImageNet_ _(1000 classes)_](#option-1---imagenet-1000-classes)
  - [Option 2 - _ImageNette_ _(10 classes)_](#option-2---imagenette-10-classes)
- [Run Example](#run-example)
- [References](#references)

<!--TOC-->

_CLIKA SDK_ example to compress `ViT` model on `ImageNet` dataset

This examples requires "[vision](https://github.com/pytorch/vision)" repository by `PyTorch` to perform dataset related operation
and to add weight decay in order to conform with the state-of-the-art training protocol for `ViT`:

## Requirements

- Python >= 3.8
- CLIKA SDK (<https://docs.clika.io/docs/installation>)
- `pip install -r image_classification/vit/requirements.txt`
- download dependencies from pytorch/vision

```shell
# pwd: image_classification/vit
mkdir vision
cd vision
wget https://raw.githubusercontent.com/pytorch/vision/71968bc4afb8892284844a7c4cbd772696f42a88/references/classification/presets.py https://raw.githubusercontent.com/pytorch/vision/71968bc4afb8892284844a7c4cbd772696f42a88/references/classification/train.py https://raw.githubusercontent.com/pytorch/vision/71968bc4afb8892284844a7c4cbd772696f42a88/references/classification/transforms.py https://raw.githubusercontent.com/pytorch/vision/71968bc4afb8892284844a7c4cbd772696f42a88/references/classification/utils.py https://raw.githubusercontent.com/pytorch/vision/71968bc4afb8892284844a7c4cbd772696f42a88/references/classification/sampler.py
touch __init__.py
cd ..
```

## Prepare Dataset

Two dataset options are available.

- ImageNet ([link](https://www.image-net.org/download.php))
- ImageNette: smaller subset of ImageNet ([link](https://github.com/fastai/imagenette))

  > Note that `ImageNette` is a dummy dataset for testing purpose only.
  >
  > In order to get proper benchmark, download `ImageNet`.

### Option 1 - _ImageNet_ _(1000 classes)_

1. Visit the official [ImageNet website](https://www.image-net.org/download.php) to get full access to the dataset.

2. Place the dataset (`ILSVRC` folder) inside `image_classification/vit`

3. The final directory tree should look as following:

    ```text
    image_classification/vit
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

### Option 2 - _ImageNette_ _(10 classes)_

1. Run the following command line

    ```shell
    # pwd: image_classification/vit
    sh prepare_imagenette_dataset.sh
    ```

2. The final directory tree should look as following:

    ```text
    FOR ImageNette:

    image_classification/vit
    ├── imagenette2-160/
    │   ├── train/
    │   │   ├── n01440764/
    │   ├── test/
    ...
    │   ├── val/
    ...
    ```

## Run Example

```shell
# pwd: image_classification/vit
python3 vit_main.py
```

## References

<https://github.com/pytorch/vision/tree/main/references/classification>
