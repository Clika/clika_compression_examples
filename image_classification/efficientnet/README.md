# EfficientNet Compressing Example

<ins>CLIKA SDK</ins> example to compress `EfficientNet` model on `ImageNet` dataset.

This examples requires "[vision](https://github.com/pytorch/vision)" repository by `PyTorch` to perform dataset
related operation and to add weight decay in order to conform with the state-of-the-art training protocol for EfficientNet:

# Requirements

- Python >= 3.8
- CLIKA SDK (https://docs.clika.io/docs/installation)
- `pip install -r image_classification/efficientnet/requirements.txt`
- Download dependencies from pytorch/vision
    ```commandline
    # pwd: image_classification/efficientnet
    mkdir vision
    cd vision
    wget https://raw.githubusercontent.com/pytorch/vision/71968bc4afb8892284844a7c4cbd772696f42a88/references/classification/presets.py https://raw.githubusercontent.com/pytorch/vision/71968bc4afb8892284844a7c4cbd772696f42a88/references/classification/train.py https://raw.githubusercontent.com/pytorch/vision/71968bc4afb8892284844a7c4cbd772696f42a88/references/classification/transforms.py https://raw.githubusercontent.com/pytorch/vision/71968bc4afb8892284844a7c4cbd772696f42a88/references/classification/utils.py https://raw.githubusercontent.com/pytorch/vision/71968bc4afb8892284844a7c4cbd772696f42a88/references/classification/sampler.py
    touch __init__.py
    cd ..
    ```

# Prepare Dataset

Two dataset options are available.

- ImageNet ([link](https://www.image-net.org/download.php))
- ImageNette: smaller subset of ImageNet ([link](https://github.com/fastai/imagenette))

  > Note that `ImageNette` is a dummy dataset for testing purpose only.
  >
  > In order to get proper benchmark, download `ImageNet`.

### Option 1 - _ImageNet_ _(1000 classes)_

1. Visit the official [ImageNet website](https://www.image-net.org/download.php) to get full access to the dataset.

2. Place the dataset (`ILSVRC` folder) inside `image_classification/efficientnet`

3. The final directory tree should look as following:

    ```
    image_classification/efficientnet
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
    ```
    # pwd: image_classification/efficientnet 
    sh prepare_imagenette_dataset.sh
    ```
2. The final directory tree should look as following:
    ```
    FOR ImageNette:
    
    image_classification/efficientnet
    ├── imagenette2-160/
    │   ├── train/
    │   │   ├── n01440764/ 
    │   ├── test/
    ...
    │   ├── val/
    ...
    ```

# Run Examples

```
# pwd: image_classification/efficientnet
python3 efficientnet_main.py
```

# References

https://github.com/pytorch/vision/tree/main/references/classification
