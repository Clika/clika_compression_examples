# RetinaNet Compressing Example
<!--TOC-->

- [Requirements](#requirements)
- [Prepare Dataset](#prepare-dataset)
- [Run Example](#run-example)
- [References](#references)

<!--TOC-->

_CLIKA SDK_ example to compress `RetinaNet` model on `COCO` dataset

This examples requires "[pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet)" repository load the RetinaNet
model and dataset operation to the optimizer in order to conform with the RetinaNet repository as close as possible

## Requirements

- Python >= 3.8
- CLIKA SDK (<https://docs.clika.io/docs/installation>)
- Clone RetinaNet project & Install dependencies

```shell
# pwd: object_detection/retinanet
git clone https://github.com/yhenon/pytorch-retinanet.git

# install requirements
pip install -r requirements.txt
```

Download trained checkpoint `coco_resnet_50_map_0_335_state_dict.pt` from the following [link](https://drive.google.com/file/d/1yLmjq3JtXi841yXWBxst0coAgR26MNBS/view).

Place it under `object_detection/retinanet`.

## Prepare Dataset

To download and prepare the dataset simply run the following command:

```shell
# pwd: object_detection/retinanet
sh prepare_coco_dataset.sh
```

The dataset directory tree should look like the following:

```text
coco
├── annotations
├── test2017
├── train2017
└── val2017
```

OR

You may download COCO dataset from [official website](https://cocodataset.org/#download) and unzip.

## Run Example

```shell
# pwd: object_detection/retinanet
python3 retinanet_main.py
```

## References

<https://github.com/yhenon/pytorch-retinanet>
