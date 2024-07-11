# YOLOv7 Compressing Example
<!--TOC-->

- [Requirements](#requirements)
- [Prepare Dataset](#prepare-dataset)
- [Run Example](#run-example)
- [Deploy `.pompom` file](#deploy-pompom-file)
- [References](#references)

<!--TOC-->

**CLIKA SDK** example to compress `Yolov7` model on `COCO` dataset

## Requirements

- Python >= 3.8
- CLIKA SDK (<https://docs.clika.io/docs/installation>)
- Clone YOLOv7 project & Install dependencies

```shell
# pwd: object_detection/yolov7
git clone https://github.com/WongKinYiu/yolov7.git
git -C yolov7 reset --hard 84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca

# install requirements
pip install -r yolov7/requirements.txt
pip install torchmetrics==1.3.2 pycocotools

# download checkpoint
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

## Prepare Dataset

To download and prepare the dataset simply run the following command:

```shell
# pwd: object_detection/yolov7
sh ./yolov7/scripts/get_coco.sh
```

The dataset directory tree should look like the following:

```text
coco
├── annotations
├── images
│   ├── test2017
│   ├── train2017
│   └── val2017
└── labels
    ├── train2017
    └── val2017
```

OR

You may download COCO dataset from [official website](https://cocodataset.org/#download) and unzip.

## Run Example

```shell
# pwd: object_detection/yolov7
python3 yolov7_main.py --output_dir outputs
```

## Deploy `.pompom` file

```python
from clika_compression.utils import get_path_to_best_clika_state_result
from clika_compression import clika_deploy


# (OPTIONAL) find the best performing pompom file
best_pompom_file_path: str = get_path_to_best_clika_state_result(
    "outputs",
    key_name="mAP_map",
    summary_json_group="evaluation",
    find_lowest=False,
)

deployed_model_path: str = clika_deploy(
    clika_state_path=best_pompom_file_path,
    output_dir_path="outputs",

    # (OPTIONAL)set this if you wish to use dynamic shapes
    # input_shapes=[(None, 3, None, None)],

    graph_author="CLIKA",
    graph_description="",
)
```

## References

<https://github.com/WongKinYiu/yolov7>
