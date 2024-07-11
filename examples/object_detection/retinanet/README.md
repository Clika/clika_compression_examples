# RetinaNet Compressing Example
<!--TOC-->

- [Requirements](#requirements)
- [Prepare Dataset](#prepare-dataset)
- [Run Example](#run-example)
- [Deploy `.pompom` file](#deploy-pompom-file)
- [References](#references)

<!--TOC-->

**CLIKA SDK** example to compress `RetinaNet` model on `COCO` dataset

## Requirements

- Python >= 3.8
- CLIKA SDK (<https://docs.clika.io/docs/installation>)
- Clone RetinaNet project & Install dependencies

```shell
# pwd: object_detection/retinanet
git clone https://github.com/yhenon/pytorch-retinanet.git

# install requirements
pip install -r requirements.txt gdown==5.1.0

# download checkpoint
gdown --fuzzy https://drive.google.com/file/d/1yLmjq3JtXi841yXWBxst0coAgR26MNBS/view
```

## Prepare Dataset

To download and prepare the dataset simply run the following command:

```shell
# pwd: object_detection/retinanet
pip install tqdm
sh prepare_coco_dataset.sh
```

The dataset directory tree should look like the following:

```text
coco
├── annotations
└── images
    ├── test2017
    ├── train2017
    └── val2017
```

OR

You may download COCO dataset from [official website](https://cocodataset.org/#download) and unzip.

## Run Example

```shell
# pwd: object_detection/retinanet
python3 retinanet_main.py --output_dir outputs
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

<https://github.com/yhenon/pytorch-retinanet>
