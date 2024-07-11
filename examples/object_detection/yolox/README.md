# YOLOX Compressing Example
<!--TOC-->

- [Requirements](#requirements)
- [Prepare Dataset](#prepare-dataset)
- [Run Example](#run-example)
- [Deploy `.pompom` file](#deploy-pompom-file)
- [Known issues](#known-issues)
- [References](#references)

<!--TOC-->

**CLIKA SDK** example to compress `YOLOX` model on `COCO` dataset

## Requirements

- Python >= 3.8
- CLIKA SDK (<https://docs.clika.io/docs/installation>)
- Clone YOLOX project & Install dependencies

```shell
# pwd: object_detection/yolox
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
git -C YOLOX reset --hard ac58e0a5e68e57454b7b9ac822aced493b553c53

pip install -r YOLOX/requirements.txt
pip install torchmetrics==1.3.2 pycocotools

# download checkpoint
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

## Prepare Dataset

Run the following script

```shell
# pwd: object_detection/yolox
sh prepare_coco_dataset.sh
```

The dataset directory tree should look like the following:

```text
COCO
├── annotations
├── test2017
├── train2017
└── val2017
```

OR

You may download COCO dataset from [official website](https://cocodataset.org/#download) and unzip.

## Run Example

```shell
# pwd: object_detection/yolox
python3 yolox_main.py --output_dir outputs
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

## Known issues

When running YOLOX with number of workers greater than `1`
you may encounter the following error which is caused by `SIGKILL` signal, and it is not recoverable:

```shell
...
[<date-time>] Starting training.
[<date-time>] Starting Warmup.
[<date-time>] FATAL Failed to determine reason of failure!
```

This error may occur when handling multiple processes and attempting to initialize shared resources by pycocotools.
To resolve this error, we recommend reducing the number of workers,
ideally to a single process (`worker=1`). This adjustment should resolve the issue.

Please note that this error is expected to be addressed in the forthcoming release
of clika-compression (v0.4.0).

## References

<https://github.com/Megvii-BaseDetection/YOLOX>
