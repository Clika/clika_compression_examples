# YOLOv7 Compression Example
<!--TOC-->

- [Requirements](#requirements)
- [Pre-Requisite](#pre-requisite)
- [Run Compression](#run-compression)
- [Deploy Compressed checkpoint](#deploy-compressed-checkpoint)
- [References](#references)

<!--TOC-->

**CLIKA SDK** example to compress `Yolov7` model on `COCO` dataset

## Requirements

- 3.11 >= Python >= 3.8
- CLIKA SDK (<https://docs.clika.io/docs/installation>)
- `pip install -r ./requirements.txt`

## Pre-Requisite

```shell
sh prepare_code.sh
sh prepare_dataset.sh
```

If the above scripts completed successfully. You should be able to see 3 new directories under the current directory.
(`yolov7`, `checkpoints`, `coco`)

```text
yolov7/
# cloned repository
├── yolov7

# trained checkpoints
├── checkpoints/
│   └── yolov7.pt

# downloaded dataset
└── coco
  ├── annotations
  ├── images
  │   ├── test2017
  │   ├── train2017
  │   └── val2017
  └── labels
      ├── train2017
      └── val2017
```

## Run Compression

```shell
# single gpu
python3 yolov7_main.py

# multi gpu
torchrun --nproc-per-node={num gpus} yolov7_main.py
```

## Deploy Compressed checkpoint

```shell
python3 yolov7_deloy.py {saved chkpt}
```

## References

<https://github.com/WongKinYiu/yolov7>
