# YOLOX Compressing Example
<ins>CLIKA SDK</ins> example to compress `YOLOX` model on `COCO` dataset


# Requirements

- Install CLIKA SDK (https://docs.clika.io/docs/installation)
- Clone YOLOX project & Install dependencies

```
# pwd: object_detection/yolox
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
git reset --hard ac58e0a5e68e57454b7b9ac822aced493b553c53
cd ..
pip install -r YOLOX/requirements.txt
pip install torchmetrics==1.0.1

# download checkpoint
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

# Prepare Dataset

Run the following script

```
# pwd: object_detection/yolox
sh prepare_coco_dataset.sh
```

The dataset directory tree should look like the following:

```
COCO
├── annotations
├── test2017
├── train2017
└── val2017
```

OR

You may download COCO dataset from [official website](https://cocodataset.org/#download) and unzip.

# Run Examples

```
# pwd: object_detection/yolox
python3 yolox_main.py
```

### Know Issues

If this error message appear

```
[1]    704647 segmentation fault (core dumped)  python yolox_main.py ...
```

One possible solution is to reduce the number of workers and batch size

# References

https://github.com/Megvii-BaseDetection/YOLOX
