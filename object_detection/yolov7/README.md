# YOLOv7 Compressing Example
<ins>CLIKA SDK</ins> example to compress `Yolov7` model on `COCO` dataset


# Requirements

- Install CLIKA SDK (https://docs.clika.io/docs/installation)
- Clone YOLOv7 project & Install dependencies
```
# pwd: object_detection/yolov7
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
git reset --hard 84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca
cd ..

# install requirements
pip install -r yolov7/requirements.txt
pip install torchmetrics

# download checkpoint
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

# Prepare Dataset

To download and prepare the dataset simply run the following command:

```
# pwd: object_detection/yolov7
sh ./yolov7/scripts/get_coco.sh
```

The dataset directory tree should look like the following:

```
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


# Run Examples

```
# pwd: object_detection/yolov7
python3 yolov7_main.py
```


# References
https://github.com/WongKinYiu/yolov7
