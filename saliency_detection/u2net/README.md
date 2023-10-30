# U-2-Net Compressing Example
<ins>CLIKA SDK</ins> example to compress `U2-Net` model on `DUTS` dataset


# Requirements

- Python >= 3.8
- CLIKA SDK (https://docs.clika.io/docs/installation)
- Clone U2Net project & Install dependencies
```
# pwd: saliency_detection/u2net
git clone https://github.com/xuebinqin/U-2-Net.git
cd U-2-Net
git reset --hard 53dc9da026650663fc8d8043f3681de76e91cfde
cd ..
pip install -r requirements.txt

# download checkpoint
https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view
```
# Prepare Dataset

Run the following script

```
# pwd: saliency_detection/u2net
sh prepare_duts_dataset.sh
```

The dataset directory tree should look like the following:

```
duts
├── DUTS-TE
│   ├── DUTS-TE-Image
│   └── DUTS-TE-Mask
└── DUTS-TR
    ├── DUTS-TR-Image
    └── DUTS-TR-Mask
```

You can also download DUTS dataset from [official website](http://saliencydetection.net/duts/).

# Run Examples

```
# pwd: saliency_detection/u2net
python3 u2net_main.py
```


# References
https://github.com/xuebinqin/U-2-Net