# IMDN Compressing Example
<ins>CLIKA SDK</ins> example to compress `IMDN` model on `DIV2K` dataset


# Requirements

- Install CLIKA SDK (https://docs.clika.io/docs/installation)
- Clone IMDN project & Install dependencies
```
# pwd: super_resolution/imdn
git clone https://github.com/Zheng222/IMDN.git
cd IMDN
git reset --hard 8f158e6a5ac9db6e5857d9159fd4a6c4214da574
cd ..
pip install -r requirements.txt
```

# Prepare Dataset

To download and prepare the dataset simply run the following command:

```
# pwd: super_resulution/IMDN
sh prepare_div2k_dataset.sh
```

The dataset directory tree should look like the following:

```
dataset
├── div2k
│   └── DIV2K_decoded
│       ├── DIV2K_HR
│       └── DIV2K_LR_bicubic
│           └── x4
└── REDS4
    ├── blur
    ├── blur_bicubic
    ├── GT
    └── sharp_bicubic
```


```
# pwd: super_resolution/imdn
python3 imdn_main.py
```

# References
https://github.com/Zheng222/IMDN
