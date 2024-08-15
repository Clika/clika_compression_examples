# IMDN Compression Example
<!--TOC-->

- [Requirements](#requirements)
- [Pre-Requisite](#pre-requisite)
- [Run Compression](#run-compression)
- [Deploy Compressed checkpoint](#deploy-compressed-checkpoint)
- [References](#references)

<!--TOC-->

**CLIKA SDK** example to compress `IMDN` model on `DIV2K` dataset

## Requirements

- 3.11 >= Python >= 3.8
- CLIKA SDK (<https://docs.clika.io/docs/installation>)
- `pip install -r ./requirements.txt`

## Pre-Requisite

```shell
sh prepare_code.sh
sh prepare_dataset.sh
```

If the above scripts completed successfully. You should be able to see 2 new directories under the current directory.
(`IMDN`, `dataset`)

```text
imdn/
# cloned repository
├── IMDN


# downloaded dataset
└── div2k
   ├── DIV2K_decoded
   ├── DIV2K_HR
   ├── DIV2K_LR_bicubic
   │   └── x4
   └── REDS4
       ├── blur
       ├── blur_bicubic
       ├── GT
       └── sharp_bicubic
```

## Run Compression

```shell
# single gpu
python3 imdn_main.py

# multi gpu
torchrun --nproc-per-node={num gpus} imdn_main.py
```

## Deploy Compressed checkpoint

```shell
python3 imdn_deloy.py {saved chkpt}
```

## References

<https://github.com/Zheng222/IMDN>
