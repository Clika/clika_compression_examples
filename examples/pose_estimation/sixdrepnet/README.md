# 6DRepNet Compression Example
<!--TOC-->

- [Requirements](#requirements)
- [Pre-Requisite](#pre-requisite)
- [Run Compression](#run-compression)
- [Deploy Compressed checkpoint](#deploy-compressed-checkpoint)
- [References](#references)

<!--TOC-->

**CLIKA SDK** example to compress `6DRepNet` model on `Pose_300W_LP` dataset, and evaluate on `AFLW2000` dataset

## Requirements

- 3.11 >= Python >= 3.8
- CLIKA SDK (<https://docs.clika.io/docs/installation>)
- `pip install -r ./requirements.txt`

## Pre-Requisite

```shell
sh prepare_code.sh
sh prepare_dataset.sh
```

If the above scripts completed successfully. You should be able to see 1 new directory under the current directory.
(`6DRepNet`)

```text
sixdrepnet/
# cloned repository & downloaded dataset
├── 6DRepNet
│   ├── sixdrepnet
│   │   └── datasets
│   │       ├── 300W_LP
│   │       └── AFLW2000
    ...
```

## Run Compression

```shell
# single gpu
python3 sixdrepnet_main.py

# multi gpu
torchrun --nproc-per-node={num gpus} sixdrepnet_main.py
```

## Deploy Compressed checkpoint

```shell
python3 sixdrepnet_deloy.py {saved chkpt}
```

## References

<https://github.com/thohemp/6DRepNet>
