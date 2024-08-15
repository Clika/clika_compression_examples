# RetinaFace Compression Example
<!--TOC-->

- [Requirements](#requirements)
- [Pre-Requisite](#pre-requisite)
- [Run Compression](#run-compression)
- [Deploy Compressed checkpoint](#deploy-compressed-checkpoint)
- [References](#references)

<!--TOC-->

**CLIKA SDK** example to compress `RetinaFace` model on `WiderFace` dataset

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
(`Pytorch_Retinaface`, `checkpoints`, `widerface`)

```text
retinaface/
# cloned repository
├── Pytorch_Retinaface

# trained checkpoints
├── checkpoints/
│   ├── mobilenet0.25_Final.pth
│   ├── mobilenetV1X0.25_pretrain.tar
│   └── Resnet50_Final.pth

# downloaded dataset
└── widerface/
    ├── wider_face_split/
    ...
    │   ├──label.txt
    ├── WIDER_test/
    │   ├──label.txt
    │   ├──images/
    │   │   ├──0--Parade/
    │   │   │   ├──0_Parade_marchingband_1_9.jpg
    ...
    ├── WIDER_train/
    │   ├──label.txt
    │   ├──images/
    ...
    ├── WIDER_val/
    │   ├──label.txt
    │   ├──images/
    ...
```

## Run Compression

```shell
# single gpu
python3 retinaface_main.py

# multi gpu
torchrun --nproc-per-node={num gpus} retinaface_main.py
```

## Deploy Compressed checkpoint

```shell
python3 retinaface_deloy.py {saved chkpt}
```

## References

<https://github.com/biubug6/Pytorch_Retinaface>
