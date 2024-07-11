# IMDN Compressing Example
<!--TOC-->

- [Requirements](#requirements)
- [Prepare Dataset](#prepare-dataset)
- [Run Example](#run-example)
- [Deploy `.pompom` file](#deploy-pompom-file)
- [References](#references)

<!--TOC-->

**CLIKA SDK** example to compress `IMDN` model on `DIV2K` dataset

## Requirements

- Python >= 3.8
- CLIKA SDK (<https://docs.clika.io/docs/installation>)
- Clone IMDN project & Install dependencies

```shell
# pwd: super_resolution/imdn
git clone https://github.com/Zheng222/IMDN.git
git -C IMDN reset --hard 8f158e6a5ac9db6e5857d9159fd4a6c4214da574

# we need these commands since scikit image api has changed for newer versions
sed -i -e 's/from skimage.measure import compare_psnr as psnr/from skimage.metrics import peak_signal_noise_ratio as psnr/g' \
IMDN/utils.py
sed -i -e 's/from skimage.measure import compare_ssim as ssim/from skimage.metrics import structural_similarity as ssim /g' \
IMDN/utils.py

pip install -r requirements.txt
```

## Prepare Dataset

To download and prepare the dataset simply run the following command:

```shell
# pwd: super_resolution/IMDN
sh prepare_div2k_REDS_dataset.sh
```

The dataset directory tree should look like the following:

```text
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

## Run Example

```shell
# pwd: super_resolution/imdn
python3 imdn_main.py --output_dir outputs
```

## Deploy `.pompom` file

```python
from clika_compression.utils import get_path_to_best_clika_state_result
from clika_compression import clika_deploy


# (OPTIONAL) find the best performing pompom file
best_pompom_file_path: str = get_path_to_best_clika_state_result(
    "outputs",
    key_name="psnr",
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

<https://github.com/Zheng222/IMDN>
