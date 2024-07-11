# 6DRepNet Compressing Example
<!--TOC-->

- [Requirements](#requirements)
- [Prepare Dataset](#prepare-dataset)
- [Run Example](#run-example)
- [Deploy `.pompom` file](#deploy-pompom-file)
- [References](#references)

<!--TOC-->

**CLIKA SDK** example to compress `6DRepNet` model on `Pose_300W_LP` dataset, and evaluate on `AFLW2000` dataset

## Requirements

- Python >= 3.8
- CLIKA SDK (<https://docs.clika.io/docs/installation>)
- Clone 6DRepNet project & Install dependencies

```shell
# pwd: pose_estimation/sixdrepnet
git clone https://github.com/thohemp/6DRepNet
git -C 6DRepNet reset --hard 0d4ccab11f49143f3e4638890d0f307f30b070f4

# install requirements
pip install -r 6DRepNet/requirements.txt
pip install gdown torchmetrics==1.3.2

# download checkpoints
gdown -O 6DRepNet/sixdrepnet/ https://drive.google.com/uc\?id\=1PL-m9n3g0CEPrSpf3KwWEOf9_ZG-Ux1Z; gdown https://drive.google.com/uc\?id\=1vPNtVu_jg2oK-RiIWakxYyfLPA9rU4R4
```

## Prepare Dataset

To download and prepare the dataset simply run the following command:

```shell
# pwd: pose_estimation/sixdrepnet
pip install gdown
sh prepare_300W_LP_AFLW2000_dataset.sh
python3 6DRepNet/sixdrepnet/create_filename_list.py --root_dir 6DRepNet/sixdrepnet/datasets/300W_LP
python3 6DRepNet/sixdrepnet/create_filename_list.py --root_dir 6DRepNet/sixdrepnet/datasets/AFLW2000
```

The dataset directory tree should look like the following:

```text
├── 6DRepNet
    ├── sixdrepnet
        └── datasets
            ├── 300W_LP
            └── AFLW2000
```

## Run Example

```shell
# pwd: pose_estimation/sixdrepnet
python3 sixdrepnet_main.py
```

## Deploy `.pompom` file

```python
from clika_compression.utils import get_path_to_best_clika_state_result
from clika_compression import clika_deploy


# (OPTIONAL) find the best performing pompom file
best_pompom_file_path: str = get_path_to_best_clika_state_result(
    "outputs",
    key_name="eval_metric_total",
    summary_json_group="evaluation",
    find_lowest=True,
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

<https://github.com/thohemp/6DRepNet>
