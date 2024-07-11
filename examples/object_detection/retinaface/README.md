# RetinaFace Compressing Example
<!--TOC-->

- [Requirements](#requirements)
- [Prepare Dataset](#prepare-dataset)
- [Run Example](#run-example)
- [Deploy `.pompom` file](#deploy-pompom-file)
- [References](#references)

<!--TOC-->

**CLIKA SDK** example to compress `RetinaFace` model on `WiderFace` dataset

## Requirements

- Python >= 3.8
- CLIKA SDK (<https://docs.clika.io/docs/installation>)
- Clone RetinaFace project & Install dependencies

```shell
# pwd: object_detection/retinaface
git clone https://github.com/biubug6/Pytorch_Retinaface.git
git -C Pytorch_Retinaface reset --hard b984b4b775b2c4dced95c1eadd195a5c7d32a60b

# install requirements
pip install -r requirements.txt gdown==5.1.0

# download checkpoints
gdown -O . https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1 --folder
mkdir -p weights && mv mobilenetV1X0.25_pretrain.tar weights/
```

## Prepare Dataset

To download and prepare the dataset simply run the following command:

```shell
# pwd: object_detection/retinaface
sh prepare_widerface_dataset.sh
```

The dataset directory tree should look like the following:

```text
object_detection/retinaface/
├── widerface/
│   ├── wider_face_split/
...
│   │   ├──label.txt
│   ├── WIDER_test/
│   │   ├──label.txt
│   │   ├──images/
│   │   │   ├──0--Parade/
│   │   │   │   ├──0_Parade_marchingband_1_9.jpg
...
│   ├── WIDER_train/
│   │   ├──label.txt
│   │   ├──images/
...
│   ├── WIDER_val/
│   │   ├──label.txt
│   │   ├──images/
...
```

## Run Example

```shell
# pwd: object_detection/retinaface
python3 retinaface_main.py --output_dir outputs
```

## Deploy `.pompom` file

```python
from clika_compression.utils import get_path_to_best_clika_state_result
from clika_compression import clika_deploy


# (OPTIONAL) find the best performing pompom file
best_pompom_file_path: str = get_path_to_best_clika_state_result(
    "outputs",
    key_name="mAP_map",
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

<https://github.com/biubug6/Pytorch_Retinaface>
