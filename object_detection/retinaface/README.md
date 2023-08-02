# RetinaFace Compressing Example

<ins>CLIKA SDK</ins> example to compress `RetinaFace` model on `WiderFace` dataset

This examples requires "[Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface.git)" repository load the RetinaFace
model and dataset  operation to the optimizer in order to conform with the RetinaFace repository as close as possible


# Requirements

- Install CLIKA SDK (https://docs.clika.io/docs/installation)
- Clone RetinaFace project & Install dependencies

```
# pwd: object_detection/retinaface
git clone https://github.com/biubug6/Pytorch_Retinaface.git
cd Pytorch_Retinaface
git reset --hard b984b4b775b2c4dced95c1eadd195a5c7d32a60b
cd ..

# install requirements
pip install -r requirements.txt
```

Download trained checkpoint `Resnet50_Final.pth` from the following [link](https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1).

Place it under `object_detection/retinaface`.

# Prepare Dataset

To download and prepare the dataset simply run the following command:

```
# pwd: object_detection/retinaface
sh prepare_widerface_dataset.sh
```

The dataset directory tree should look like the following:

```
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

# Run Examples

```
# pwd: object_detection/retinaface
python3 retinaface_main.py
```

# References

https://github.com/biubug6/Pytorch_Retinaface