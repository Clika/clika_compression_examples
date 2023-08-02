#!/bin/bash

SCRIPT_DIR=$(dirname -- "$0")


python3 ${SCRIPT_DIR}/example_utils/download_gdrive.py 1vgCABX1JI3NGBzsHxwBXlmRjaLV3NIsG "retinaface_gt_v1.1.zip"
wget https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip -P ${SCRIPT_DIR}
wget https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip -P ${SCRIPT_DIR}
wget https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_test.zip -P ${SCRIPT_DIR}

mkdir ${SCRIPT_DIR}/widerface
unzip ${SCRIPT_DIR}/wider_face_split.zip -d ${SCRIPT_DIR}/widerface/
unzip ${SCRIPT_DIR}/WIDER_test.zip -d ${SCRIPT_DIR}/widerface/
unzip ${SCRIPT_DIR}/WIDER_train.zip -d ${SCRIPT_DIR}/widerface/
unzip ${SCRIPT_DIR}/WIDER_val.zip -d ${SCRIPT_DIR}/widerface/
unzip ${SCRIPT_DIR}/retinaface_gt_v1.1.zip -d ${SCRIPT_DIR}/widerface/

mv ${SCRIPT_DIR}/widerface/test/label.txt ${SCRIPT_DIR}/widerface/WIDER_test
mv ${SCRIPT_DIR}/widerface/train/label.txt ${SCRIPT_DIR}/widerface/WIDER_train
mv ${SCRIPT_DIR}/widerface/val/label.txt ${SCRIPT_DIR}/widerface/WIDER_val

rm -r ${SCRIPT_DIR}/widerface/test
rm -r ${SCRIPT_DIR}/widerface/train
rm -r ${SCRIPT_DIR}/widerface/val

rm ${SCRIPT_DIR}/WIDER_test.zip
rm ${SCRIPT_DIR}/WIDER_train.zip
rm ${SCRIPT_DIR}/WIDER_val.zip
rm ${SCRIPT_DIR}/retinaface_gt_v1.1.zip

mv ${SCRIPT_DIR}/widerface/WIDER_test ${SCRIPT_DIR}/widerface/test
mv ${SCRIPT_DIR}/widerface/WIDER_train ${SCRIPT_DIR}/widerface/train
mv ${SCRIPT_DIR}/widerface/WIDER_val ${SCRIPT_DIR}/widerface/val
