#!/bin/bash

SCRIPT_DIR=$(dirname -- "$0")
echo $SCRIPT_DIR
python3 ${SCRIPT_DIR}/../../common/download_coco.py --data_dir ${SCRIPT_DIR}/coco

rm -rf ${SCRIPT_DIR}/coco/images
mkdir ${SCRIPT_DIR}/coco/images
mv ${SCRIPT_DIR}/coco/train2017 ${SCRIPT_DIR}/coco/images/train2017
mv ${SCRIPT_DIR}/coco/test2017 ${SCRIPT_DIR}/coco/images/test2017
mv ${SCRIPT_DIR}/coco/val2017 ${SCRIPT_DIR}/coco/images/val2017
