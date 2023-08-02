#!/bin/bash

SCRIPT_DIR=$(dirname -- "$0")
echo $SCRIPT_DIR
python3 ${SCRIPT_DIR}/example_utils/prepare_dataset.py --data_dir ${SCRIPT_DIR}

# TODO: make yolov7 like folder with ln links?
