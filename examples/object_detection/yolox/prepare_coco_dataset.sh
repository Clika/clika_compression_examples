#!/bin/bash

SCRIPT_DIR=$(dirname -- "$0")
echo "downloading files at: " ${SCRIPT_DIR} # relative directory path
echo $SCRIPT_DIR
python3 ${SCRIPT_DIR}/../../common/download_coco.py --data_dir ${SCRIPT_DIR}/COCO
