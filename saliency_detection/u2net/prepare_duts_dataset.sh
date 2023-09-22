#!/bin/bash

SCRIPT_DIR=$(dirname -- "$0")
DATASET_DIR=${1:-$(dirname -- "$0")}
echo $SCRIPT_DIR
python3 ${SCRIPT_DIR}/example_utils/prepare_dataset.py --data_dir "$DATASET_DIR"
rm -f "$DATASET_DIR"/duts/DUTS-TR.zip
rm -f "$DATASET_DIR"/duts/DUTS-TE.zip
