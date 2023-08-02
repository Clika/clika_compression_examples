#!/bin/bash

SCRIPT_DIR=$(dirname -- "$0")
echo $SCRIPT_DIR
python3 ${SCRIPT_DIR}/example_utils/prepare_dataset.py --data_dir ${SCRIPT_DIR}
rm -f ${SCRIPT_DIR}/duts/DUTS-TR.zip
rm -f ${SCRIPT_DIR}/duts/DUTS-TE.zip
