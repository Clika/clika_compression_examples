#!/bin/bash
SCRIPT_DIR=$(dirname -- "$0")
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz -P ${SCRIPT_DIR}
tar -xvzf ${SCRIPT_DIR}/imagenette2-160.tgz -C ${SCRIPT_DIR}
rm ${SCRIPT_DIR}/imagenette2-160.tgz
