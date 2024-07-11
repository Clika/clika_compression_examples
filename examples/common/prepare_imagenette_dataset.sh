#!/bin/bash
SCRIPT_DIR=${1:-$(dirname -- "$0")}
echo "downloading files at: " ${SCRIPT_DIR} # relative directory path
wget -nc https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz -P "${SCRIPT_DIR}"
tar -xvzf "${SCRIPT_DIR}"/imagenette2-160.tgz -C "${SCRIPT_DIR}" --skip-old-files
rm "${SCRIPT_DIR}"/imagenette2-160.tgz
