#!/usr/bin/env sh

SCRIPT_DIR=$(realpath $(dirname -- "$0"))
git clone https://github.com/ultralytics/ultralytics.git ${SCRIPT_DIR}/ultralytics
cd ${SCRIPT_DIR}/ultralytics && git pull && git checkout v8.3.77
mv ${SCRIPT_DIR}/ultralytics/ultralytics ${SCRIPT_DIR}/temp
rm -rf ${SCRIPT_DIR}/ultralytics
mv ${SCRIPT_DIR}/temp ${SCRIPT_DIR}/ultralytics