#!/bin/bash
SCRIPT_DIR=${1:-$(dirname -- "$0")}
echo "downloading files at: " ${SCRIPT_DIR} # relative directory path

mkdir "${SCRIPT_DIR}"/6DRepNet/sixdrepnet/datasets/

gdown https://drive.google.com/uc\?id\=0B7OEHD3T4eCkVGs0TkhUWFN6N1k
mv 300W-LP.zip "${SCRIPT_DIR}"/6DRepNet/sixdrepnet/datasets/
unzip -qq "${SCRIPT_DIR}"/6DRepNet/sixdrepnet/datasets/300W-LP.zip -d "${SCRIPT_DIR}"/6DRepNet/sixdrepnet/datasets/
rm "${SCRIPT_DIR}"/6DRepNet/sixdrepnet/datasets/300W-LP.zip

wget http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip
mv AFLW2000-3D.zip "${SCRIPT_DIR}"/6DRepNet/sixdrepnet/datasets/
unzip -qq "${SCRIPT_DIR}"/6DRepNet/sixdrepnet/datasets/AFLW2000-3D.zip -d "${SCRIPT_DIR}"/6DRepNet/sixdrepnet/datasets/
rm "${SCRIPT_DIR}"/6DRepNet/sixdrepnet/datasets/AFLW2000-3D.zip
