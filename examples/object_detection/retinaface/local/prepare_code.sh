#!/usr/bin/env sh

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")/..

git clone https://github.com/biubug6/Pytorch_Retinaface.git ${SCRIPTPATH}/Pytorch_Retinaface
git -C ${SCRIPTPATH}/Pytorch_Retinaface reset --hard b984b4b775b2c4dced95c1eadd195a5c7d32a60b
mkdir -p ${SCRIPTPATH}/checkpoints
pip install gdown
gdown -O ${SCRIPTPATH}/checkpoints https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1 --folder
