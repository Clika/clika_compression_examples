#!/usr/bin/env sh

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")/..

git clone https://github.com/WongKinYiu/yolov7.git ${SCRIPTPATH}/yolov7
git -C ${SCRIPTPATH}/yolov7 reset --hard 84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca
mkdir -p ${SCRIPTPATH}/checkpoints
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt -P ${SCRIPTPATH}/checkpoints
