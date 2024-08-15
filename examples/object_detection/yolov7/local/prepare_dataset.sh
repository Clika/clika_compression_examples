#!/usr/bin/env sh

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")/..

sh ${SCRIPTPATH}/yolov7/scripts/get_coco.sh
