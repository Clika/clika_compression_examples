#!/usr/bin/env sh

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")/..

URL="https://github.com/pytorch/vision.git"
DIR_NAME="torchvision_reference"
FULL_PATH="$SCRIPTPATH/$DIR_NAME"
BRANCH=v0.18.0
echo "Cloning repo"
git clone --filter=blob:none --no-checkout ${URL} ${FULL_PATH} --quiet
cd ${FULL_PATH}
git sparse-checkout set --cone --quiet
git checkout ${BRANCH} --quiet
git sparse-checkout set references/ --quiet
echo "Flattening out the directories a bit"
mv ${FULL_PATH}/references/* ${FULL_PATH}
#rm -r ${FULL_PATH}/references/

echo "Creating empty __init__.py recursively at " ${FULL_PATH}
find ${FULL_PATH} -not -path '*/.*' -type d -exec touch "{}/__init__.py" \;
#rm -rf ${FULL_PATH}/.git
