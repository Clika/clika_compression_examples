#!/usr/bin/env sh

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")/..

git clone https://github.com/thohemp/6DRepNet ${SCRIPTPATH}/6DRepNet
git -C ${SCRIPTPATH}/6DRepNet reset --hard 0d4ccab11f49143f3e4638890d0f307f30b070f4
rm -rf ${SCRIPTPATH}/6DRepNet/.git

sed -i -e 's/torch.autograd.Variable/torch.tensor/g' ${SCRIPTPATH}/6DRepNet/sixdrepnet/utils.py

pip install gdown
gdown -O ${SCRIPTPATH}/checkpoints/ https://drive.google.com/uc\?id\=1PL-m9n3g0CEPrSpf3KwWEOf9_ZG-Ux1Z
gdown -O ${SCRIPTPATH}/checkpoints/ https://drive.google.com/uc\?id\=1vPNtVu_jg2oK-RiIWakxYyfLPA9rU4R4
