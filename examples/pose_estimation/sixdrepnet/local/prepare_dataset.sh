#!/usr/bin/env sh

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")/..

echo "downloading files at: " ${SCRIPTPATH} # relative directory path

mkdir ${SCRIPTPATH}/6DRepNet/sixdrepnet/datasets/

pip install gdown
gdown https://drive.google.com/uc\?id\=0B7OEHD3T4eCkVGs0TkhUWFN6N1k
mv 300W-LP.zip ${SCRIPTPATH}/6DRepNet/sixdrepnet/datasets/
unzip -qq ${SCRIPTPATH}/6DRepNet/sixdrepnet/datasets/300W-LP.zip -d ${SCRIPTPATH}/6DRepNet/sixdrepnet/datasets/
rm ${SCRIPTPATH}/6DRepNet/sixdrepnet/datasets/300W-LP.zip

wget http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip
mv AFLW2000-3D.zip ${SCRIPTPATH}/6DRepNet/sixdrepnet/datasets/
unzip -qq ${SCRIPTPATH}/6DRepNet/sixdrepnet/datasets/AFLW2000-3D.zip -d ${SCRIPTPATH}/6DRepNet/sixdrepnet/datasets/
rm ${SCRIPTPATH}/6DRepNet/sixdrepnet/datasets/AFLW2000-3D.zip

python3 ${SCRIPTPATH}/6DRepNet/sixdrepnet/create_filename_list.py --root_dir ${SCRIPTPATH}/6DRepNet/sixdrepnet/datasets/300W_LP
python3 ${SCRIPTPATH}/6DRepNet/sixdrepnet/create_filename_list.py --root_dir ${SCRIPTPATH}/6DRepNet/sixdrepnet/datasets/AFLW2000
