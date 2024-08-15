#!/usr/bin/env sh

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")/..

# download widerface dataset
wget https://www.dropbox.com/scl/fi/k1kgeachzl9jlb5hph2gm/retinaface_gt_v1.1.zip\?rlkey\=3x6so7wm3m2rxjpk66odcny6c\&e\=2\&dl\=1 -O retinaface_gt_v1.1.zip -P ${SCRIPTPATH}
wget https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip -P ${SCRIPTPATH}
wget https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip -P ${SCRIPTPATH}
wget https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_test.zip -P ${SCRIPTPATH}

# unzip ${SCRIPTPATH}/wider_face_split.zip -d ${SCRIPTPATH}/widerface/
unzip ${SCRIPTPATH}/WIDER_test.zip -q -d ${SCRIPTPATH}/widerface/
unzip ${SCRIPTPATH}/WIDER_train.zip -q -d ${SCRIPTPATH}/widerface/
unzip ${SCRIPTPATH}/WIDER_val.zip -q -d ${SCRIPTPATH}/widerface/
unzip ${SCRIPTPATH}/retinaface_gt_v1.1.zip -d ${SCRIPTPATH}/widerface/

# cleanup unzipped `retinaface_gt_v1.1.zip``
mv ${SCRIPTPATH}/widerface/test/label.txt ${SCRIPTPATH}/widerface/WIDER_test
rm -r ${SCRIPTPATH}/widerface/test
mv ${SCRIPTPATH}/widerface/train/label.txt ${SCRIPTPATH}/widerface/WIDER_train
rm -r ${SCRIPTPATH}/widerface/train
mv ${SCRIPTPATH}/widerface/val/label.txt ${SCRIPTPATH}/widerface/WIDER_val
rm -r ${SCRIPTPATH}/widerface/val

# cleanup downloaded zip files
rm ${SCRIPTPATH}/WIDER_test.zip
rm ${SCRIPTPATH}/WIDER_train.zip
rm ${SCRIPTPATH}/WIDER_val.zip
rm ${SCRIPTPATH}/retinaface_gt_v1.1.zip

# rename the dataset folder
mv ${SCRIPTPATH}/widerface/WIDER_test ${SCRIPTPATH}/widerface/test
mv ${SCRIPTPATH}/widerface/WIDER_train ${SCRIPTPATH}/widerface/train
mv ${SCRIPTPATH}/widerface/WIDER_val ${SCRIPTPATH}/widerface/val
