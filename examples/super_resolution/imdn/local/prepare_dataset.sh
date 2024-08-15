#!/usr/bin/env sh

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")/..

mkdir -p ${SCRIPTPATH}/dataset
mkdir -p ${SCRIPTPATH}/dataset/div2k
mkdir -p ${SCRIPTPATH}/dataset/div2k/DIV2K_decoded

wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip -P ${SCRIPTPATH}/dataset/div2k/
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip -P ${SCRIPTPATH}/dataset/div2k/
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip -P ${SCRIPTPATH}/dataset/div2k/
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip -P ${SCRIPTPATH}/dataset/div2k/

unzip ${SCRIPTPATH}/dataset/div2k/DIV2K_train_LR_bicubic_X2.zip -d ${SCRIPTPATH}/dataset/div2k/
unzip ${SCRIPTPATH}/dataset/div2k/DIV2K_train_LR_bicubic_X3.zip -d ${SCRIPTPATH}/dataset/div2k/
unzip ${SCRIPTPATH}/dataset/div2k/DIV2K_train_LR_bicubic_X4.zip -d ${SCRIPTPATH}/dataset/div2k/
unzip ${SCRIPTPATH}/dataset/div2k/DIV2K_train_HR.zip -d ${SCRIPTPATH}/dataset/div2k/

mv ${SCRIPTPATH}/dataset/div2k/DIV2K_train_LR_bicubic/X2 ${SCRIPTPATH}/dataset/div2k/DIV2K_train_LR_bicubic/x2
mv ${SCRIPTPATH}/dataset/div2k/DIV2K_train_LR_bicubic/X3 ${SCRIPTPATH}/dataset/div2k/DIV2K_train_LR_bicubic/x3
mv ${SCRIPTPATH}/dataset/div2k/DIV2K_train_LR_bicubic/X4 ${SCRIPTPATH}/dataset/div2k/DIV2K_train_LR_bicubic/x4
mv ${SCRIPTPATH}/dataset/div2k/DIV2K_train_LR_bicubic ${SCRIPTPATH}/dataset/div2k/DIV2K_LR_bicubic
mv ${SCRIPTPATH}/dataset/div2k/DIV2K_train_HR ${SCRIPTPATH}/dataset/div2k/DIV2K_HR

mv ${SCRIPTPATH}/dataset/div2k/DIV2K_HR ${SCRIPTPATH}/dataset/div2k/DIV2K_decoded/DIV2K_HR
mv ${SCRIPTPATH}/dataset/div2k/DIV2K_LR_bicubic ${SCRIPTPATH}/dataset/div2k/DIV2K_decoded/DIV2K_LR_bicubic

#rm -f ${SCRIPTPATH}/div2k/DIV2K_train_LR_bicubic_X2.zip
#rm -f ${SCRIPTPATH}/div2k/DIV2K_train_LR_bicubic_X3.zip
#rm -f ${SCRIPTPATH}/div2k/DIV2K_train_LR_bicubic_X4.zip
#rm -f ${SCRIPTPATH}/div2k/DIV2K_train_HR.zip

wget https://github.com/JingyunLiang/VRT/releases/download/v0.0/testset_REDS4.tar.gz -P ${SCRIPTPATH}/dataset/
tar -xvzf ${SCRIPTPATH}/dataset/testset_REDS4.tar.gz -C ${SCRIPTPATH}/dataset/
#rm -f ${SCRIPTPATH}/dataset/testset_REDS4.tar.gz
mv ${SCRIPTPATH}/REDS4 ${SCRIPTPATH}/REDS
