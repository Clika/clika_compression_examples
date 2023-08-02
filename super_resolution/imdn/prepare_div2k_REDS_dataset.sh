#!/bin/bash
SCRIPT_DIR=$(dirname -- "$0")
mkdir -p ${SCRIPT_DIR}/dataset
mkdir -p ${SCRIPT_DIR}/dataset/div2k
mkdir -p ${SCRIPT_DIR}/dataset/div2k/DIV2K_decoded

wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip -P ${SCRIPT_DIR}/dataset/div2k/
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip -P ${SCRIPT_DIR}/dataset/div2k/


unzip ${SCRIPT_DIR}/dataset/div2k/DIV2K_train_LR_bicubic_X4.zip -d ${SCRIPT_DIR}/dataset/div2k/
unzip ${SCRIPT_DIR}/dataset/div2k/DIV2K_train_HR.zip -d ${SCRIPT_DIR}/dataset/div2k/

mv ${SCRIPT_DIR}/dataset/div2k/DIV2K_train_LR_bicubic/X4 ${SCRIPT_DIR}/dataset/div2k/DIV2K_train_LR_bicubic/x4
mv ${SCRIPT_DIR}/dataset/div2k/DIV2K_train_LR_bicubic ${SCRIPT_DIR}/dataset/div2k/DIV2K_LR_bicubic
mv ${SCRIPT_DIR}/dataset/div2k/DIV2K_train_HR ${SCRIPT_DIR}/dataset/div2k/DIV2K_HR

mv ${SCRIPT_DIR}/dataset/div2k/DIV2K_HR ${SCRIPT_DIR}/dataset/div2k/DIV2K_decoded/DIV2K_HR
mv ${SCRIPT_DIR}/dataset/div2k/DIV2K_LR_bicubic ${SCRIPT_DIR}/dataset/div2k/DIV2K_decoded/DIV2K_LR_bicubic

rm -f ${SCRIPT_DIR}/div2k/DIV2K_train_LR_bicubic_X4.zip
rm -f ${SCRIPT_DIR}/div2k/DIV2K_train_HR.zip

wget https://github.com/JingyunLiang/VRT/releases/download/v0.0/testset_REDS4.tar.gz -P ${SCRIPT_DIR}/dataset/
tar -xvzf  ${SCRIPT_DIR}/dataset/testset_REDS4.tar.gz -C ${SCRIPT_DIR}/dataset/
rm -f ${SCRIPT_DIR}/dataset/testset_REDS4.tar.gz