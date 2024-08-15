#!/usr/bin/env sh

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")/..

git clone https://github.com/Zheng222/IMDN.git ${SCRIPTPATH}/IMDN
git -C ${SCRIPTPATH}/IMDN reset --hard 8f158e6a5ac9db6e5857d9159fd4a6c4214da574

# we need these commands since scikit image api has changed for newer versions
sed -i -e 's/from skimage.measure import compare_psnr as psnr/from skimage.metrics import peak_signal_noise_ratio as psnr/g' \
	${SCRIPTPATH}/IMDN/utils.py
sed -i -e 's/from skimage.measure import compare_ssim as ssim/from skimage.metrics import structural_similarity as ssim /g' \
	${SCRIPTPATH}/IMDN/utils.py

echo "Creating empty __init__.py recursively at " ${SCRIPTPATH}/IMDN
find ${SCRIPTPATH}/IMDN -not -path '*/.*' -type d -exec touch "{}/__init__.py" \;
