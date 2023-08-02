import argparse
import hashlib
import itertools
import sys
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from tqdm import tqdm


def _calculate_md5(file_dir: str, chunk_size: int = 1024 * 1024) -> str:
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but indicates that we are
    # not using the MD5 checksum for cryptography. This enables its usage in restricted environments like FIPS. Without
    # it torchvision.datasets is unusable in these environments since we perform a MD5 check everywhere.
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(file_dir, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_integrity(file_dir: str, md5: Optional[str] = None) -> bool:
    if not Path(file_dir).exists():
        return False
    if md5 is None:
        return True
    return md5 == _calculate_md5(file_dir)


def download_url(
        url: str, save_dir: str, filename: Optional[str] = None, md5: Optional[str] = None
) -> None:
    """Download a file from an url and place it in save_dir.
    Args:
        url (str): URL to download file from
        save_dir (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    save_dir = Path(save_dir)
    file_dir = save_dir.joinpath(filename)
    if check_integrity(str(file_dir), md5) and file_dir.exists():
        print(f"Using downloaded and verified file: {file_dir}")
        return

    # download the file
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    _tmp_file = save_dir.joinpath(f"{filename}.temp")
    with _tmp_file.open("wb") as file:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            progress_bar.update(len(data))
    progress_bar.close()

    if check_integrity(str(_tmp_file), md5):
        _tmp_file.rename(file_dir)  # download success
    else:
        # _tmp_file.unlink(missing_ok=True)
        raise RuntimeError("An error occurred while downloading, please try again.")


def view(imgs: list, labels: list):
    imgs = sorted(imgs)
    labels = sorted(labels)

    for i, (img_dir, label_dir) in enumerate(zip(imgs, labels)):
        if i > 3:
            return
        img = cv2.imread(str(img_dir))
        label = cv2.imread(str(label_dir))
        img_label = np.concatenate((img, label), axis=1)
        cv2.imshow(f"{img_dir.stem}", img_label)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def check_if_dataset_exist(data_dir):
    imgs = list(data_dir.glob("DUTS-TR/*/*.jpg"))
    labels = list(data_dir.glob("DUTS-TR/*/*.png"))
    if not (Path(data_dir)).exists():
        print(f'generating dataset inside {Path(data_dir).resolve()}')
        return False
    if len(imgs) == 21106 and len(labels) == 21106:
        print('dataset already exists and include the exact number of files, will NOT recreate the dataset')
        return True
    else:
        print(
            f'Warning! dataset directory already exists but does NOT include all files please remove '
            f'{Path(data_dir).resolve()} before running this script again or using any examples')
        return True


def main(data_dir: str = None, show_imgs_labels=False):
    if data_dir is None:
        data_dir = Path(__file__).parent.joinpath("duts")
    else:
        data_dir = Path(data_dir).joinpath('duts')

    dataset_exists = check_if_dataset_exist(data_dir)
    if dataset_exists:
        if show_imgs_labels:  # to make sure that the labels and image are aligned
            imgs = list(data_dir.glob("DUTS-TR/*/*.jpg"))
            labels = list(data_dir.glob("DUTS-TR/*/*.png"))
            view(imgs, labels)
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    download_data = [
        {
            "file_name": "DUTS-TR.zip",
            "url": "http://saliencydetection.net/duts/download/DUTS-TR.zip",
            "md5": "30256edc326a4f6c880414f16114c356"  # md5sum DUTS-TR.zip
        },
        {
            "file_name": "DUTS-TE.zip",
            "url": "http://saliencydetection.net/duts/download/DUTS-TE.zip",
            "md5": "ed8b6d5995132fb2900657e46a584923"
        }
    ]

    for _data in download_data:
        _data_dir = data_dir.joinpath(_data["file_name"])
        download_url(
            url=_data["url"],
            save_dir=str(_data_dir.parent),
            filename=_data_dir.name,
            md5=_data["md5"]
        )

        with zipfile.ZipFile(str(_data_dir), "r") as zip_ref:
            zip_ref.extractall(str(_data_dir.parent))

    # https://github.com/xuebinqin/U-2-Net/issues/9#issuecomment-626903897
    # https://github.com/xuebinqin/BASNet/issues/4#issuecomment-498073313
    imgs = list(data_dir.glob("DUTS-TR/*/*.jpg"))
    labels = list(data_dir.glob("DUTS-TR/*/*.png"))
    for p in tqdm(itertools.chain(imgs, labels), total=len(imgs), desc='Preprocessing images'):
        img = cv2.imread(str(p))
        img = cv2.flip(img, 1)
        _new_file_name = f"{hash(p.stem)}{p.suffix}"
        cv2.imwrite(f"{p.parent.joinpath(_new_file_name)}", img)
    if show_imgs_labels:  # to make sure that the labels and image are aligned
        view(imgs, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare DUTS dataset ')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='path to data directory to save the dataset to ')
    parser.add_argument('--show_imgs_labels', action='store_true', default=False,
                        help='shows a few Images and labels for debugging purposes ')

    args = parser.parse_args()

    main(args.data_dir, args.show_imgs_labels)
