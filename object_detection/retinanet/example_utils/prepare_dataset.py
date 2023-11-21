import argparse
import hashlib
import itertools
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from tqdm import tqdm
import cv2


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
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

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


def main(data_dir: str):
    data_dir = Path(data_dir).joinpath("coco")
    data_dir.mkdir(parents=True, exist_ok=True)

    download_data = [
        {
            "file_name": "val2017.zip",
            "url": "http://images.cocodataset.org/zips/val2017.zip",
            "md5": "442b8da7639aecaf257c1dceb8ba8c80",
        },
        {
            "file_name": "train2017.zip",
            "url": "http://images.cocodataset.org/zips/train2017.zip",
            "md5": "cced6f7f71b7629ddf16f17bbcfab6b2",
        },
        {
            "file_name": "test2017.zip",
            "url": "http://images.cocodataset.org/zips/test2017.zip",
            "md5": "77ad2c53ac5d0aea611d422c0938fb35",
        },
        {
            "file_name": "annotations_trainval2017.zip",
            "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "md5": "f4bbac642086de4f52a3fdda2de5fa2c",
        },
    ]

    for _data in download_data:
        _data_dir = data_dir.joinpath(_data["file_name"])
        download_url(
            url=_data["url"],
            save_dir=str(_data_dir.parent),
            filename=_data_dir.name,
            md5=_data["md5"],
        )

        with zipfile.ZipFile(str(_data_dir), "r") as zip_ref:
            zip_ref.extractall(str(_data_dir.parent))

    image_dir = data_dir.joinpath("images").resolve()
    shutil.rmtree(path=str(image_dir))
    image_dir.mkdir()
    (data_dir / "train2017").rename(image_dir / "train2017")
    (data_dir / "test2017").rename(image_dir / "test2017")
    (data_dir / "val2017").rename(image_dir / "val2017")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare COCO dataset ")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="path to data directory to save the dataset to ",
    )

    args = parser.parse_args()

    main(args.data_dir)
