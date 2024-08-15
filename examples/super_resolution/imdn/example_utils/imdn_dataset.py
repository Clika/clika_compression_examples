from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image


class REDS(torch.utils.data.Dataset):
    """REDS evaluation dataset
    x4 scaling only (x2, x3 not available)
    """

    def __init__(self, data_dir: str, scale=4):
        """
        :param data_dir: test data folder
        """
        assert scale == 4, "Only x4 scaling support for REDS"

        data_dir = Path(data_dir)
        dir_hr = data_dir.joinpath("GT")
        self.dir_hr = sorted((str(f) for f in dir_hr.rglob("*.png")))
        dir_lr = data_dir.joinpath("sharp_bicubic")
        self.dir_lr = sorted((str(f) for f in dir_lr.rglob("*.png")))

    def __getitem__(self, idx):
        """Parse dataloader output to satisfy CLIKA SDK input requirements.
        (https://docs.clika.io/docs/next/compression-constrains/cco_inputs_requirements#Dataset-Dataloader)
        """
        hr_img = Image.open(self.dir_hr[idx])
        lr_img = Image.open(self.dir_lr[idx])

        return TF.to_tensor(lr_img), TF.to_tensor(hr_img)

    def __len__(self):
        return len(self.dir_hr)
