import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent / "Pytorch_Retinaface"))
from data import WiderFaceDetection


class WiderFaceEvalDataset(WiderFaceDetection):
    """Custom eval loader
    No eval dataloader defined inside `biubug6/Pytorch_Retinaface` repository.

    REFERENCE
    https://github.com/biubug6/Pytorch_Retinaface/blob/b984b4b775b2c4dced95c1eadd195a5c7d32a60b/data/wider_face.py#L9
    """

    def __init__(self, txt_path: str, size: Optional[int] = None):
        """
        :param txt_path: WiderFace `label.txt` path (e.g. "widerface/train/label.txt")
        :param size: ...
        """
        super().__init__(txt_path, None)
        self.size = size

        self.pad_offset = 32
        self.bgr_mean = np.array([104, 117, 123], dtype="float32")

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        # preprocess without augmentation
        # https://github.com/biubug6/Pytorch_Retinaface/blob/b984b4b775b2c4dced95c1eadd195a5c7d32a60b/data/data_augment.py#L203-L206
        if self.size is None:
            _ = height % self.pad_offset
            hb = 0 if _ == 0 else self.pad_offset - _
            _ = width % self.pad_offset
            wb = 0 if _ == 0 else self.pad_offset - _
            if hb != 0 or wb != 0:
                img = cv2.copyMakeBorder(
                    img,
                    hb // 2,
                    hb - hb // 2,
                    wb // 2,
                    wb - wb // 2,
                    borderType=cv2.BORDER_CONSTANT,
                    value=self.bgr_mean.tolist(),
                )
        else:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        img = img.astype("float32") - self.bgr_mean
        img = img.transpose(2, 0, 1)  # HWC --> CHW

        labels = self.words[index]
        annotations = np.zeros((0, 4))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 4))
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        target[:, 0::2] /= width
        target[:, 1::2] /= height

        return torch.from_numpy(img), target
