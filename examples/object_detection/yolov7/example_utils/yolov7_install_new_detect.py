import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent / "yolov7"))
from models import yolo


def _new_forward(self, x):
    """
    Detect is the last component of YoloV7 graph
    Since we are focusing on the fine-tunable graph ignore `self.end2end`, `self.include_nms` and `self.concat`
    Assume `self.training = True`

    REFERENCE:
    https://github.com/WongKinYiu/yolov7/blob/84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca/models/yolo.py#L42
    """
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
        bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
    return x


def install_fixed_detect():
    # We have to do this until we support torch.compile(..., fullgraph=False) meaning we can compile submodules and not all-or-nothing
    yolo.Detect.forward = _new_forward
