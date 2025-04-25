import torch
import math
from typing import List
from ultralytics.nn.modules.block import (
    SPPF,
    C2f, C2fAttn, RepNCSPELAN4, SPPELAN, C3f, HGBlock,
)
from ultralytics.nn.modules.head import (
    Detect, v10Detect, WorldDetect, Pose, Segment, OBB
)
from ultralytics.nn.tasks import BaseModel
from ultralytics.utils.tal import dist2bbox, make_anchors


def _fix_SPPF(self, x):
    """
    Has to be fixed since torch.dynamo (torch 2.3.1) fails to understand the original expression
    Which is:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))  <-- Problem, 'generator' statement
        return self.cv2(torch.cat_(y, 1))
    """
    x = self.cv1(x)
    y1 = self.m(x)
    y2 = self.m(y1)
    return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


def _fix_C2f(self, x):
    """
    Has to be fixed since torch.dynamo (torch 2.3.1) fails to understand the original expression
    Which is:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)  <-- Problem, 'generator' statement
        return self.cv2(torch.cat(y, 1))
    """
    y = list(self.cv1(x).chunk(2, 1))
    for m in self.m:
        y.append(m(y[-1]))
    return self.cv2(torch.cat(y, 1))


def _fix_C2fAttn(self, x, guide):
    """
    Has to be fixed since torch.dynamo (torch 2.3.1) fails to understand the original expression
    Which is:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)  <-- Problem, 'generator' statement
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))
    """
    y = list(self.cv1(x).chunk(2, 1))
    for m in self.m:
        y.append(m(y[-1]))
    y.append(self.attn(y[-1], guide))
    return self.cv2(torch.cat(y, 1))


def _fix_RepNCSPELAN4(self, x):
    """
    Has to be fixed since torch.dynamo (torch 2.3.1) fails to understand the original expression
    Which is:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])  <-- Problem, 'generator' statement
        return self.cv4(torch.cat(y, 1))
    """
    y = list(self.cv1(x).chunk(2, 1))
    for m in [self.cv2, self.cv3]:
        y.append(m(y[-1]))
    return self.cv4(torch.cat(y, 1))


def _fix_SPPELAN(self, x):
    """
    Has to be fixed since torch.dynamo (torch 2.3.1) fails to understand the original expression
    Which is:
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])  <-- Problem, 'generator' statement
        return self.cv5(torch.cat(y, 1))
    """
    y = [self.cv1(x)]
    for m in [self.cv2, self.cv3, self.cv4]:
        y.append(m(y[-1]))
    return self.cv5(torch.cat(y, 1))


def _fix_C3f(self, x):
    """
    Has to be fixed since torch.dynamo (torch 2.3.1) fails to understand the original expression
    Which is:
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)  <-- Problem, 'generator' statement
        return self.cv3(torch.cat(y, 1))
    """
    y = [self.cv2(x), self.cv1(x)]
    for m in self.m:
        y.append(m(y[-1]))
    return self.cv3(torch.cat(y, 1))


def _fix_HGBlock(self, x):
    """
    Has to be fixed since torch.dynamo (torch 2.3.1) fails to understand the original expression
    Which is:
        y = [x]
        y.extend(m(y[-1]) for m in self.m)  <-- Problem, 'generator' statement
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y
    """
    y = [x]
    for m in self.m:
        y.append(m(y[-1]))
    y = self.ec(self.sc(torch.cat(y, 1)))
    return y + x if self.add else y


def _fix_Detect(self, x):
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    return x


def _fix_Segment(self, x):
    p = self.proto(x[0])  # mask protos
    bs = p.shape[0]  # batch size

    mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
    x = Detect.forward(self, x)
    return x, mc, p


def _fix_OBB(self, x):
    bs = x[0].shape[0]  # batch size
    angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
    angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
    x = Detect.forward(self, x)
    return x, angle

def _fix_Pose(self, x):
    bs = x[0].shape[0]  # batch size
    kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
    x = Detect.forward(self, x)
    return x, kpt


def _fix_WorldDetect(self, x, text):
    """Concatenates and returns predicted bounding boxes and class probabilities."""
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
    return x


def BaseModel_predict_once_forward(self, x, profile=False, visualize=False, embed=None):
    """
    Fix for the BaseModel predict once.
    This line:
        y.append(x if m.i in self.save else None)  # save output

    And this line:
        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

    Messes up torch.dynamo because Ultralytics repo at some point injects this attribute from random location.
    It PyTorch previous to 2.6 it could handle it, this is a regression on the torch.dynamo side.
    """
    y, dt, embeddings = [], [], []  # outputs
    inputs_dict: dict = {
        -1: x
    }
    for idx, m in enumerate(self.model):
        if isinstance(m.f, int):
            indices = [m.f]
        else:
            indices = list(m.f)
        inputs_to_next = [inputs_dict[iii] for iii in indices]
        if profile:
            self._profile_one_layer(m, x, dt)
        if len(inputs_to_next) == 1:
            x = m(inputs_to_next[0])  # run
        else:
            x = m(inputs_to_next)
        inputs_dict[m.i] = x
        inputs_dict[-1] = x
        if embed and m.i in embed:
            embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
            if m.i == max(embed):
                return torch.unbind(torch.cat(embeddings, 1), dim=0)
    return x


def install_fixes():
    SPPF.forward = _fix_SPPF
    C2f.forward = _fix_C2f
    C2fAttn.forward = _fix_C2fAttn
    RepNCSPELAN4.forward = _fix_RepNCSPELAN4
    SPPELAN.forward = _fix_SPPELAN
    C3f.forward = _fix_C3f
    HGBlock.forward = _fix_HGBlock
    Segment.forward = _fix_Segment
    OBB.forward = _fix_OBB
    Pose.forward = _fix_Pose
    BaseModel._predict_once = BaseModel_predict_once_forward

    WorldDetect.forward = _fix_WorldDetect
    Detect.forward = _fix_Detect
    v10Detect.forward = _fix_Detect


