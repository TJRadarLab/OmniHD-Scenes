# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# Written by [TONGJI] [Lianqing Zheng]
# All rights reserved. Unauthorized distribution prohibited.
# Feel free to reach out for collaboration opportunities.
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
import torch
from torch import nn
from torchvision.models.resnet import resnet18
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from torchvision.utils import save_image
from mmdet3d.models.fusion_layers import apply_3d_transformation
import torch.nn.functional as F

import torch
import torch.nn as nn

class BevEncode(nn.Module):
    def __init__(self, c_in, c_out):
        super(BevEncode, self).__init__()
        self.bevencode = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.bevencode(x[0])
        return x

