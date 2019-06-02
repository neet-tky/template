import torch
from torch import nn

import torchvision
from torchvision.ops.boxes import nms

from _utils import decode
from data import voc as cfg

class Detect(nn.Module):
    def __init__(self):
        super(Detect, self).__init__()

        pass

    def forward(self, output):
        pass

    