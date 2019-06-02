import torch
from torch import nn

import torchvision
from torchvision.ops.boxes import nms

from _utils import decode
from data import voc as cfg

class Detect(nn.Module):
    def __init__(self):
        super(Detect, self).__init__()

        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k

        self.nms = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')

        self.conf_thresh = conf_thresh
        self.variances = cfg['variance']

    def forward(self, output):
        pass

    