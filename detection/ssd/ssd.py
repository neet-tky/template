from math import sqrt
from itertools import product

import torch
import torch.nn.functional as F
from torch import nn

from dataset.config import *
from .l2norm import *

class AnchorGenerator(nn.Module):
    def __init__(self, cfg):
        super(AnchorGenerator, self).__init__()

        self.image_size = cfg['min_dim']

        self.num_anchor = len(cfg["aspect_ratio"])
        self.variance = cfg["variance"] or [0.1]
        self.feature_maps = cfg["feature_maps"]
        self.min_sizes = cfg["min_sizes"]
        self.max_sizes = cfg["max_sizes"]

        self.steps = cfg["steps"]
        self.aspect_ratio = cfg["aspect_ratio"]
        self.version = cfg["name"]

        for v in self.variance:
            if v <= 0:
                raise ValueError("Variances must be greater than 0")

    def forward(self):
        mean = []

        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]

                # unit center x, y
                cx = (j + .5) / f_k
                cy = (i + .5) / f_k

                # aspect_ratio: 1
                # rei size: sqrt(s_k * s_(k + 1))
                s_k = self.min_sizes[k] / self.image_size
                mean += [ cx, cy, s_k, s_k ]

                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                for ar in self.aspect_ratio[k]:
                    mean += [ cx, cy, s_k * sqrt(ar), s_k / sqrt(ar) ]
                    mean += [ cx, cy, s_k / sqrt(ar), s_k * sqrt(ar) ]


        output = torch.Tensor(mean).view(-1, 4)
        return output

def permute_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)

    return layer

def concat_box_prediction_layers(conf_box, loc_box):
    box_cls_flattened = []
    box_loc_flattened = []

    for box_cls_per_lebel, box_loc_per_lebel in zip(conf_box, loc_box):
        N, AxC, H, W = box_cls_per_lebel.shape
        Ax4 = box_loc_per_lebel.shape[1]

        A = Ax4 // 4
        C = AxC // A

        #if fpn
        box_cls_per_lebel = permute_flatten(box_cls_per_lebel, N, A, C, H, W)
        box_loc_per_lebel = permute_flatten(box_loc_per_lebel, N, A, 4, H ,W)
#        print(box_cls_per_lebel.shape, box_loc_per_lebel.shape)
        box_cls_flattened.append(box_cls_per_lebel)
        box_loc_flattened.append(box_loc_per_lebel)

    box_cls = torch.cat(box_cls_flattened, dim=1)
    box_loc = torch.cat(box_loc_flattened, dim=1)

    return box_cls, box_loc

class SSD(nn.Module):
    def __init__(self, vgg, extras, head, phase="train", class_num=21):
        super(SSD, self).__init__()

        # model
        self.backbone = nn.ModuleList(vgg)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc_delta = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.f4_2 = 23

        #
        self.phase = phase
        self.anchor_generator = AnchorGenerator(voc)
        self.anchor = self.anchor_generator.forward()

        if self.phase == "test":
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(class_num, 0, 200, .01, .45)
            # num_anchor = 200, pos_neg = .01, nms = .45

    def forward(self, img):
        # variables
        x = img
        features = []
        loc_delta, conf = [], []

        # feedforward
        for i in range(self.f4_2):
            x = self.backbone[i](x)
        x = self.L2Norm(x)
        features.append(x)

        for i in range(self.f4_2, len(self.backbone)):
            x = self.backbone[i](x)
        features.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x))

            if k % 2 == 1:
                features.append(x)

        # gene feature MAPs
        for f, l, c in zip(features, self.loc_delta, self.conf):
#            print(f.shape)
            loc_delta.append((l(f)))
            conf.append(c(f))

        # premute and flatten
        conf, loc_delta = concat_box_prediction_layers(conf, loc_delta)

        if self.phase == "test":
            output = self.detect(loc_delta, self.softmax(conf), self.anchor)

        else:
            output = (loc_delta, conf, self.anchor)

        return output

def vgg16(base, i, batch_norm=False):
    layers = []
    in_channels = i

    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]

        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]

            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    layers += [ pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True) ]
    return layers

def add_extras(base, i, batch_norm=False):
    layers = []
    in_channels = i
    flag = False

    for k, v in enumerate(base):
        if in_channels != 'S':
            if v == 'S':
                layers += [ nn.Conv2d(in_channels, base[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1) ]

            else:
                layers += [ nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag] ) ]

            flag = not flag
        in_channels = v

    return layers

def multibox(vgg, extra_layers, cfg, num_class):
    loc, conf = [], []
    backbone_multi = [21, -2]

    for k, v in enumerate(backbone_multi):
        loc += [ nn.Conv2d(in_channels=vgg[v].out_channels, out_channels=cfg[k] * 4, kernel_size=3, padding=1) ]
        conf += [ nn.Conv2d(in_channels=vgg[v].out_channels, out_channels=cfg[k] * num_class, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers[1::2], 2):
        loc += [ nn.Conv2d(in_channels=v.out_channels, out_channels=cfg[k] * 4, kernel_size=3, padding=1) ]
        conf += [ nn.Conv2d(in_channels=v.out_channels, out_channels=cfg[k] * num_class, kernel_size=3, padding=1) ]

    return vgg, extra_layers, (loc, conf)

base = {
    "320": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    "512": [],
}

extras = {
    "320": [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    "512": [],
}

mbox = {
    "320": [4, 6, 6, 6, 4, 4],
    "512": [],
}

def build_ssd(phase="train", size=320, class_num = 21):
    _vgg, _extras, _head = multibox(vgg16(base[str(size)], 3), add_extras(extras[str(size)], 1024), mbox[str(size)], class_num)

    return SSD(_vgg, _extras, _head, phase, class_num)

if __name__=='__main__':
    img = torch.Tensor(10, 3, 320, 320).random_(0, 1)

    net = build_ssd()
    loc, conf, anchor = net(img)

    print(loc.shape)
    print(conf.shape)
    print(anchor.shape)
