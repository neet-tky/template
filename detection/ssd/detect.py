import torch
from torch import nn

import torchvision
from torchvision.ops.boxes import nms

from _utils import decode
from data import voc as cfg

class Detect(nn.Module):
    def __init__(self, num_classes, bkg_label, top_k, nms_thresh, conf_thresh):
        super(Detect, self).__init__()

        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k

        self.nms = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')

        self.conf_thresh = conf_thresh
        self.variances = cfg['variance']

    def forward(self, preds_loc, preds_conf, anchors):
        num_batch = preds_loc.shape[0]
        num_anchor = anchors.shape[0]
        det_loc = torch.zeros(num_batch, self.num_classes, self.top_k, 4)
        det_conf = torch.zeros(num_batch, self.num_classes, self.top_k, 1)
        confidence = preds_conf.view(num_batch, num_anchor, self.num_classes).tranpose(2, 1)


        # decode predicions into bboxes
        for num in range(num_batch):
            decode_boxes = decode(preds_loc[num], anchors, self.variances)
            conf_scores = confidence[i].clone

            for cat in range(self.num_classes):
                c_mask = conf_scores[cat].gt(self.conf_thresh)
                scores = conf_scores[cat][cmask]

                if scores.shape[0] == 0:
                    continue

                l_mask = c_mask.unsqueeze(1).expand_as(decode_boxes)
                boxes = decode_boxes[l_mask].view(-1, 4)

                boxes, scores = self.filter_proposals(boxes, scores, img_shape, num_anchor)
                det_loc[num, cat] = boxes
                det_conf[num, cat] = scores

        return det_conf, det_conf

        def filter_proposals(self, proposals, scores, img_shape, num_anchors_per_levels):
            pass
    