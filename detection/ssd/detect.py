import torch
from torch import nn

import torchvision
from torchvision.ops import boxes as box_ops

from ._utils import decode
from dataset.config import voc as cfg

class Detect(nn.Module):
    def __init__(self, top_k, num_classes, conf_thresh, iou_thresh, img_size=320, min_size=0):
        super(Detect, self).__init__()
        
        self.top_k = top_k
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.variance = cfg['variance']
        self.img_size = (cfg['min_dim'] / img_size, cfg['min_dim'] / img_size)
        self.min_size = min_size
        
    def filter_proposals(self, loc_delta, anchors):
        decoded_boxes = decode(loc_delta, anchors, self.variance)
        decoded_boxes = box_ops.clip_boxes_to_image(decoded_boxes, self.img_size)
        keep = box_ops.remove_small_boxes(decoded_boxes, self.min_size)
        decoded_boxes = decoded_boxes[keep]
        
        return decoded_boxes
    
    def forward(self, conf_preds, loc_delta, anchors):
        num_batch = conf_p.shape[0]
        num_anchors = anchors.shape[0]

        preds = torch.zeros(num_batch, self.num_classes, self.top_k, 5)

        confidense = conf_p.view(num_batch, num_anchors, self.num_classes).transpose(2, 1)
        
        for num in range(num_batch):
            decoded_boxes = self.filter_proposals(loc_delta[num], anchors)
            conf_score = confidense[num].clone()
            
            for cat in range(self.num_classes):
                c_mask = conf_score[cat].gt(self.conf_thresh)
                scores = conf_score[cat][c_mask]
                
                if scores.shape[0] == 0:
                    continue
                
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                keep = box_ops.nms(boxes, scores, self.iou_thresh)[:self.top_k]

                preds[num, cat, :keep.shape[0]] = torch.cat([scores[keep].unsqueeze(1), boxes[keep]], dim=1)        
        
        return preds

if __name__=='__main__':
    conf_p = torch.FloatTensor(5, 15, 21).random_(0, 100) / 100
    loc_delta = torch.FloatTensor(5, 15, 4).random_(0, 100) / 200
    loc_delta[:, :, 2:] += loc_delta[:, :, :2]

    anchors = torch.FloatTensor(15, 4).random_(0, 100) / 100
    anchors[:, 2:] += anchors[:, :2]

    detect = Detect(5, 21, 0.5, 0.7)
    preds = detect(conf_p, loc_delta, anchors)
    print(preds)