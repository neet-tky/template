import torch
from torch import nn

from torchvision.ops import boxes as box_ops

import _utils as det_utils

class MultiLoss(nn.Module):
    def __init__(self, num_classes=21, high_threshold=.9, low_threshold=.3, variances=[.1, .2], negpos_ratio=3, device='cpu'):
        super(MultiLoss, self).__init__()
        self.num_classes = num_classes
        self.negapos_ratio = negpos_ratio
        self.device = device
        self.variances = variances

        self.box_similarity = box_ops.box_iou
        self.proposal_matcher = det_utils.Matcher(high_threshold, low_threshold, True)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.l1 = nn.SmoothL1Loss()

    def assign_targets_to_anchors(self, anchors, targets):
        gt_labels = targets['labels']
        gt_boxes = targets['boxes']


        ## ここあとで消す
        gt_boxes[:, :, 2:] += gt_boxes[:, :, :2]
        anchors[:, 2:] += anchors[:, :2]

        batch_num = gt_boxes.shape[0]
        anchors_num = anchors.shape[0]

        matched_gt_boxes = torch.FloatTensor(batch_num, anchors_num, 4)
        matched_gt_labels = torch.LongTensor(batch_num, anchors_num)

        for num in range(batch_num):
            truths = gt_boxes[num][:, :]
            labels = gt_labels[num]

            match_quality_matrix = self.box_similarity(truths, anchors)
            matched_ind = self.proposal_matcher(match_quality_matrix)

            # match_gt_box is gene
            matched_gt_boxes_per_batch = truths[matched_ind.clamp(min=0)]
            matched_gt_boxes_per_batch = det_utils.encode(matched_gt_boxes_per_batch, anchors, self.variances)

            # match_gt_labels is gene
            bg_ind = matched_ind < 0
            matched_labels_per_batch = labels[matched_ind.clamp(min=0)]
            matched_labels_per_batch[bg_ind] = 0

            matched_gt_boxes[num] = matched_gt_boxes_per_batch
            matched_gt_labels[num] = matched_labels_per_batch

        return matched_gt_labels, matched_gt_boxes

    def compute_loss(self, preds_loc, preds_conf, match_gt_box, match_gt_label):
        pos_anchors = match_gt_label > 0  # [B, A]
        num_batch = preds_loc.shape[0]  # B
        num_anchor = preds_loc.shape[1]

        # compute loc: [BxA, 4] vs [BxA, 4]
        loc_pos_idx = pos_anchors.unsqueeze(pos_anchors.dim()).expand_as(preds_loc)  # [B, A, 4]
        loss_loc = self.l1(preds_loc[loc_pos_idx].view(-1, 4), match_gt_box[loc_pos_idx].view(-1, 4))

        # compute conf: [BxA] <- [BxA, C] vs [BxA]
        loss_conf_all = self.cross_entropy(preds_conf.view(-1, self.num_classes), match_gt_label.view(-1))
        loss_conf_all = loss_conf_all.view(num_batch, -1)

        # Hard Negative Mining
        num_pos = pos_anchors.long().sum(dim=1, keepdim=True)
        num_hard_nega = self.negapos_ratio * num_pos

        loss_conf_pos = loss_conf_all[pos_anchors].sum()
        loss_conf_neg = loss_conf_all.clone()
        loss_conf_neg[pos_anchors] = 0
        loss_conf_neg, _ = loss_conf_neg.sort(1, descending=True)

        hardness_rank = torch.LongTensor(range(num_anchor)).unsqueeze(0).expand_as(loss_conf_neg).to(self.device)
        hard_nega = hardness_rank < num_hard_nega
        loss_conf_hard_neg = loss_conf_neg[hard_nega].sum()

        # total loss
        N = num_pos.sum()
        loss_conf = (loss_conf_hard_neg + loss_conf_pos) / N

        return loss_loc, loss_conf

    def hard_negative_mining(self, loss):
        return loss

    def forward(self, preds, targets):
        preds_loc_delta, preds_conf, anchors = preds

        # gene target_gt_boxes
        matched_gt_labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
        loss_loc, loss_conf = self.compute_loss(preds_loc_delta, preds_conf, matched_gt_boxes, matched_gt_labels)

        return loss_loc, loss_conf

if __name__=='__main__':
    preds = [
        torch.FloatTensor(10, 20, 4).random_(0, 200) / 200,
        torch.FloatTensor(10, 20, 21).random_(0, 100) / 100,
        torch.FloatTensor(20, 4).random_(0, 200) / 200
    ]

    targets = {
       'boxes': torch.FloatTensor(10, 12, 4).random_(0, 200) / 200,
       'labels':  torch.LongTensor(10, 12).random_(1, 21),
    }

    loss = MultiLoss().forward(preds, targets)
