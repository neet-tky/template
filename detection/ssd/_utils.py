import math
import torch

class Matcher(object):

    BELOW_LOW_TRESHOLD = -1
    BETWEEN_THRESHOLD = -2
    def __init__(self, high_threshold=.7, low_threshold=.3, allow_low_quality_match=False):
        super(Matcher, self).__init__()
        assert high_threshold >= low_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_match = allow_low_quality_match

    def __call__(self, match_quality_matrix):
        if match_quality_matrix.numel() == 0:
            if match_quality_matrix.shape[0] == 0:
                raise ValueError("No gt-boxes available for one of the images during training")
            else:
                raise  ValueError("No proposal-boxes available for one of the images during training")

        matched_values, matches = match_quality_matrix.max(dim=0)

        if self.allow_low_quality_match:
            all_match = matches.clone()

        low_thresh_ind = matched_values < self.low_threshold
        matches[low_thresh_ind] = self.BELOW_LOW_TRESHOLD

        between_ind = (matched_values >= self.low_threshold) & (matched_values < self.high_threshold)
        matches[between_ind] = self.BETWEEN_THRESHOLD

        if self.allow_low_quality_match:
            self.see_low_quality_matches_(matches, all_match, match_quality_matrix)

        return matches

    def see_low_quality_matches_(self, matches, all_match, match_quality_matrix):
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)

        gt_pred_pairs_of_highest_quality = torch.nonzero(match_quality_matrix == highest_quality_foreach_gt[:, None])
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]

        matches[pred_inds_to_update] = all_match[pred_inds_to_update]

def encode(matched, anchors, variances):
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - anchors[:, :2]
    g_cxcy /= (variances[0] * anchors[:, 2:])

    g_wh = (matched[:, 2:] - matched[:, :2]) / anchors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]

    return torch.cat([g_cxcy, g_wh], 1)

def decode(matched, anchors, variances):
    pass