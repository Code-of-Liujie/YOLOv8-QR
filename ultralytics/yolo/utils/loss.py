# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import bbox_iou
from .tal import bbox2dist


def Wasserstein(box1, box2, xywh=True):
    box2 = box2.T
    if xywh:
        b1_cx, b1_cy = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
        b1_w, b1_h = box1[2] - box1[0], box1[3] - box1[1]
        b2_cx, b2_cy = (box2[0] + box2[0]) / 2, (box2[1] + box2[3]) / 2
        b1_w, b1_h = box2[2] - box2[0], box2[3] - box2[1]
    else:
        b1_cx, b1_cy, b1_w, b1_h = box1[0], box1[1], box1[2], box1[3]
        b2_cx, b2_cy, b2_w, b2_h = box2[0], box2[1], box2[2], box2[3]
    cx_L2Norm = torch.pow((b1_cx - b2_cx), 2)
    cy_L2Norm = torch.pow((b1_cy - b2_cy), 2)
    p1 = cx_L2Norm + cy_L2Norm
    w_FroNorm = torch.pow((b1_w - b2_w)/2, 2)
    h_FroNorm = torch.pow((b1_h - b2_h)/2, 2)
    p2 = w_FroNorm + h_FroNorm
    return p1 + p2


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367."""

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).sum()
        return loss


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum



        # iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # nwd = torch.exp(
        #     -torch.pow(Wasserstein(pred_bboxes[fg_mask].T, target_bboxes[fg_mask], xywh=False), 1 / 2) / 1.0)

        # loss_iou1 = ((1.0 - iou).mean()) * 0 + ((1.0 - nwd).mean()) * 1

        #loss_iou = loss_iou + loss_iou1


        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class KeypointLoss(nn.Module):

    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / (torch.sum(kpt_mask != 0) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return kpt_loss_factor * ((1 - torch.exp(-e)) * kpt_mask).mean()
