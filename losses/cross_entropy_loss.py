import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from losses.utils import weight_reduce_loss, expand_onehot_labels


def binary_cross_entropy(pred: torch.Tensor,
                         label: torch.Tensor,
                         weight: Optional[torch.Tensor] = None,
                         reduction: str = 'mean',
                         avg_factor: Optional[int] = None,
                         class_weight: Optional[List[float]] = None) -> torch.Tensor:

    if pred.dim() != label.dim():
        label, weight = expand_onehot_labels(label, weight, pred.size(-1))

    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def cross_entropy(pred: torch.Tensor,
                  label: torch.Tensor,
                  weight: Optional[torch.Tensor] = None,
                  reduction: str = 'mean',
                  avg_factor: Optional[int] = None,
                  class_weight: Optional[List[float]] = None) -> torch.Tensor:

    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 use_sigmoid: bool = False,
                 reduction: str = 'mean',
                 class_weight: Optional[List[float]] = None,
                 loss_weight: float = 1.0):

        super().__init__()
        
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score: torch.Tensor,
                label: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> torch.Tensor:

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight, device=cls_score.device)
        else:
            class_weight = None
            
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls