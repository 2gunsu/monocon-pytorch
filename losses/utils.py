import torch
import functools
import torch.nn.functional as F

from typing import Optional, Tuple


def reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    assert reduction in ['none', 'mean', 'sum']
    reduction_enum = F._Reduction.get_enum(reduction)

    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss: torch.Tensor, 
                       weight: Optional[torch.Tensor] = None, 
                       reduction: str = 'mean', 
                       avg_factor: Optional[float] = None) -> torch.Tensor:

    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            loss = (loss.sum() / avg_factor)
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                reduction: str = 'mean',
                avg_factor: float = None,
                **kwargs) -> torch.Tensor:
        
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper



def expand_onehot_labels(labels: torch.Tensor, 
                         label_weights: Optional[torch.Tensor], 
                         label_channels: int) -> Tuple[torch.Tensor]:
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(
        (labels >= 0) & (labels < label_channels), as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)

    return bin_labels, bin_label_weights
