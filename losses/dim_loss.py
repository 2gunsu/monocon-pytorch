import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class DimAwareL1Loss(nn.Module):
    def __init__(self, loss_weight: Optional[float] = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self,
                input: torch.Tensor,
                target: torch.Tensor,
                dimension: torch.Tensor) -> torch.Tensor:

        dimension = dimension.clone().detach()

        loss = torch.abs(input - target) / dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(input, target) / loss.mean()
        loss = (loss * compensation_weight)

        return loss.mean() * self.loss_weight