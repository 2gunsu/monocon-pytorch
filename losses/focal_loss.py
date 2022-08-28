import torch
import torch.nn as nn


class GaussianFocalLoss(nn.Module):
    def __init__(self, 
                 loss_weight: float = 1.0, 
                 gamma: float = 2.0, 
                 beta: float = 4.0, 
                 alpha: float = -1.0):
        
        super().__init__()
        
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        
        self.eps = 1e-12

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        num_pos = pos_inds.sum()

        neg_weights = torch.pow(1 - target, self.beta)

        loss = 0

        pos_loss = torch.log(input + self.eps) * torch.pow((1 - input), self.gamma) * pos_inds
        neg_loss = torch.log((1 - input) + self.eps) * torch.pow(input, self.gamma) * neg_weights * neg_inds

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if self.alpha >= 0:
            pos_loss = self.alpha * pos_loss
            neg_loss = (1 - self.alpha) * neg_loss

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss.mean() * self.loss_weight