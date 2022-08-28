import torch
import torch.nn as nn


class LaplacianAleatoricUncertaintyLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self,
                input: torch.Tensor,
                target: torch.Tensor,
                log_variance: torch.Tensor) -> torch.Tensor:

        log_variance = log_variance.flatten()
        input = input.flatten()
        target = target.flatten()

        loss = 1.4142 * torch.exp(-log_variance) * torch.abs(input - target) + log_variance

        return loss.mean() * self.loss_weight