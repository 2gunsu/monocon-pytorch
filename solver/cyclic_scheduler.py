import math

from typing import Tuple, List
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CyclicScheduler(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 total_steps: int,
                 target_lr_ratio: Tuple[int, int] = (10, 1e-4),
                 target_momentum_ratio: Tuple[float, float] = (0.85 / 0.95, 1.),
                 period_up: float = 0.4):
        
        assert (optimizer.__class__.__name__ == 'AdamW'), \
            "Currently, this scheduler only supports 'AdamW' optimizer."
        
        self.total_steps = total_steps
        
        self.target_lr_ratio = target_lr_ratio
        self.target_momentum_ratio = target_momentum_ratio
        
        self.period_up = period_up
        self.steps_up = int(self.total_steps * self.period_up)
        
        for group in optimizer.param_groups:
            group.setdefault('initial_momentum', group['betas'][0])
        self.base_momentum = [
            group['initial_momentum']
            for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch=-1)
        
    
    def get_lr(self) -> List[float]:
        
        self.set_momentum()
        
        # Phase 1: LR Step-Up
        if (self._step_count < self.steps_up):
            return [self._annealing_func(base_lr * 1.0,
                                         base_lr * self.target_lr_ratio[0],
                                         (self._step_count - 0) / (self.steps_up - 0))
                    for base_lr in self.base_lrs]
        
        # Phase 2: LR Step-Down
        else:
            return [self._annealing_func(base_lr * self.target_lr_ratio[0],
                                         base_lr * self.target_lr_ratio[1],
                                         (self._step_count - self.steps_up) / (self.total_steps - self.steps_up))
                    for base_lr in self.base_lrs]
        

    def set_momentum(self):
        # Phase 1: Beta Step-Down
        if (self._step_count < self.steps_up):
            regular_momentums = [self._annealing_func(base_momentum * 1.0,
                                                      base_momentum * self.target_momentum_ratio[0],
                                                      (self._step_count - 0) / (self.steps_up - 0))
                                 for base_momentum in self.base_momentum]
        
        # Phase 2: Beta Step-Up
        else:
            regular_momentums = [self._annealing_func(base_momentum * self.target_momentum_ratio[0],
                                                      base_momentum * self.target_momentum_ratio[1],
                                                      (self._step_count - self.steps_up) / (self.total_steps - self.steps_up))
                                 for base_momentum in self.base_momentum]
        
        for param_group, mom in zip(self.optimizer.param_groups, regular_momentums):
            param_group['betas'] = (mom, param_group['betas'][1])
            

    def _annealing_func(self, start: float, end: float, factor: float, weight: float = 1.) -> float:
        cos_out = math.cos(math.pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out
