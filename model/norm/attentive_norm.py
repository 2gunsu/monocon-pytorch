import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from torch.nn.modules.batchnorm import _BatchNorm


class HSigmoidv2(nn.Module):
    def __init__(self, inplace: bool = False):
        
        super().__init__()

        self.inplace = inplace

    def forward(self, x: torch.Tensor):
        out = F.relu6((x + 3.), inplace=self.inplace) / 6.
        return out


class AttnWeights(nn.Module):
    def __init__(self,
                 attn_mode: int,
                 num_features: int,
                 num_affine_trans: int,
                 num_groups: int = 1,
                 use_rsd: bool = True,
                 use_maxpool: bool = False,
                 use_bn: bool = True,
                 eps: float = 1e-3):
        
        super().__init__()

        if use_rsd:
            use_maxpool = False

        self.num_affine_trans = num_affine_trans
        self.use_rsd = use_rsd
        self.use_maxpool = use_maxpool
        self.eps = eps
        
        if not self.use_rsd:
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        layers = []
        if attn_mode == 0:
            layers = [
                nn.Conv2d(num_features, num_affine_trans, 1, bias=not use_bn),
                nn.BatchNorm2d(num_affine_trans) if use_bn else nn.Identity(),
                HSigmoidv2()]
        elif attn_mode == 1:
            if num_groups > 0:
                assert num_groups <= num_affine_trans
                layers = [
                    nn.Conv2d(num_features, num_affine_trans, 1, bias=False),
                    nn.GroupNorm(num_channels=num_affine_trans,
                                 num_groups=num_groups),
                    HSigmoidv2()]
            else:
                layers = [
                    nn.Conv2d(num_features, num_affine_trans, 1, bias=False),
                    nn.BatchNorm2d(num_affine_trans)
                    if use_bn else nn.Identity(),
                    HSigmoidv2()]

        self.attention = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                self._constant_init(m, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        b, c, h, w = x.size()
        
        if self.use_rsd:
            var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True)
            y = mean * (var + self.eps).rsqrt()
            
        else:
            y = self.avgpool(x)
            if self.use_maxpool:
                y = y + F.max_pool2d(x, (h, w), stride=(h, w)).view(b, c, 1, 1)
        return self.attention(y).view(b, self.num_affine_trans)
    
    def _constant_init(self, module: nn.Module, val: float, bias: float = 0.):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)


    def _kaiming_init(self,
                      module: nn.Module,
                      a: float = 0.,
                      mode: str = 'fan_out',
                      nonlinearity: str = 'relu',
                      bias: float = 0.,
                      dist: str = 'normal'):
        
        assert dist in ['uniform', 'normal']
        if hasattr(module, 'weight') and module.weight is not None:
            init_func = nn.init.kaiming_normal_ \
                if (dist != 'uniform') else nn.init.kaiming_uniform_
            init_func(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            
        if hasattr(module, 'bias') and (module.bias is not None):
            nn.init.constant_(module.bias, bias)


class AttnBatchNorm2d(nn.BatchNorm2d):
    def __init__(self,
                 num_features: int,
                 num_affine_trans: int,
                 attn_mode: int = 0,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 track_running_stats: bool = True,
                 use_rsd: bool = True,
                 use_maxpool: bool = False,
                 use_bn: bool = True,
                 eps_var: float = 1e-3):
        
        super().__init__(num_features, affine=False, eps=eps, momentum=momentum, track_running_stats=track_running_stats)

        self.num_affine_trans = num_affine_trans
        self.attn_mode = attn_mode
        self.use_rsd = use_rsd
        self.eps_var = eps_var

        self.weight_ = nn.Parameter(torch.Tensor(num_affine_trans, num_features))
        self.bias_ = nn.Parameter(torch.Tensor(num_affine_trans, num_features))

        self.attn_weights = AttnWeights(attn_mode,
                                        num_features,
                                        num_affine_trans,
                                        use_rsd=use_rsd,
                                        use_maxpool=use_maxpool,
                                        use_bn=use_bn,
                                        eps=eps_var)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.weight_, 1., 0.1)
        nn.init.normal_(self.bias_, 0., 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x)
        size = output.size()
        y = self.attn_weights(x)

        weight = (y @ self.weight_)
        bias = (y @ self.bias_)
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return (weight * output) + bias
