import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn

from typing import Tuple, List


class Conv2dBlock(nn.Module):
    def __init__(self, 
                 in_planes: int, 
                 out_planes: int, 
                 kernel_size: int = 3, 
                 stride: int = 1, 
                 bias: bool = True):
        
        super().__init__()
        
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(kernel_size // 2),
                              bias=bias)
        self.add_module('bn1', nn.BatchNorm2d(out_planes))
        self.relu = nn.ReLU(inplace=False)

    @property
    def norm1(self):
        return getattr(self, 'bn1')

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.norm1(x)
        x = self.relu(x)
        return x


class IDAUp(nn.Module):
    def __init__(self, 
                 in_channels_list: Tuple[int], 
                 up_factors_list: Tuple[int], 
                 out_channels: int):
        
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        for i in range(1, len(in_channels_list)):
            in_channels = in_channels_list[i]
            up_factors = int(up_factors_list[i])

            proj = Conv2dBlock(in_channels, out_channels, kernel_size=3, stride=1, bias=False)
            node = Conv2dBlock((out_channels * 2), out_channels, kernel_size=3, stride=1, bias=False)
            up = nn.ConvTranspose2d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=(up_factors * 2),
                                    stride=up_factors,
                                    padding=(up_factors // 2),
                                    output_padding=0,
                                    groups=out_channels,
                                    bias=False)
            self.fill_upconv_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
            
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def fill_upconv_weights(self, upconv: nn.Module) -> None:
        w = upconv.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

    def forward(self, layers: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        assert len(self.in_channels_list) == len(layers), \
            '{} vs {} layers'.format(len(self.in_channels_list), len(layers))

        for i in range(1, len(layers)):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            node = getattr(self, 'node_' + str(i))

            layers[i] = upsample(project(layers[i]))
            layers[i] = node(torch.cat([layers[i-1], layers[i]], 1))

        return layers


class DLAUp(nn.Module):
    def __init__(self, 
                 in_channels_list: List[int] = (64, 128, 256, 512), 
                 scales_list: Tuple[int] = (1, 2, 4, 8),
                 start_level: int = 2):
        
        super().__init__()
        
        scales_list = np.array(scales_list, dtype=int)
        self.in_channels_list = in_channels_list
        self.start_level = start_level

        for i in range(len(in_channels_list) - 1):
            j = (- i - 2)
            setattr(self, 'ida_{}'.format(i), IDAUp(in_channels_list=in_channels_list[j:],
                                                    up_factors_list=(scales_list[j:] // scales_list[j]),
                                                    out_channels=in_channels_list[j]))
            
            scales_list[j + 1:] = scales_list[j]
            in_channels_list[j + 1:] = [in_channels_list[j] for _ in in_channels_list[j + 1:]]
            
        self.init_weights()

    def init_weights(self):
        for i in range(len(self.in_channels_list) - 1):
            getattr(self, 'ida_{}'.format(i)).init_weights

    def forward(self, layers: Tuple[torch.Tensor]) -> torch.Tensor:
        layers = layers[self.start_level:]
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            layers[-i - 2:] = ida(layers[-i - 2:])
        return [layers[-1]]
