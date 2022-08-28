import os
import sys
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from typing import Tuple, List
from torch.nn.modules.batchnorm import _BatchNorm


class BasicBlock(nn.Module):
    def __init__(self, 
                 inplanes: int, 
                 planes: int, 
                 stride: int = 1, 
                 dilation: int = 1):
        
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.stride = stride

    def forward(self, 
                x: torch.Tensor, 
                residual: torch.Tensor = None) -> torch.Tensor:
        
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    
    expansion = 2

    def __init__(self, 
                 inplanes: int, 
                 planes: int, 
                 stride: int = 1, 
                 dilation: int = 1):
        
        super().__init__()
        
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion

        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        
        self.stride = stride

    def forward(self, 
                x: torch.Tensor, 
                residual: torch.Tensor = None) -> torch.Tensor:
        
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 residual: bool = False):
        
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1)) 
        x = self.bn(x)
        if self.residual:
            x = x + children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, 
                 levels: int, 
                 block: nn.Module, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1,
                 level_root: bool = False, 
                 root_dim: int = 0, 
                 root_kernel_size: int = 1,
                 dilation: int = 1, 
                 root_residual: bool = False):
        
        super().__init__()
        
        if root_dim == 0:
            root_dim = 2 * out_channels
            
        if level_root:
            root_dim = root_dim + in_channels
            
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
            
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels

        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
            
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels))
            
    def forward(self, 
                x: torch.Tensor, 
                residual=None, 
                children=None):
        
        children = [] if (children is None) else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    
    arch_settings = {
        34: (BasicBlock, (1, 1, 1, 2, 2, 1), (16, 32, 64, 128, 256, 512), False),
        46: (Bottleneck, (1, 1, 1, 2, 2, 1), (16, 32, 64, 64, 128, 256), False),
        60: (Bottleneck, (1, 1, 1, 2, 3, 1), (16, 32, 128, 256, 512, 1024), False),
        102: (Bottleneck, (1, 1, 1, 3, 4, 1), (16, 32, 128, 256, 512, 1024), True)}

    def __init__(self, 
                 num_layers: int, 
                 in_channels: int = 3,
                 pretrained: bool = True):
        
        super().__init__()
        
        assert num_layers in self.arch_settings.keys(), \
            f"Argument 'num_layers' must be one in {list(self.arch_settings.keys())}"
        block, levels, channels, residual_root = self.arch_settings[num_layers]
        
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.channels = channels
        
        self.base_layer = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=False))
        
        self.level0 = self._make_multilevel_conv(channels[0], channels[0], num_levels=levels[0])
        self.level1 = self._make_multilevel_conv(channels[0], channels[1], num_levels=levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root)
        
        if pretrained:
            self.load_imagenet_weights(num_layers=self.num_layers)
        else:
            self.init_weights()
            
    def load_imagenet_weights(self, num_layers: int):
        
        NUM_LAYERS_TO_HASH = {
            34: ('dla34', 'ba72cf86'),
            46: ('dla46_c', '2bfd52c3'),
            60: ('dla60', '24839fc4'),
            102: ('dla102', 'd94d9790')}
        
        arch_name, hash = NUM_LAYERS_TO_HASH[num_layers]

        base_url = 'http://dl.yf.io/dla/models/imagenet'
        url = os.path.join(base_url, f'{arch_name}-{hash}.pth')
        
        state_dict = model_zoo.load_url(url)
        self.load_state_dict(state_dict, strict=False)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x, y = self.base_layer(x), []
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return tuple(y)
                    
    def _make_multilevel_conv(self, 
                              inplanes: int, 
                              planes: int, 
                              num_levels: int, 
                              stride: int = 1,
                              dilation: int = 1):
        
        modules = []
        for i in range(num_levels):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if (i == 0) else 1,
                          padding=dilation, 
                          bias=False, 
                          dilation=dilation),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=False)])
            inplanes = planes
        return nn.Sequential(*modules)
    
    def get_out_channels(self, start_level: int) -> List[int]:
        return list(self.channels[start_level:])
