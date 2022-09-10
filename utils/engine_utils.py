import os
import sys
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from yacs.config import CfgNode
from datetime import datetime
from typing import Dict, Any
from contextlib import redirect_stdout

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.monocon_configs import _C as cfg


def generate_random_seed(seed: int = None) -> int:
    if (seed is not None) and (seed != -1):
        return seed
    
    seed = np.random.randint(2 ** 31)
    return seed


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_default_cfg() -> CfgNode:
    return cfg.clone()


def load_cfg(cfg_file: str) -> CfgNode:
    cfg_ = get_default_cfg()
    cfg_.set_new_allowed(True)
    cfg_.merge_from_file(cfg_file)
    return cfg_


def export_cfg(cfg: CfgNode, save_path: str) -> None:
    with open(save_path, 'w') as f:
        with redirect_stdout(f):
            print(cfg.dump())


def export_dict_to_csv(data: Dict[str, Any], save_path: str):
    df = pd.DataFrame.from_dict(data)
    df.to_csv(save_path, index=False)
    

def move_data_device(data_dict: Dict[str, Any], 
                     device: str = None) -> Dict[str, Any]:
    
    if (device is None) or not torch.cuda.is_available():
        device = 'cpu'
    
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = data_dict[k].to(device)

    if 'label' in data_dict.keys():
        label = data_dict['label']
        for k in label.keys():
            label[k] = label[k].to(device)
        data_dict['label'] = label
    
    return data_dict


def reduce_loss_dict(loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    return sum([v for v in loss_dict.values()])


def tprint(message: str, indent: bool = False) -> None:
    cur_time = str(datetime.now())[:-7]
    message = f'[{cur_time}] {message}'
    if indent:
        message = '\n' + message
    print(message)


def progress_to_string_bar(current_prog: int,
                           total_prog: int,
                           bins: int = 10, 
                           non_filled_chr: str = ' ',
                           filled_chr: str = '#') -> str:
    
    prog_perc = current_prog / total_prog
    assert (0.0 <= prog_perc <= 1.0)
    
    prog_str = [non_filled_chr,] * bins
    num_filled = int(prog_perc / (1 / bins))
    
    for idx in range(num_filled):
        prog_str[idx] = filled_chr
    
    prog_str = ''.join(prog_str)
    prog_str = f'[{prog_str}][{prog_perc * 100:5.2f}%]'
    return prog_str