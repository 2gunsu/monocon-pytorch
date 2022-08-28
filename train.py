import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from engine import MonoconEngine
from utils.engine_utils import get_default_cfg


# Some Torch Settings
torch_version = int(torch.__version__.split('.')[1])
if torch_version >= 7:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# Get Config from 'config/monocon_configs.py'
cfg = get_default_cfg()

# Initialize Engine
engine = MonoconEngine(cfg)

# Start Training from Scratch
# Output files will be saved to 'cfg.OUTPUT_DIR'.
engine.train()