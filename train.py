import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from engine.monocon_engine import MonoconEngine
from utils.engine_utils import tprint, get_default_cfg, set_random_seed, generate_random_seed



# Some Torch Settings
torch_version = int(torch.__version__.split('.')[1])
if torch_version >= 7:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# Get Config from 'config/monocon_configs.py'
cfg = get_default_cfg()


# Set Benchmark
# If this is set to True, it may consume more memory. (Default: True)
if cfg.get('USE_BENCHMARK', True):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    tprint(f"CuDNN Benchmark is enabled.")


# Set Random Seed
seed = cfg.get('SEED', -1)
seed = generate_random_seed(seed)
set_random_seed(seed)

cfg.SEED = seed
tprint(f"Using Random Seed {seed}")


# Initialize Engine
engine = MonoconEngine(cfg)


# Start Training from Scratch
# Output files will be saved to 'cfg.OUTPUT_DIR'.
engine.train()