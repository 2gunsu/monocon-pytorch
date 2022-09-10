import os
import sys
import torch
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from engine.monocon_engine import MonoconEngine
from utils.engine_utils import tprint, load_cfg, generate_random_seed, set_random_seed


# Arguments
parser = argparse.ArgumentParser('MonoCon Tester for KITTI 3D Object Detection Dataset')
parser.add_argument('--config_file',
                    type=str,
                    help="Path of the config file (.yaml)")
parser.add_argument('--checkpoint_file', 
                    type=str,
                    help="Path of the checkpoint file (.pth)")
parser.add_argument('--gpu_id', type=int, default=0, help="Index of GPU to use for testing")
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--save_dir', 
                    type=str,
                    help="Path of the directory to save the visualized results")

args = parser.parse_args()


# Some Torch Settings
torch_version = int(torch.__version__.split('.')[1])
if torch_version >= 7:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# Load Config
cfg = load_cfg(args.config_file)
cfg.GPU_ID = args.gpu_id


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

tprint(f"Using Random Seed {seed}")


# Initialize Engine
engine = MonoconEngine(cfg, auto_resume=False, is_test=True)
engine.load_checkpoint(args.checkpoint_file, verbose=True)


# Evaluate
if args.evaluate:
    tprint("Mode: Evaluation")
    engine.evaluate()


# Visualize
if args.visualize:
    tprint("Mode: Visualization")
    engine.visualize(args.save_dir, draw_items=['2d', '3d', 'bev'])