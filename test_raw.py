import os
import sys
import torch
import argparse

from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.visualizer import Visualizer
from model.detector import MonoConDetector
from utils.engine_utils import tprint, move_data_device
from dataset.kitti_raw_dataset import KITTIRawDataset


# Arguments
parser = argparse.ArgumentParser('MonoCon Tester for KITTI Raw Dataset')
parser.add_argument('--data_dir',
                    type=str,
                    help="Path where sequence images are saved")
parser.add_argument('--calib_file',
                    type=str,
                    help="Path to calibration file (.txt)")
parser.add_argument('--checkpoint_file', 
                    type=str,
                    help="Path of the checkpoint file (.pth)")
parser.add_argument('--gpu_id', type=int, default=0, help="Index of GPU to use for testing")
parser.add_argument('--fps', type=int, default=25, help="FPS of the result video")
parser.add_argument('--save_dir', 
                    type=str,
                    help="Path of the directory to save the inferenced video")

args = parser.parse_args()



# Main

# (1) Build Dataset
dataset = KITTIRawDataset(args.data_dir, args.calib_file)


# (2) Build Model
device = f'cuda:{args.gpu_id}'

detector = MonoConDetector()
detector.load_checkpoint(args.checkpoint_file)
detector.to(device)
detector.eval()

tprint(f"Checkpoint '{args.checkpoint_file}' is loaded to model.")


# (3) Inference
vis_results = []

with torch.no_grad():
    for data in tqdm(dataset, desc="Collecting Results..."):
        data = move_data_device(data, device)
        vis_result = detector.batch_eval(data, get_vis_format=True)
        vis_results.extend(vis_result)
    

# (4) Visualize
visualizer = Visualizer(dataset, vis_format=vis_results)
visualizer.export_as_video(args.save_dir, plot_items=['2d', '3d', 'bev'], fps=args.fps)