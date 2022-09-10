import os
import sys
import cv2
import glob
import numpy as np

from typing import Dict, Any
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from transforms import *
from utils.engine_utils import tprint


default_raw_transforms = [
    Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], keep_origin=True),
    Pad(size_divisor=32),
    ToTensor(),
    Convert_3D_to_4D(),
]


# Helper Class for KITTI Raw Visualization
class SimpleCalib:
    def __init__(self, calib_dict: Dict[str, Any]):
        self.P2 = calib_dict['P_rect_02']


class KITTIRawDataset(Dataset):
    def __init__(self, 
                 image_dir: str, 
                 calib_file: str,
                 img_extension: str = 'png'):
        
        super().__init__()
        
        assert os.path.isdir(image_dir), f"Argument 'image_dir' does not exist."
        assert os.path.isfile(calib_file), f"Argument 'calib_file' must be '.txt' file."
        
        img_extension = img_extension.replace('.', '')
        
        self.image_dir = image_dir
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, fr'*.{img_extension}')))
        self.calib = SimpleCalib(self._parse_calib(calib_file))
        
        self.transforms = Compose(default_raw_transforms)
        
        tprint(f"Found {len(self.image_files)} images in '{image_dir}'.")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img = cv2.imread(self.image_files[idx])
        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        
        metas = {
            'idx': idx,
            'image_path': self.image_files[idx],
            'ori_shape': img.shape}
        
        data_dict = {
            'img': img,
            'img_metas': metas,
            'calib': self.calib}
        return self.transforms(data_dict)
        
    def _parse_calib(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            calibs = f.readlines()
            
        calib_dict = {}
        for calib in calibs:
            key, value = calib.split(': ')
            value = value.replace('\n', '')
            
            if key[:2] in ['S_', 'R_', 'P_', 'T_']:
                value = np.array(value.split(' ')).astype(np.float32)
                if key[:2] == 'P_':
                    value = value.reshape(3, 4)
            
            calib_dict.update({key: value})
        return calib_dict
