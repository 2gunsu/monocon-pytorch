import os
import sys
import cv2
import json
import numpy as np

from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from engine.kitti_eval import kitti_eval
from utils.data_classes import KITTICalibration, KITTIMultiObjects

# Fixed Root
IMAGESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ImageSets')


class BaseKITTIMono3DDataset(Dataset):
    def __init__(self, 
                 base_root: str, 
                 split: str,
                 pad_divisor: int = 32,
                 preload_gt_info: bool = False):
        
        super().__init__()
        
        assert os.path.isdir(base_root)
        self.base_root = base_root
        
        assert split in ['train', 'val', 'trainval', 'test']
        self.split = split
        
        with open(os.path.join(IMAGESET_DIR, f'{split}.txt')) as f:
            file_prefix = f.readlines()
        self.file_prefix = [fp.replace('\n', '') for fp in file_prefix]
        
        sub_root = 'testing' if (split == 'test') else 'training'
        
        # Image Files
        self.image_dir = os.path.join(base_root, sub_root, 'image_2')
        self.image_files = [os.path.join(self.image_dir, f'{fp}.png') for fp in self.file_prefix]
        
        # Calibration Files
        self.calib_dir = os.path.join(base_root, sub_root, 'calib')
        self.calib_files = [os.path.join(self.calib_dir, f'{fp}.txt') for fp in self.file_prefix]
        
        # Label Files
        self.label_dir, self.label_files = None, []
        if (split != 'test'):
            self.label_dir = os.path.join(base_root, sub_root, 'label_2')
            self.label_files = [os.path.join(self.label_dir, f'{fp}.txt') for fp in self.file_prefix]
        
        self.pad_divisor = pad_divisor
        
        self.gt_annos = None
        if preload_gt_info:
            gt_infos = self.collect_gt_infos()
            self.gt_annos = [gt_info['annos'] for gt_info in gt_infos]
            
    def __len__(self):
        return len(self.file_prefix)
    
    def __getitem__(self, idx: int):
        raise NotImplementedError
    
    def load_image(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        image_arr = cv2.imread(self.image_files[idx])
        image_data = cv2.cvtColor(image_arr, code=cv2.COLOR_BGR2RGB)
        
        img_metas = {
            'idx': idx,
            'split': self.split,
            'sample_idx': int(os.path.basename(self.image_files[idx]).split('.')[0]),
            'image_path': self.image_files[idx],
            'ori_shape': image_data.shape[:2]}
        return (image_data, img_metas)
    
    def load_calib(self, idx: int) -> KITTICalibration:
        return KITTICalibration(self.calib_files[idx])
    
    def load_label(self, idx: int) -> KITTIMultiObjects:
        calib = self.load_calib(idx)
        return KITTIMultiObjects.get_objects_from_label(self.label_files[idx], calib)
    
    def collect_gt_infos(self, verbose: bool = False) -> List[Dict[str, Any]]:
        
        # Entire objects which include 'DontCare' class are required for evaluation.
        # If 'ignored_flag' is True, Filtered objects are converted to the original objects.
        ignored_flag = False
        if self.load_label(0).ignore_dontcare:
            ignored_flag = True
            
        results = []
        num_samples = len(self)
        
        iter_ = range(num_samples)
        if verbose:
            iter_ = tqdm(iter_, desc="Collecting GT Infos...")
        
        for idx in iter_:
            
            _, img_metas = self.load_image(idx)
            
            calib = self.load_calib(idx)
            calib_dict = calib.get_info_dict()
            
            obj_cls = self.load_label(idx)
            if ignored_flag:
                obj_cls = obj_cls.original_objects
            obj_dict = obj_cls.info_dict
            
            results.append(
                {'image': img_metas,
                 'calib': calib_dict,
                 'annos': obj_dict})
        return results  
    
    def evaluate(self, 
                 kitti_format_results: Dict[str, Any],
                 eval_classes: List[str] = ['Pedestrian', 'Cyclist', 'Car'],
                 eval_types: List[str] = ['bbox', 'bev', '3d'],
                 verbose: bool = True,
                 save_path: str = None) -> Dict[str, float]:
        
        if self.gt_annos is None:
            gt_infos = self.collect_gt_infos(verbose=verbose)
            gt_annos = [info['annos'] for info in gt_infos]
            
            self.gt_annos = gt_annos

        ap_dict = dict()
        
        for name, result in kitti_format_results.items():
            if '2d' in name:
                eval_types=['bbox']
            result_string, result_dict = kitti_eval(
                gt_annos=self.gt_annos,
                dt_annos=result,
                current_classes=eval_classes,
                eval_types=eval_types)
            
            for ap_type, ap_value in result_dict.items():
                ap_dict[f'{name}/{ap_type}'] = float(f'{ap_value:.4f}')

            if verbose and ('2d' not in name):
                print(result_string)
        
        if save_path is not None:
            with open(save_path, 'w') as make_json:
                json.dump(ap_dict, make_json)
        return ap_dict
