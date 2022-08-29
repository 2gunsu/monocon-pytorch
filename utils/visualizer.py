import os
import cv2
import sys
import torch
import numpy as np

from typing import Union, Tuple, List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.geometry_ops import extract_corners_from_bboxes_3d, points_cam2img
from dataset.base_dataset import BaseKITTIMono3DDataset


CLASSES = ['Pedestrian', 'Cyclist', 'Car']
CLASS_IDX_TO_COLOR = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
}


class Visualizer:
    def __init__(self, 
                 dataset: BaseKITTIMono3DDataset,
                 vis_format: List[Dict[str, Any]]):
        
        # Dataset which provides ground-truth annotations.
        # Length of the dataset must be equal to the length of 'vis_format'.
        assert (len(dataset) == len(vis_format)), \
            "Length of the dataset must be equal to the length of 'vis_format'."
        self.dataset = dataset
        
        # Parse 'vis_format'
        self.pred_bbox_2d = [f['img_bbox2d'] for f in vis_format]
        self.pred_bbox_3d = [f['img_bbox'] for f in vis_format]
    
    
    def get_labels(self, idx: int, search_key: Union[List[str], str]) -> List[np.ndarray]:
        label = self.dataset[idx]['label']
        mask = label['mask'].type(torch.BoolTensor)
        
        if isinstance(search_key, str):
            search_key = [search_key,]
        
        result = []
        for key in search_key:
            search_value = label[key][mask].numpy()
            result.append(search_value)
        return result
        
    
    def plot_bboxes_2d(self, idx: int, save_path: str = None) -> Union[None, np.ndarray]:
        
        # Load Image
        image = self.dataset.load_image(idx)[0]         # (H, W, 3)
        
        # Load 2D Predicted Boxes and Draw
        pred_bboxes = self.pred_bbox_2d[idx]
        for c_idx, pred_bbox in enumerate(pred_bboxes):
            
            if len(pred_bbox) == 0:
                continue
            
            color = CLASS_IDX_TO_COLOR[c_idx]
            for box in pred_bbox:
                box = box[:-1].astype(np.int)
                image = self._add_transparent_box(image, box, color, alpha=0.2)
        
        if save_path is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, image)
        else:
            return image
        
        
    def plot_bboxes_3d(self, idx: int, save_path: str = None) -> Union[None, np.ndarray]:
        
        # Load Image
        image = self.dataset.load_image(idx)[0]         # (H, W, 3)
        
        # Load Calib
        calib = self.dataset.load_calib(idx)
        intrinsic_mat = calib.P2                        # (3, 4)
        
        # Load 3D Predicted Boxes
        pred_bboxes_3d = self.pred_bbox_3d[idx]['boxes_3d']
        pred_labels_3d = self.pred_bbox_3d[idx]['labels_3d']
        
        if len(pred_bboxes_3d) > 0:
            # Draw 3D Boxes
            line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                            (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
            # pred_bboxes_3d = self._post_process_bboxes_3d(pred_bboxes_3d)
            
            for bbox_3d, label_3d in zip(pred_bboxes_3d, pred_labels_3d):
                corners = extract_corners_from_bboxes_3d(bbox_3d.unsqueeze(0))[0]           # (8, 3)
                
                proj_corners = points_cam2img(corners, intrinsic_mat)                       # (8, 2)
                proj_corners = (proj_corners - 1).round().astype(np.int)                    # (8, 2)
                
                color = CLASS_IDX_TO_COLOR[label_3d.item()]
                for start, end in line_indices:
                    image = cv2.line(image, 
                                    (proj_corners[start, 0], proj_corners[start, 1]),
                                    (proj_corners[end, 0], proj_corners[end, 1]),
                                    color,
                                    thickness=2,
                                    lineType=cv2.LINE_AA)
        
        if save_path is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, image)
        else:
            return image
    
    # TODO
    def plot_bev(self, idx: int, save_path: str = None) -> Union[None, np.ndarray]:
        raise NotImplementedError
    
    
    def _add_transparent_box(self, 
                             image: np.ndarray, 
                             box_coordinate: Tuple[int, int, int, int],
                             color: Tuple[int, int, int],
                             alpha: float = 0.2) -> np.ndarray:

        x1, y1, x2, y2 = box_coordinate
        
        ori_image = image.copy()
        ori_image = cv2.rectangle(ori_image, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)
        
        overlay = image.copy()
        overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        return cv2.addWeighted(overlay, alpha, ori_image, (1 - alpha), 0)
