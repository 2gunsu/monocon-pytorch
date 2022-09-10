import os
import cv2
import sys
import torch
import numpy as np

from tqdm.auto import tqdm
from torch.utils.data import Dataset
from typing import Union, Tuple, List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.engine_utils import tprint
from utils.geometry_ops import extract_corners_from_bboxes_3d, points_cam2img


CLASSES = ['Pedestrian', 'Cyclist', 'Car']
CLASS_IDX_TO_COLOR = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
}


class Visualizer:
    def __init__(self, 
                 dataset: Dataset,
                 vis_format: List[Dict[str, Any]],
                 scale_hw: Tuple[float, float] = None):
        
        # Dataset which provides ground-truth annotations.
        # Length of the dataset must be equal to the length of 'vis_format'.
        assert (len(dataset) == len(vis_format)), \
            "Length of the dataset must be equal to the length of 'vis_format'."
        self.dataset = dataset
        
        # Parse 'vis_format'
        self.pred_bbox_2d = [f['img_bbox2d'] for f in vis_format]
        self.pred_bbox_3d = [f['img_bbox'] for f in vis_format]
        
        # Scale parameter needed to fit the predicted boxes to the original image.
        if (scale_hw is None):
            scale_hw = np.array([1., 1.])
        self.scale_hw = scale_hw
        
        # Mode
        self.mode = 'raw' if (dataset.__class__.__name__ == 'KITTIRawDataset') else 'normal'
    
    
    def get_labels(self, idx: int, search_key: Union[List[str], str]) -> List[np.ndarray]:
        
        assert (self.mode == 'normal'), \
            "This method is only available in 'normal' mode."
        
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
        if self.mode == 'normal':
            image = self.dataset.load_image(idx)[0]         # (H, W, 3)
        else:
            image = self.dataset[idx]['ori_img']
        
        # Load 2D Predicted Boxes and Draw
        pred_bboxes = self.pred_bbox_2d[idx]
        for c_idx, pred_bbox in enumerate(pred_bboxes):
            
            if len(pred_bbox) == 0:
                continue
            
            color = CLASS_IDX_TO_COLOR[c_idx]
            for box in pred_bbox:
                s = np.reciprocal(np.array([*self.scale_hw[::-1], *self.scale_hw[::-1]]))
                box = (box[:-1] * s).astype(np.int)
                image = self._add_transparent_box(image, box, color, alpha=0.2)
        
        if save_path is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, image)
        else:
            return image
        
        
    def plot_bboxes_3d(self, idx: int, save_path: str = None) -> Union[None, np.ndarray]:
        
        # Load Image
        if self.mode == 'normal':
            image = self.dataset.load_image(idx)[0]         # (H, W, 3)
        else:
            image = self.dataset[idx]['ori_img']
        
        # Load Calib
        if self.mode == 'normal':
            calib = self.dataset.load_calib(idx)
        else:
            calib = self.dataset[idx]['calib'][0]
            
        intrinsic_mat = calib.P2                        # (3, 4)
        
        # Load 3D Predicted Boxes
        pred_bboxes_3d = self.pred_bbox_3d[idx]['boxes_3d']
        pred_labels_3d = self.pred_bbox_3d[idx]['labels_3d']
        
        if len(pred_bboxes_3d) > 0:
            # Draw 3D Boxes
            line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                            (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))

            for bbox_3d, label_3d in zip(pred_bboxes_3d, pred_labels_3d):
                corners = extract_corners_from_bboxes_3d(bbox_3d.unsqueeze(0))[0]               # (8, 3)
                
                proj_corners = points_cam2img(corners, intrinsic_mat)                           # (8, 2)
                
                s = np.reciprocal(self.scale_hw[::-1])
                proj_corners = ((proj_corners - 1).round() * s).astype(np.int)                  # (8, 2)
                
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
    
    
    def plot_bev(self, idx: int, save_path: str = None) -> Union[None, np.ndarray]:
        
        MAX_DIST = 60
        SCALE = 10

        # Create BEV Space
        R = (MAX_DIST * SCALE)
        space = np.zeros((R * 2, R * 2, 3), dtype=np.uint8)
        
        for theta in np.linspace(0, np.pi, 7):
            space = cv2.line(space,
                            pt1=(int(R - R * np.cos(theta)), int(R - R * np.sin(theta))),
                            pt2=(R, R),
                            color=(255, 255, 255), 
                            thickness=2,
                            lineType=cv2.LINE_AA)
        
        for radius in np.linspace(0, R, 5):
            if radius == 0:
                continue
            
            space = cv2.circle(space, 
                               center=(R, R), 
                               radius=int(radius), 
                               color=(255, 255, 255), 
                               thickness=2,
                               lineType=cv2.LINE_AA)
        space = space[:R, :, :]
        
        # Load 3D Predicted Boxes
        pred_bboxes_3d = self.pred_bbox_3d[idx]['boxes_3d']                 # (N, 7)
        pred_labels_3d = self.pred_bbox_3d[idx]['labels_3d']                # (N,)
        
        # Draw BEV Boxes on Space
        if len(pred_bboxes_3d) > 0:
            pred_bev = pred_bboxes_3d[:, [0, 2, 3, 5, 6]]                   # (N, 5) / (XYWHR)
            
            pred_bev[:, :-1] *= SCALE
            pred_bev[:, 1] *= (-1)
            pred_bev[:, :2] += R
            
            for idx, bev in enumerate(pred_bev):
                
                bev = tuple(bev.numpy())
                box = cv2.boxPoints((bev[:2], bev[2:4], (bev[4] * 180 / np.pi)))
                box = np.int0(box)
                
                label = pred_labels_3d[idx].item()
                color = CLASS_IDX_TO_COLOR[label]
                space = cv2.drawContours(space, [box], -1, color, thickness=-1, lineType=cv2.LINE_AA)            
        
        if save_path is not None:
            space = cv2.cvtColor(space, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, space)
        else:
            return space
        
        
    def export_as_video(self,
                        save_dir: str,
                        plot_items: List[str] = ['2d', '3d', 'bev'],
                        fps: int = 20) -> None:
        
        assert (self.mode == 'raw'), "This method is only available in 'raw' mode."
        
        item_to_draw_func = {
            '2d': self.plot_bboxes_2d,
            '3d': self.plot_bboxes_3d,
            'bev': self.plot_bev}
        
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            
        for plot_item in plot_items:
            vid_file = os.path.join(save_dir, f'{plot_item}.mp4')
            vis_results = []
            
            for idx in tqdm(range(len(self.dataset)), desc=f"Visualizing '{plot_item}'..."):
                vis_result = item_to_draw_func[plot_item](idx, save_path=None)
                vis_results.append(vis_result)
            
            img_size = vis_results[0].shape[:2][::-1]           # (W, H)    
            vid_writer = cv2.VideoWriter(vid_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, img_size)
            
            for v in vis_results:
                v = v.astype(np.uint8)
                v = cv2.cvtColor(v, code=cv2.COLOR_RGB2BGR)
                vid_writer.write(v)
            vid_writer.release()
            
            tprint(f"Video for '{plot_item}' is exported to '{vid_file}'.")
            
        
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
