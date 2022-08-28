import cv2
import numpy as np

import math
import torch
import matplotlib.pyplot as plt

from typing import Any, Dict
from torchvision.utils import make_grid


def _auto_scale_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.max() <= 1.0:
        tensor = (tensor * 255.).type(torch.IntTensor)
    return tensor.clamp(0, 255)


def visualize_tensor(tensor: torch.Tensor, 
                     invert_channels: bool = False,
                     return_tensor: bool = False):
    
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    
    if tensor.dim() == 3:
        if tensor.shape[0] in [1, 3]:
            tensor = _auto_scale_tensor(tensor.permute(1, 2, 0)).detach().cpu().numpy()
        else:
            tensor = _auto_scale_tensor(tensor)
            tensor = [t for t in tensor.unsqueeze(1)]
            num_grids = len(tensor)
            
            tensor = make_grid(tensor, nrow=math.ceil(math.sqrt(num_grids)), padding=1).permute(1, 2, 0).detach().cpu().numpy()
    
    elif tensor.dim() == 4:
        if tensor.shape[1] in [1, 3]:
            tensor = _auto_scale_tensor(tensor)
            num_grids = len(tensor)
            tensor = make_grid(tensor, nrow=math.ceil(math.sqrt(num_grids)), padding=1).permute(1, 2, 0).detach().cpu().numpy()
            
        else:
            print("It seems like the tensor is not in the correct format.")
    
    if invert_channels:
        tensor = tensor[:, :, ::-1]
    
    if not return_tensor:  
        plt.figure(figsize=(20, 20))
        plt.imshow(tensor)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
    else:
        return tensor


def plot_bboxes_2d(data_dict: Dict[str, Any],
                   title: str = None, 
                   invert_channels: bool = False,
                   scale_factor: int = 2,
                   save_path: str = None,
                   return_as_array: bool = False):
    
    # Parse data from 'data_dict'.
    img_tensor = data_dict['img']
    bbox_coord = data_dict['label']['gt_bboxes'][0]
    center_coord = data_dict['label']['centers2d'][0]
    
    img_tensor = visualize_tensor(img_tensor, return_tensor=True, invert_channels=invert_channels).copy()
    if img_tensor.dtype in ['float32', 'float64'] and img_tensor.max() > 10.:
        img_tensor = img_tensor.astype(np.uint8)
        
    bboxes = bbox_coord.detach().cpu().numpy().astype(np.int)
    centers = center_coord.detach().cpu().numpy().astype(np.int)
    
    for space_center, bbox in zip(centers, bboxes):
        img_tensor = cv2.rectangle(img_tensor, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        bbox_center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
        img_tensor = cv2.line(img_tensor, bbox_center, bbox_center, (0, 255, 0), 2)
        img_tensor = cv2.line(img_tensor, space_center, space_center, (0, 255, 0), 2)
        
        img_tensor = cv2.arrowedLine(img_tensor, space_center, bbox_center, (0, 255, 0), 2, cv2.LINE_AA)
        
    if return_as_array:
        if (img_tensor.dtype in [np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64]):
            img_tensor = img_tensor.astype(np.uint8)
        return cv2.resize(img_tensor, dsize=(0, 0), fx=scale_factor, fy=scale_factor)
    
    if save_path is None:
        plt.figure(figsize=(20, 20))
        plt.imshow(img_tensor)
        plt.xticks([])
        plt.yticks([])
        
        if title is not None:
            plt.title(title)
        
        plt.show()
        plt.close()
        
    else:
        if invert_channels:
            img_tensor = img_tensor[:, :, ::-1]
        img_tensor = cv2.resize(img_tensor, dsize=(0, 0), fx=scale_factor, fy=scale_factor)
        cv2.imwrite(save_path, img_tensor)



def draw_projected_box3d(image: np.ndarray, 
                         qs: np.ndarray, 
                         color=(0, 255, 0), 
                         thickness=2):
    
    """ 
    Draw 3d bounding box in image
    
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    
    if qs is None:
        return image
    
    if isinstance(qs, list):
        qs = np.array(qs)
    if qs.ndim == 2:
        qs = qs[np.newaxis, ...]
    qs = qs.astype(np.int32)
    
    for single_qs in qs:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(image, (single_qs[i, 0], single_qs[i, 1]), (single_qs[j, 0], single_qs[j, 1]), color, thickness, cv2.LINE_AA)
            
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(image, (single_qs[i, 0], single_qs[i, 1]), (single_qs[j, 0], single_qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(image, (single_qs[i, 0], single_qs[i, 1]), (single_qs[j, 0], single_qs[j, 1]), color, thickness, cv2.LINE_AA)
    return image


def plot_kpts_2d(data_dict: Dict[str, Any],
                 scale_factor: int = 2,
                 invert_channels: bool = False,
                 save_path: str = None,
                 return_as_array: bool = False):
    
    # Parse data from 'data_dict'.
    img_tensor = data_dict['img']
    
    mask = data_dict['label']['mask'][0].type(torch.BoolTensor)
    
    kpts_2d = data_dict['label']['gt_kpts_2d'][0][mask]
    valid_kpts = data_dict['label']['gt_kpts_valid_mask'][0][mask]
    
    # Image
    img_tensor = visualize_tensor(img_tensor, return_tensor=True, invert_channels=invert_channels).copy()
    if img_tensor.dtype in ['float32', 'float64'] and img_tensor.max() > 10.:
        img_tensor = img_tensor.astype(np.uint8)
    
    # Keypoints
    # 'kpts_arr': (# Boxes, # Keypoints, 2)
    # 'mask_arr': (# Boxes, # Keypoints, 1)
    kpts_arr = kpts_2d.data.reshape(-1, 9, 2).detach().cpu().numpy().astype(int)
    mask_arr = valid_kpts.data.detach().cpu().numpy()[..., np.newaxis]
    
    # Draw
    try:
        kpts = np.concatenate([kpts_arr, mask_arr], axis=-1).astype(np.int)
        for kpt in kpts:
            for k in kpt:
                if k[-1] == 1:
                    c = (255, 0, 0)
                elif k[-1] == 2:
                    c = (0, 255, 0)
                
                img_tensor = cv2.line(img_tensor, (k[0], k[1]), (k[0], k[1]), c, 3, cv2.LINE_AA)
    
    except Exception as e:
        print(f'Error: "{e}"')
    
    if return_as_array:
        if (img_tensor.dtype in [np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64]):
            img_tensor = img_tensor.astype(np.uint8)
        return cv2.resize(img_tensor, dsize=(0, 0), fx=scale_factor, fy=scale_factor)
            
    if save_path is None:
        plt.figure(figsize=(20, 20))
        plt.imshow(img_tensor)
        plt.xticks([])
        plt.yticks([])
        
        plt.show()
        plt.close()
        
    else:
        if invert_channels:
            img_tensor = img_tensor[:, :, ::-1]
        img_tensor = cv2.resize(img_tensor, dsize=(0, 0), fx=scale_factor, fy=scale_factor)
        cv2.imwrite(save_path, img_tensor)
    
        

def plot_all(results: Dict[str, Any], save_path: str = None):
    
    # (H, W, C) format
    bboxes_2d = plot_bboxes_2d(data_dict=results,
                               invert_channels=True,
                               return_as_array=True)
    
    kpts_2d = plot_kpts_2d(data_dict=results,
                           invert_channels=True,
                           return_as_array=True)

    margin = np.zeros((20, *bboxes_2d.shape[1:]))
    plot = np.concatenate([bboxes_2d, margin, kpts_2d], axis=0)
    
    if ('float' in str(plot.dtype)) and (plot.max() > 100.):
        plot = plot.astype(np.uint8)
        
    if save_path is None:
        plt.figure(figsize=(20, 20))
        plt.imshow(plot)
        plt.xticks([])
        plt.yticks([])
        
        plt.show()
        plt.close()
        
    else:
        plot = plot[:, :, ::-1]
        plot = (plot * 255.).astype(np.uint8)
        cv2.imwrite(save_path, plot)
