import os
import sys
import torch
import random
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typing import Tuple, Dict, Any

from transforms.default_transforms import BaseTransform


class RandomCrop3D(BaseTransform):
    def __init__(self, 
                 prob: float = 0.50, 
                 crop_size: Tuple[int, int] = (320, 960),
                 hide_kpts_in_crop_area: bool = False,
                 area_filter_thres: float = 0.20):
        
        super().__init__(True, True, False, True)
        
        assert (0.0 <= prob <= 1.0)
        self.prob = prob
        
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size
        self.hide_kpts_in_crop_area = hide_kpts_in_crop_area
    
        assert (0.0 <= area_filter_thres < 1.0)
        self.area_filter_thres = area_filter_thres
    
    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        
        img_shape = data_dict['img_metas']['ori_shape']
        assert sum(np.array(self.crop_size) <= np.array(img_shape)) == 2, \
            f"Crop size should be smaller than image size. (crop size: {self.crop_size}, image size: {img_shape})"
        
        data_dict, is_cropped = self._select_crop_pos(data_dict)
        if not is_cropped:
            return data_dict
        
        else:
            
            img = data_dict['img']
            metas = data_dict['img_metas']
            label = data_dict['label']
            
            crop_coord = metas['crop_coord']                                                # (x_min, y_min, x_max, y_max)
            
            # Crop Labels
            gt_bboxes = label['gt_bboxes']                                                  # (# Max Objs, 4)
            ori_mask, new_mask = label['mask'], []                                          # (# Max Objs,)
            
            for idx, (gt_bbox, mask) in enumerate(zip(gt_bboxes, ori_mask)):
                
                if (mask == 0):
                    new_mask.append(0.)
                    continue
                
                rel_type, coord = self._compare_two_area(
                    frame_bbox=np.array(crop_coord),
                    gt_bbox=gt_bbox)
                    
                if rel_type == 'within':
                    new_mask.append(1.)
                    continue
                    
                elif rel_type == 'inters':
                    
                    lt = coord[:2]
                    rb = coord[2:]
                    
                    new_box_w = (rb[0] - lt[0])
                    new_box_h = (rb[1] - lt[1])
                    
                    new_box_area = new_box_w * new_box_h
                    ori_box_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
                    
                    ratio = new_box_area / ori_box_area
                    validity = (ratio >= self.area_filter_thres)
                    
                    if validity:
                        new_mask.append(1.)
                    else:
                        new_mask.append(0.)
                        continue
                    gt_bboxes[idx] = coord
                    
                else:
                    new_mask.append(0.)
                    continue
                
            new_mask = np.array(new_mask)
            updated_mask = np.logical_and(ori_mask, new_mask)
            
            # to solve the problem: after random crop, the target becomes to None, just return original data
            if not(new_mask.any()):
                return data_dict
            
            # 'gt_bboxes' and 'gt_labels'
            data_dict['label']['gt_bboxes'] = (gt_bboxes * updated_mask[..., np.newaxis])
            data_dict['label']['gt_labels'] *= updated_mask
            
            # 'gt_bboxes_3d' and 'gt_labels_3d'
            data_dict['label']['gt_bboxes_3d'] *= updated_mask[..., np.newaxis]
            data_dict['label']['gt_labels_3d'] *= updated_mask
        
            # 'centers2d' and 'depths'
            data_dict['label']['centers2d'] *= updated_mask[..., np.newaxis]
            data_dict['label']['depths'] *= updated_mask
            
            # 'gt_kpts_2d' and 'gt_kpts_valid_mask'
            data_dict['label']['gt_kpts_2d'] *= updated_mask[..., np.newaxis]
            data_dict['label']['gt_kpts_valid_mask'] *= updated_mask[..., np.newaxis]
            
            # 'mask'
            data_dict['label']['mask'] = updated_mask
            
            if self.hide_kpts_in_crop_area:
                data_dict = self._filter_kpts(data_dict)
            
            # Crop Image
            canvas = np.zeros_like(img)
            canvas[crop_coord[1]: crop_coord[3], crop_coord[0]: crop_coord[2], :] = \
                img[crop_coord[1]: crop_coord[3], crop_coord[0]: crop_coord[2], :]
            data_dict['img'] = canvas
            return data_dict
        
    def _select_crop_pos(self, data_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        metas = data_dict['img_metas']
        if random.random() >= self.prob:
            metas['is_cropped'] = False
            metas['crop_coord'] = (0, 0, 0, 0)
            data_dict['img_metas'] = metas
            return (data_dict, False)
        
        else:
            
            ori_h, ori_w = metas['ori_shape']
            crop_h, crop_w = self.crop_size
            
            crop_h_start = random.randint(0, (ori_h - crop_h))
            crop_w_start = random.randint(0, (ori_w - crop_w))
            
            crop_h_end = crop_h_start + crop_h
            crop_w_end = crop_w_start + crop_w
            
            # (x_min, y_min, x_max, y_max) format
            crop_coord = (crop_w_start, crop_h_start, crop_w_end, crop_h_end)
            
            metas['is_cropped'] = True
            metas['crop_coord'] = crop_coord
            return (data_dict, True)
        
    def _filter_kpts(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        gt_kpts_2d = data_dict['label']['gt_kpts_2d']                       # (# Max Objs, 18)
        gt_kpts_valid_mask = data_dict['label']['gt_kpts_valid_mask']       # (# Max Objs, 9)
        obj_masks = data_dict['label']['mask']                              # (# Max Objs,)
        
        valid_region = data_dict['img_metas']['crop_coord']                 # (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = valid_region
        
        # 'gt_kpt'      : (18,)
        # 'gt_kpt_mask' : (9,)
        for obj_idx, (gt_kpt, gt_kpt_mask, obj_mask) in enumerate(zip(gt_kpts_2d, gt_kpts_valid_mask, obj_masks)):
            
            if obj_mask == 0:
                continue
            
            gt_kpt_coord = gt_kpt.reshape(9, 2)
            
            x_coord = gt_kpt_coord[:, 0]
            x_cond = np.logical_and(
                (x_min <= x_coord),
                (x_coord <= x_max),
            )
            
            y_coord = gt_kpt_coord[:, 1]
            y_cond = np.logical_and(
                (y_min <= y_coord),
                (y_coord <= y_max),
            )
            
            cond = np.logical_and(x_cond, y_cond)
            gt_kpt_mask[~cond] = 1
            
            data_dict['label']['gt_kpts_valid_mask'][obj_idx] = gt_kpt_mask
        return data_dict
    
    def _compare_two_area(self,
                          frame_bbox: np.ndarray, 
                          gt_bbox: np.ndarray) -> Tuple[str, Any]:
        
        f_xmin, f_ymin, f_xmax, f_ymax = frame_bbox
        g_xmin, g_ymin, g_xmax, g_ymax = gt_bbox
        
        inter_xmin = max(f_xmin, g_xmin)
        inter_ymin = max(g_ymin, f_ymin)
        inter_xmax = min(f_xmax, g_xmax)
        inter_ymax = min(g_ymax, f_ymax)
        
        inter = np.array([inter_xmin, inter_ymin, inter_xmax, inter_ymax])
        if np.allclose(inter, gt_bbox):
            return ('within', gt_bbox)
        
        elif (inter_xmax <= inter_xmin) or (inter_ymax <= inter_ymin):
            return ('out', None)
        
        else:
            return ('inters', inter) 
        

class RandomRangeCrop3D(BaseTransform):
    def __init__(self, 
                 prob: float = 0.50,
                 height_range: Tuple[int, int] = (256, 320),
                 aspect_ratio: float = 3.0,
                 hide_kpts_in_crop_area: bool = True,
                 area_filter_thres: float = 0.20):
        
        super().__init__(True, True, False, True)
        
        assert (0.0 <= prob <= 1.0)
        self.prob = prob
        
        if isinstance(height_range, int):
            height_range = (height_range, height_range)
        self.height_range = height_range
        self.width_range = (
            int(height_range[0] * aspect_ratio), 
            int(height_range[1] * aspect_ratio))
        
        self.hide_kpts_in_crop_area = hide_kpts_in_crop_area
    
        assert (0.0 <= area_filter_thres < 1.0)
        self.area_filter_thres = area_filter_thres
    
    
    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        
        data_dict, is_cropped = self._select_crop_pos(data_dict)
        if not is_cropped:
            return data_dict
        
        else:
            
            img = data_dict['img']
            metas = data_dict['img_metas']
            label = data_dict['label']
            
            crop_coord = metas['crop_coord']                                                # (x_min, y_min, x_max, y_max)
            
            # Crop Labels
            gt_bboxes = label['gt_bboxes']                                                  # (# Max Objs, 4)
            ori_mask, new_mask = label['mask'], []                                          # (# Max Objs,)
            
            for idx, (gt_bbox, mask) in enumerate(zip(gt_bboxes, ori_mask)):
                
                if (mask == 0):
                    new_mask.append(0.)
                    continue
                
                rel_type, coord = self._compare_two_area(
                    frame_bbox=np.array(crop_coord),
                    gt_bbox=gt_bbox)
                    
                if rel_type == 'within':
                    new_mask.append(1.)
                    continue
                    
                elif rel_type == 'inters':
                    
                    lt = coord[:2]
                    rb = coord[2:]
                    
                    new_box_w = (rb[0] - lt[0])
                    new_box_h = (rb[1] - lt[1])
                    
                    new_box_area = new_box_w * new_box_h
                    ori_box_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
                    
                    ratio = new_box_area / ori_box_area
                    validity = (ratio >= self.area_filter_thres)
                    
                    if validity:
                        new_mask.append(1.)
                    else:
                        new_mask.append(0.)
                        continue
                    gt_bboxes[idx] = coord
                    
                else:
                    new_mask.append(0.)
                    continue
                
            new_mask = np.array(new_mask)
            updated_mask = np.logical_and(ori_mask, new_mask)
            
            # 'gt_bboxes' and 'gt_labels'
            data_dict['label']['gt_bboxes'] = (gt_bboxes * updated_mask[..., np.newaxis])
            data_dict['label']['gt_labels'] *= updated_mask
            
            # 'gt_bboxes_3d' and 'gt_labels_3d'
            data_dict['label']['gt_bboxes_3d'] *= updated_mask[..., np.newaxis]
            data_dict['label']['gt_labels_3d'] *= updated_mask
        
            # 'centers2d' and 'depths'
            data_dict['label']['centers2d'] *= updated_mask[..., np.newaxis]
            data_dict['label']['depths'] *= updated_mask
            
            # 'gt_kpts_2d' and 'gt_kpts_valid_mask'
            data_dict['label']['gt_kpts_2d'] *= updated_mask[..., np.newaxis]
            data_dict['label']['gt_kpts_valid_mask'] *= updated_mask[..., np.newaxis]
            
            # 'mask'
            data_dict['label']['mask'] = updated_mask
            
            if self.hide_kpts_in_crop_area:
                data_dict = self._filter_kpts(data_dict)
            
            
            # Crop Image
            canvas = np.zeros_like(img)
            canvas[crop_coord[1]: crop_coord[3], crop_coord[0]: crop_coord[2], :] = \
                img[crop_coord[1]: crop_coord[3], crop_coord[0]: crop_coord[2], :]
            data_dict['img'] = canvas
            return data_dict
        
    
    def _select_crop_pos(self, data_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        
        metas = data_dict['img_metas']
        if random.random() >= self.prob:
            metas['is_cropped'] = False
            metas['crop_coord'] = (0, 0, 0, 0)
            data_dict['img_metas'] = metas
            return (data_dict, False)
        
        else:
            ori_h, ori_w = metas['ori_shape']

            crop_h = random.randint(*self.height_range)
            crop_w = random.randint(*self.width_range)
            
            crop_h_start = random.randint(0, (ori_h - crop_h))
            crop_w_start = random.randint(0, (ori_w - crop_w))
            
            crop_h_end = crop_h_start + crop_h
            crop_w_end = crop_w_start + crop_w
            
            # (x_min, y_min, x_max, y_max) format
            crop_coord = (crop_w_start, crop_h_start, crop_w_end, crop_h_end)
            
            metas['is_cropped'] = True
            metas['crop_coord'] = crop_coord
            return (data_dict, True)
        
        
    def _filter_kpts(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        
        gt_kpts_2d = data_dict['label']['gt_kpts_2d']                       # (# Max Objs, 18)
        gt_kpts_valid_mask = data_dict['label']['gt_kpts_valid_mask']       # (# Max Objs, 9)
        obj_masks = data_dict['label']['mask']                              # (# Max Objs,)
        
        valid_region = data_dict['img_metas']['crop_coord']                 # (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = valid_region
        
        # 'gt_kpt'      : (18,)
        # 'gt_kpt_mask' : (9,)
        for obj_idx, (gt_kpt, gt_kpt_mask, obj_mask) in enumerate(zip(gt_kpts_2d, gt_kpts_valid_mask, obj_masks)):
            
            if obj_mask == 0:
                continue
            
            gt_kpt_coord = gt_kpt.reshape(9, 2)
            
            x_coord = gt_kpt_coord[:, 0]
            x_cond = np.logical_and(
                (x_min <= x_coord),
                (x_coord <= x_max),
            )
            
            y_coord = gt_kpt_coord[:, 1]
            y_cond = np.logical_and(
                (y_min <= y_coord),
                (y_coord <= y_max),
            )
            
            cond = np.logical_and(x_cond, y_cond)
            gt_kpt_mask[~cond] = 1
            
            data_dict['label']['gt_kpts_valid_mask'][obj_idx] = gt_kpt_mask
        return data_dict
    
    
    def _compare_two_area(self,
                          frame_bbox: np.ndarray, 
                          gt_bbox: np.ndarray) -> Tuple[str, Any]:
        
        f_xmin, f_ymin, f_xmax, f_ymax = frame_bbox
        g_xmin, g_ymin, g_xmax, g_ymax = gt_bbox
        
        inter_xmin = max(f_xmin, g_xmin)
        inter_ymin = max(g_ymin, f_ymin)
        inter_xmax = min(f_xmax, g_xmax)
        inter_ymax = min(g_ymax, f_ymax)
        
        inter = np.array([inter_xmin, inter_ymin, inter_xmax, inter_ymax])
        if np.allclose(inter, gt_bbox):
            return ('within', gt_bbox)
        
        elif (inter_xmax <= inter_xmin) or (inter_ymax <= inter_ymin):
            return ('out', None)
        
        else:
            return ('inters', inter) 
