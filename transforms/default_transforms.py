import os
import sys
import cv2
import torch
import numpy as np

from numpy import random
from numbers import Number
from typing import Tuple, Union, List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from transforms import BaseTransform


class Resize3D(BaseTransform):
    def __init__(self, target_hw: Union[int, Tuple[int, int]] = None):
        super().__init__(True, True, True, True)
        
        if (target_hw is not None) and isinstance(target_hw, int):
            target_hw = (target_hw, target_hw)
        self.target_hw = target_hw
        
    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        
        if self.target_hw is None:
            return data_dict
         
        # Resize Image
        img = data_dict['img']                          # (H, W, C) / np.ndarray
        ori_hw = img.shape[:2]
        img = cv2.resize(img, self.target_hw[::-1])     # (target_H, target_W, C) / np.ndarray
        data_dict['img'] = img
        
        # Get Rescale Factor
        scale_hw = (np.array(self.target_hw) / np.array(ori_hw))
        
        # Update Meta
        data_dict['img_metas']['scale_hw'] = scale_hw
        data_dict['img_metas']['ori_shape'] = self.target_hw
        
        # Update Calib
        data_dict['calib'].rescale(*scale_hw[::-1])
        
        # Resize Label
        data_dict['label']['gt_bboxes'] *= np.array([*scale_hw[::-1], *scale_hw[::-1]])     # 'gt_bboxes'
        data_dict['label']['centers2d'] *= scale_hw[::-1]                                   # 'centers2d'
        data_dict['label']['gt_kpts_2d'] *= np.tile(scale_hw[::-1], 9)                      # 'gt_kpts_2d'
        
        return data_dict


class PhotometricDistortion(BaseTransform):
    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Tuple[float, float] = (0.5, 1.5),
                 saturation_range: Tuple[float, float] = (0.5, 1.5),
                 hue_delta: int = 18):
        
        super().__init__(True, False, False, False)
        
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        
    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        # Convert RGB to BGR
        img = data_dict['img'].astype(np.float32)
        img = img[:, :, ::-1]
        
        # Random Brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            img = (img + delta)

        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                img = (img * alpha)

        # Convert BGR to HSV
        img = self._convert_color_factory('bgr', 'hsv')(img)

        # Random Saturation
        if random.randint(2):
            img[..., 1] = (img[..., 1] * random.uniform(self.saturation_lower, self.saturation_upper))

        # Random Hue
        if random.randint(2):
            img[..., 0] = img[..., 0] + random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] = (img[..., 0][img[..., 0] > 360] - 360)
            img[..., 0][img[..., 0] < 0] = (img[..., 0][img[..., 0] < 0]) + 360

        # Convert HSV to BGR
        img = self._convert_color_factory('hsv', 'bgr')(img)

        # Random Contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img = (img * alpha)

        # Randomly Swap Channels
        if random.randint(2):
            img = img[..., random.permutation(3)]
            
        # Convert BGR to RGB
        img = img[:, :, ::-1]
        
        data_dict['img'] = img
        return data_dict
        
    def _convert_color_factory(self, src: str, dst: str):
        code = getattr(cv2, f'COLOR_{src.upper()}2{dst.upper()}')
        def convert_color(img):
            out_img = cv2.cvtColor(img, code)
            return out_img
        return convert_color
    
    
class RandomShift(BaseTransform):
    def __init__(self, 
                 prob: float = 0.50, 
                 shift_range: Tuple[float, float] = (-32.0, 32.0),
                 hide_kpts_in_shift_area: bool = True):
        
        super().__init__(True, True, True, True)
        
        assert (0.0 <= prob <= 1.0)
        self.prob = prob
        
        assert len(shift_range) == 2, \
            "Argument 'shift_range' must be given as a tuple of length 2."
        self.shift_range = shift_range
        self.hide_kpts_in_shift_area = hide_kpts_in_shift_area
        
    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        
        if random.random() >= self.prob:
            return self._break(data_dict)
        
        else:
            metas = data_dict['img_metas']
            img_shape = metas['ori_shape']
            
            sft_x = int(random.uniform(*self.shift_range))
            sft_y = int(random.uniform(*self.shift_range))
            
            # Shift Labels
            label = data_dict['label']
            updated_mask = label['mask'].copy()                         # (# Max Objs,)
            
            # (1) gt_bboxes
            gt_bboxes = label['gt_bboxes'].copy()                       # (# Max Objs, 4)
            
            gt_bboxes[..., 0::2] += sft_x
            gt_bboxes[..., 0::2] = np.clip(gt_bboxes[..., 0::2], 0, img_shape[1])
            
            gt_bboxes[..., 1::2] += sft_y
            gt_bboxes[..., 1::2] = np.clip(gt_bboxes[..., 1::2], 0, img_shape[0])
            
            bbox_w = gt_bboxes[..., 2] - gt_bboxes[..., 0]
            bbox_h = gt_bboxes[..., 3] - gt_bboxes[..., 1]
            validity = (bbox_w > 1) & (bbox_h > 1)
            
            if not validity.any():
                return self._break(data_dict)
            
            metas['is_shifted'] = True
            metas['shift_params'] = (sft_x, sft_y)
            data_dict['img_metas'] = metas
            
            updated_mask = np.logical_and(validity, updated_mask)                       # (# Max Objs,)
            gt_bboxes = (gt_bboxes * updated_mask[..., np.newaxis])                     # (# Max Objs, 4)
            data_dict['label']['gt_bboxes'] = gt_bboxes
            
            
            # (2) gt_labels
            data_dict['label']['gt_labels'] *= updated_mask                             # (# Max Objs,)
            
            # (3) gt_bboxes_3d
            data_dict = self._shift_calib(data_dict, sft_x, sft_y)
            data_dict['label']['gt_bboxes_3d'] *= updated_mask[..., np.newaxis]         # (# Max Objs, 7)
            
            # (4) gt_labels_3d
            data_dict['label']['gt_labels_3d'] *= updated_mask                          # (# Max Objs,)
            
            # (5) centers2d
            centers2d = data_dict['label']['centers2d'].copy()                          # (# Max Objs, 2)
            
            centers2d[..., 0] += sft_x
            centers2d[..., 1] += sft_y
            
            centers2d *= updated_mask[..., np.newaxis]                                  # (# Max Objs, 2)
            data_dict['label']['centers2d'] = centers2d
            
            # (6) depths
            data_dict['label']['depths'] *= updated_mask                                # (# Max Objs,)
            
            # (7) gt_kpts_2d
            gt_kpts_2d = data_dict['label']['gt_kpts_2d'].copy()                        # (# Max Objs, 18)
            
            gt_kpts_2d[..., 0::2] += sft_x
            gt_kpts_2d[..., 1::2] += sft_y
            
            gt_kpts_2d *= updated_mask[..., np.newaxis]                                 # (# Max Objs, 18)
            data_dict['label']['gt_kpts_2d'] = gt_kpts_2d
            data_dict['label']['gt_kpts_valid_mask'] *= updated_mask[..., np.newaxis]   # (# Max Objs, 9)
            
            # (8) mask
            data_dict['label']['mask'] = updated_mask                                   # (# Max Objs,)
            

            # Shift Image
            img = data_dict['img']      # (H, W, C)
            canvas = np.zeros_like(img)
            
            new_x = max(0, sft_x)
            orig_x = max(0, -sft_x)
            
            new_y = max(0, sft_y)
            orig_y = max(0, -sft_y)
            
            new_h = img_shape[0] - np.abs(sft_y)
            new_w = img_shape[1] - np.abs(sft_x)
            
            canvas[new_y: (new_y + new_h), new_x: (new_x + new_w)] = \
                img[orig_y: (orig_y + new_h), orig_x: (orig_x + new_w)]
            data_dict['img'] = canvas
            return data_dict
    
    def _break(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        metas = data_dict['img_metas']
        metas['is_shifted'] = False
        metas['shift_params'] = (0, 0)
        data_dict['img_metas'] = metas
        
        return data_dict
    
    def _shift_calib(self, data_dict: Dict[str, Any], sft_x: int, sft_y: int) -> Dict[str, Any]:
        calib = data_dict['calib']
        
        proj_mat = calib.P2
        proj_mat[0, 2] += sft_x
        proj_mat[1, 2] += sft_y
        
        calib.P2 = proj_mat
        data_dict['calib'] = calib
        
        return data_dict
    
    def _filter_kpts(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        # Calculate valid label region
        metas = data_dict['img_metas']
        
        img_hw = metas['ori_shape']
        sft_x, sft_y = metas['shift_params']
        
        valid_xmin, valid_ymin, valid_xmax, valid_ymax = 0, 0, 0, 0
        if (sft_x >= 0):
            valid_xmin, valid_xmax = (sft_x, img_hw[1])
        else:
            valid_xmin, valid_xmax = (0, (img_hw[1] - sft_x))
            
        if (sft_y >= 0):
            valid_ymin, valid_ymax = (0, (img_hw[0] - sft_y))
        else:
            valid_ymin, valid_ymax = (-sft_y, img_hw[0])
            
        # Filter keypoints based on valid label region
        kpts = data_dict['label']['gt_kpts_2d']                     # (# Max Objs, 18)
        kpts_mask = data_dict['label']['gt_kpts_valid_mask']        # (# Max Objs, 9)
        objs_mask = data_dict['label']['mask']                      # (# Max Objs,)
        
        for idx, (kpt, kpt_mask, obj_mask) in enumerate(zip(kpts, kpts_mask, objs_mask)):
            if (obj_mask == 0):
                continue
            
            kpt = kpt.reshape(9, 2)             # (9, 2)
            kpt_x, kpt_y = kpt[:, 0], kpt[:, 1]
            
            kpt_x_flag = np.logical_and((kpt_x >= valid_xmin), (kpt_x <= valid_xmax))
            kpt_y_flag = np.logical_and((kpt_y >= valid_ymin), (kpt_y <= valid_ymax))

            kpt_flag = np.logical_and(kpt_x_flag, kpt_y_flag)
            kpt_mask[~kpt_flag] = 1
            
            data_dict['label']['gt_kpts_valid_mask'][idx] = kpt_mask
        return data_dict
        
        

class RandomHorizontalFlip(BaseTransform):
    def __init__(self, prob: float = 0.50):
        super().__init__(True, True, True, True)
        
        assert (0.0 <= prob <= 1.0)
        self.prob = prob
        
    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() >= self.prob:
            
            metas = data_dict['img_metas']
            metas['is_flipped'] = False
            data_dict['img_metas'] = metas
            
            return data_dict
        
        else:
            # Flip Image
            img = data_dict['img']
            data_dict['img'] = img[:, ::-1, :]
            
            # Update Meta
            metas = data_dict['img_metas']
            metas['is_flipped'] = True
            
            # Update Calibration
            calib = data_dict['calib']
            w = img.shape[1]
            
            P2 = calib.P2
            P2[0, 2] = (w - P2[0, 2] - 1)
            P2[0, 3] = -P2[0, 3]
            calib.P2 = P2
            
            data_dict['calib'] = calib
            
            # Flip Label
            #   --> ['gt_bboxes', 'gt_bboxes_3d', 'centers2d', 'gt_kpts_2d']
            label = data_dict['label']
            
            label['centers2d'][..., 0] = (w - label['centers2d'][..., 0] - 1) * label['mask']
            label['gt_bboxes'] = (self._flip_bboxes_2d(label['gt_bboxes'], metas) * label['mask'][..., np.newaxis])
            label['gt_bboxes_3d'] = (self._flip_bboxes_3d(label['gt_bboxes_3d']) * label['mask'][..., np.newaxis])
            
            # Change order of 'gt_kpts_2d'
            gt_kpts_2d = label['gt_kpts_2d'].copy()
            gt_kpts_2d[..., 0::2] = ((w - gt_kpts_2d[..., 0::2] - 1) * label['mask'][..., np.newaxis])
            max_objs, _ = gt_kpts_2d.shape
            
            gt_kpts_2d = gt_kpts_2d.reshape(max_objs, -1, 2)
            gt_kpts_2d[:, [0, 1, 2, 3, 4, 5, 6, 7]] = gt_kpts_2d[:, [1, 0, 3, 2, 5, 4, 7, 6]]
            label['gt_kpts_2d'] = gt_kpts_2d.reshape(max_objs, -1)
            
            # Change order of 'gt_kpts_valid_mask'
            gt_kpts_valid_mask = label['gt_kpts_valid_mask'].copy()
            gt_kpts_valid_mask[:, [0, 1, 2, 3, 4, 5, 6, 7]] = gt_kpts_valid_mask[:, [1, 0, 3, 2, 5, 4, 7, 6]]
            label['gt_kpts_valid_mask'] = gt_kpts_valid_mask
                
            data_dict['label'] = label
            return data_dict
            
    def _flip_bboxes_3d(self, bboxes_3d: np.ndarray) -> np.ndarray:
        
        # 'bboxes_3d': (B, K, 7)
        bboxes_3d[..., 0] = (bboxes_3d[..., 0] * -1)
        bboxes_3d[..., -1] = (bboxes_3d[..., -1] * -1) + np.pi
        return bboxes_3d
    
    def _flip_bboxes_2d(self, 
                        bboxes_2d: np.ndarray, 
                        img_metas: Dict[str, Any]) -> np.ndarray:
        
        # 'bboxes_2d': (B, K, 4)
        ref_width = img_metas['ori_shape'][1]
        flipped = bboxes_2d.copy()
        
        flipped[..., 0] = (ref_width - bboxes_2d[..., 2])
        flipped[..., 2] = (ref_width - bboxes_2d[..., 0])
        return flipped


class Normalize(BaseTransform):
    def __init__(self, 
                 mean: List[float], 
                 std: List[float],
                 keep_origin: bool = False):
        
        super().__init__(True, False, False, False)
            
        if isinstance(mean, Number):
            mean = [mean,] * 3
        if isinstance(std, Number):
            std = [std,] * 3
            
        self.mean = mean
        self.std = std
        
        self.keep_origin = keep_origin
        
    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        
        img = data_dict['img']                          # (H, W, C) / np.ndarray
        img = img.astype(np.float32)
        
        if self.keep_origin:
            data_dict['ori_img'] = img.copy()
        
        mean = np.array(self.mean).reshape(1, 1, -1)
        std = np.array(self.std).reshape(1, 1, -1)
        
        norm_img = (img - mean) / std
        data_dict['img'] = norm_img
        return data_dict


class Pad(BaseTransform):
    def __init__(self, size_divisor: int):
        super().__init__(True, True, False, False)
        
        self.size_divisor = size_divisor
        
    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        
        # Pad Image
        img = data_dict['img']                          # (H, W, C) / np.ndarray
        ori_h, ori_w = img.shape[:2]
        padded_h = int(np.ceil(ori_h / self.size_divisor)) * self.size_divisor
        padded_w = int(np.ceil(ori_w / self.size_divisor)) * self.size_divisor
        
        canvas = np.zeros((padded_h, padded_w, 3), dtype=img.dtype)
        canvas[:ori_h, :ori_w, :] = img
        data_dict['img'] = canvas
        
        # Update Meta
        img_metas = data_dict['img_metas']
        img_metas['pad_shape'] = (padded_h, padded_w)
        data_dict['img_metas'] = img_metas
        
        return data_dict
        
        
class ToTensor(BaseTransform):
    def __init__(self):
        super().__init__(True, False, False, True)

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        
        # Convert image to tensor
        if 'img' in data_dict.keys():
            try:
                img = torch.Tensor(data_dict['img']).permute(2, 0, 1)
            except:
                img = torch.Tensor(data_dict['img'].copy()).permute(2, 0, 1)
            data_dict['img'] = img
            
        # Convert ground-truth data to tensor
        if 'label' in data_dict.keys():
            label = data_dict['label']
            for k, v in label.items():
                label[k] = torch.Tensor(v).unsqueeze(0)
            data_dict['label'] = label
        return data_dict
    

# Only used in testing for raw dataset
class Convert_3D_to_4D(BaseTransform):
    def __init__(self):
        super().__init__(True, True, False, False)
        
    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        
        # Image Tensor
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                if v.dim() == 3:
                    data_dict[k] = v.unsqueeze(0)
        
        # Image Metas
        for k, v in data_dict['img_metas'].items():
            data_dict['img_metas'][k] = [v,]
            
        
        # Calib
        data_dict['calib'] = [data_dict['calib']]
                    
        return data_dict