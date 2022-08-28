import os
import sys
import torch
import numpy as np

from typing import Tuple, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.tensor_ops import gaussian_radius, generate_gaussian_target


# Constants
PI = np.pi


# Target Generator
class TargetGenerator:
    def __init__(self, 
                 num_classes: int = 3,
                 max_objs: int = 30, 
                 num_kpt: int = 9,
                 num_alpha_bins: int = 12):
        
        self.num_classes = num_classes
        self.max_objs = max_objs
        self.num_kpt = num_kpt
        self.num_alpha_bins = num_alpha_bins
        

    def __call__(self, 
                 input_dict: Dict[str, Any],
                 feat_shape: Tuple[int]) -> Dict[str, torch.Tensor]:
    
        device = input_dict['img'].device
        
        metas = input_dict['img_metas']
        label = input_dict['label']
        
        ori_h, ori_w = metas['pad_shape'][0]
        batch_size, _, feat_h, feat_w = feat_shape
        h_ratio, w_ratio = (feat_h / ori_h), (feat_w / ori_w)

        target = self._create_empty_target(feat_shape=feat_shape, device=device)

        for b_idx in range(batch_size):
            
            # Mask
            mask = label['mask'][b_idx].type(torch.BoolTensor)
            
            # Valid 2D Bboxes
            bboxes = label['gt_bboxes'][b_idx][mask]
            bbox_labels = label['gt_labels'][b_idx][mask].type(torch.LongTensor)
            if len(bboxes) < 1:
                continue
            
            bbox_ctx = (bboxes[:, 0] + bboxes[:, 2]) * w_ratio / 2.
            bbox_cty = (bboxes[:, 1] + bboxes[:, 3]) * h_ratio / 2.
            bbox_ct = torch.cat([bbox_ctx.unsqueeze(1), bbox_cty.unsqueeze(1)], dim=1)
            
            # Valid 2D Keypoints
            kpts_2d = label['gt_kpts_2d'][b_idx][mask]
            kpts_2d = kpts_2d.reshape(-1, self.num_kpt, 2)
            
            kpts_2d[:, :, 0] = (kpts_2d[:, :, 0] * w_ratio)
            kpts_2d[:, :, 1] = (kpts_2d[:, :, 1] * h_ratio)
            kpts_mask = label['gt_kpts_valid_mask'][b_idx][mask]
            
            # Valid 3D Bboxes and Depth
            bboxes_3d = label['gt_bboxes_3d'][b_idx][mask]
            depth = label['depths'][b_idx][mask]
            
            for o_idx, ct in enumerate(bbox_ct):
                
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                
                feat_box_h = (bboxes[o_idx, 3] - bboxes[o_idx, 1]) * h_ratio
                feat_box_w = (bboxes[o_idx, 2] - bboxes[o_idx, 0]) * w_ratio
                
                dim = bboxes_3d[o_idx][3:6]
                alpha = bboxes_3d[o_idx][6]
                
                kpt_2d = kpts_2d[o_idx]
                kpt_mask = kpts_mask[o_idx]
                
                target_radius = gaussian_radius((feat_box_h, feat_box_w))
                target_radius = max(0, int(target_radius))
                c_idx = bbox_labels[o_idx]
                
                generate_gaussian_target(target['center_heatmap_target'][b_idx, c_idx],
                                         center=[ctx_int, cty_int],
                                         radius=target_radius)
                
                target['indices'][b_idx, o_idx] = (cty_int * feat_w + ctx_int)
                
                target['wh_target'][b_idx, o_idx] = torch.Tensor([feat_box_w, feat_box_h])
                target['offset_target'][b_idx, o_idx] = torch.Tensor([(ctx - ctx_int), (cty - cty_int)])
                
                target['dim_target'][b_idx, o_idx] = dim
                target['depth_target'][b_idx, o_idx] = depth[o_idx]
                
                alpha_cls, alpha_offset = self._convert_angle_to_class(alpha)
                target['alpha_cls_target'][b_idx, o_idx] = alpha_cls
                target['alpha_offset_target'][b_idx, o_idx] = alpha_offset
                
                target['mask_target'][b_idx, o_idx] = 1
                
                # Keypoints
                for k_idx in range(self.num_kpt):
                    kpt = kpt_2d[k_idx]
                    kptx_int, kpty_int = kpt.int()
                    kptx, kpty = kpt
                    
                    vis_level = kpt_mask[k_idx]
                    if vis_level < 1:
                        continue
                    
                    target['center2kpt_offset_target'][b_idx, o_idx, (k_idx * 2)] = (kptx - ctx_int)
                    target['center2kpt_offset_target'][b_idx, o_idx, (k_idx * 2) + 1] = (kpty - cty_int)
                    
                    target['mask_center2kpt_offset'][b_idx, o_idx, (k_idx * 2): ((k_idx + 1) * 2)] = 1
                    is_kpt_inside_feat = (0 <= kptx_int < feat_w) and (0 <= kpty_int < feat_h)
                    if not is_kpt_inside_feat:
                        continue
                    
                    generate_gaussian_target(target['kpt_heatmap_target'][b_idx, k_idx],
                                             center=[kptx_int, kpty_int],
                                             radius=target_radius)
                    
                    target['indices_kpt'][b_idx, o_idx, k_idx] = (kpty_int * feat_w + kptx_int)

                    target['kpt_heatmap_offset_target'][b_idx, o_idx, (k_idx * 2)] = (kptx - kptx_int)
                    target['kpt_heatmap_offset_target'][b_idx, o_idx, (k_idx * 2) + 1] = (kpty - kpty_int)
                    target['mask_kpt_heatmap_offset'][b_idx, o_idx, (k_idx * 2): ((k_idx + 1) * 2)] = 1
                    
        target['indices_kpt'] = (target['indices_kpt']).reshape(batch_size, -1)
        target['mask_target'] = (target['mask_target']).type(torch.BoolTensor)
        return target
    
    
    def _convert_angle_to_class(self, angle: float):
        angle = angle % (2 * PI)
        assert (angle >= 0 and angle <= 2 * PI)
        
        angle_per_class = 2 * PI / float(self.num_alpha_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * PI)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle
            
        
    def _create_empty_target(self, feat_shape: Tuple[int], device: str = None) -> Dict[str, torch.Tensor]:
        batch_size, _, feat_h, feat_w = feat_shape
        container = {
            'center_heatmap_target': torch.zeros((batch_size, self.num_classes, feat_h, feat_w)),
            'wh_target': torch.zeros((batch_size, self.max_objs, 2)),
            'offset_target': torch.zeros((batch_size, self.max_objs, 2)),
            'dim_target': torch.zeros((batch_size, self.max_objs, 3)),
            'alpha_cls_target': torch.zeros((batch_size, self.max_objs, 1)),
            'alpha_offset_target': torch.zeros((batch_size, self.max_objs, 1)),
            'depth_target': torch.zeros((batch_size, self.max_objs, 1)),
            'center2kpt_offset_target': torch.zeros((batch_size, self.max_objs, self.num_kpt * 2)),
            'kpt_heatmap_target': torch.zeros((batch_size, self.num_kpt, feat_h, feat_w)),
            'kpt_heatmap_offset_target': torch.zeros((batch_size, self.max_objs, self.num_kpt * 2)),
            
            'indices': torch.zeros((batch_size, self.max_objs)).type(torch.LongTensor),
            'indices_kpt': torch.zeros((batch_size, self.max_objs, self.num_kpt)).type(torch.LongTensor),
            
            'mask_target': torch.zeros((batch_size, self.max_objs)),
            'mask_center2kpt_offset': torch.zeros((batch_size, self.max_objs, self.num_kpt * 2)),
            'mask_kpt_heatmap_offset': torch.zeros((batch_size, self.max_objs, self.num_kpt * 2))}
        
        if device is None:
            device = 'cpu'
        for k in container.keys():
            container[k] = container[k].to(device)
        return container
