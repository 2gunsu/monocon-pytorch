import os
import sys
import torch
import numpy as np

from typing import List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_classes import KITTICalibration
from utils.geometry_ops import points_cam2img, extract_corners_from_bboxes_3d


CLASSES = ('Pedestrian', 'Cyclist', 'Car')


def get_valid_bboxes_3d(result_3d: Dict[str, torch.Tensor],
                        img_metas: Dict[str, Any],
                        calib: KITTICalibration,
                        batch_idx: int) -> Dict[str, Any]:
    
    bboxes = result_3d['boxes_3d']
    scores = result_3d['scores_3d']
    labels = result_3d['labels_3d']
    
    sample_idx = img_metas['sample_idx'][batch_idx]
    
    if len(bboxes) == 0:
        return dict(
            bbox=np.zeros([0, 4]),
            box3d_camera=np.zeros([0, 7]),
            scores=np.zeros([0]),
            label_preds=np.zeros([0, 4]),
            sample_idx=sample_idx)
        
    P0 = calib.P0.astype(np.float32)
    viewpad = np.eye(4)
    viewpad[:P0.shape[0], :P0.shape[1]] = P0
    P0 = viewpad
    
    V2C = calib.V2C.astype(np.float32)
    viewpad = np.eye(4)
    viewpad[:V2C.shape[0], :V2C.shape[1]] = V2C
    V2C = viewpad
    
    P2 = calib.P2.astype(np.float32)
    img_shape = img_metas['ori_shape'][batch_idx]
    P2 = bboxes.new_tensor(P2)
    
    bboxes_in_camera = bboxes
    
    # Convert Camera to Lidar
    ori_loc = bboxes[:, :3]
    ori_loc = torch.cat([ori_loc, torch.ones(ori_loc.shape[0], 1)], dim=1)
    new_loc = (ori_loc @ torch.Tensor(np.linalg.inv(P0 @ V2C)).t())[:, :-1]
    
    ori_dim = bboxes[:, 3:6]
    new_dim = ori_dim[:, [2, 0, 1]]
    
    rot = bboxes[:, 6]
    
    bboxes_in_lidar = torch.cat([new_loc, new_dim, rot.unsqueeze(1)], dim=-1)
    
    box_corners = extract_corners_from_bboxes_3d(bboxes_in_camera)
    box_corners_in_image = points_cam2img(box_corners, P2, get_as_tensor=True)
    
    min_xy = torch.min(box_corners_in_image, dim=1)[0]
    max_xy = torch.max(box_corners_in_image, dim=1)[0]
    boxes_2d = torch.cat([min_xy, max_xy], dim=1)
    
    image_shape = bboxes.new_tensor(img_shape)
    valid_cam_inds = ((boxes_2d[:, 0] < image_shape[1]) &
                      (boxes_2d[:, 1] < image_shape[0]) &
                      (boxes_2d[:, 2] > 0) &
                      (boxes_2d[:, 3] > 0))
    valid_inds = valid_cam_inds
    
    if valid_inds.sum() > 0:
        return dict(
            bbox=boxes_2d[valid_inds, :].numpy(),
            box3d_camera=bboxes_in_camera[valid_inds].detach().numpy(),
            box3d_lidar=bboxes_in_lidar[valid_inds].detach().numpy(),
            scores=scores[valid_inds].detach().numpy(),
            label_preds=labels[valid_inds].numpy(),
            sample_idx=sample_idx)
        
    else:
        return dict(
            bbox=np.zeros([0, 4]),
            box3d_camera=np.zeros([0, 7]),
            box3d_lidar=np.zeros([0, 7]),
            scores=np.zeros([0]),
            label_preds=np.zeros([0, 4]),
            sample_idx=sample_idx)
    
        

def convert_to_kitti_3d(results_3d: List[Dict[str, torch.Tensor]],
                        img_metas: Dict[str, Any],
                        calibs: List[KITTICalibration]) -> List[Dict[str, Any]]:

    returns = []
    
    if img_metas.get('scale_hw'):
        scale_hw = img_metas['scale_hw'][0]
    else:
        scale_hw = (1., 1.,)  
    scale_hw = np.array([*scale_hw[::-1], *scale_hw[::-1]])
    scale_hw = np.reciprocal(scale_hw)

    for batch_idx, result_3d in enumerate(results_3d):
        
        sample_idx = img_metas['sample_idx'][batch_idx]
        image_shape = img_metas['ori_shape'][batch_idx]     # (H, W)
        calib = calibs[batch_idx]
        
        annos = []
        valid_box_dict = get_valid_bboxes_3d(result_3d, img_metas, calib, batch_idx)
        
        anno = {
            'name': [],
            'truncated': [],
            'occluded': [],
            'alpha': [],
            'bbox': [],
            'dimensions': [],
            'location': [],
            'rotation_y': [],
            'score': []}
        
        if len(valid_box_dict['bbox']) > 0:
            
            box_2d_preds = valid_box_dict['bbox']
            box_preds = valid_box_dict['box3d_camera']
            scores = valid_box_dict['scores']
            label_preds = valid_box_dict['label_preds']

            for box, bbox, score, label in zip(box_preds, box_2d_preds, scores, label_preds):
                
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                
                anno['name'].append(CLASSES[int(label)])
                anno['truncated'].append(0.0)
                anno['occluded'].append(0)
                anno['alpha'].append(-np.arctan2(box[0], box[2]) + box[6])
                anno['bbox'].append(bbox * scale_hw)
                anno['dimensions'].append(box[3:6])
                anno['location'].append(box[:3])
                anno['rotation_y'].append(box[6])
                anno['score'].append(score)

            anno = {k: np.stack(v) for k, v in anno.items()}
            annos.append(anno)

        else:
            anno = {
                'name': np.array([]),
                'truncated': np.array([]),
                'occluded': np.array([]),
                'alpha': np.array([]),
                'bbox': np.zeros([0, 4]),
                'dimensions': np.zeros([0, 3]),
                'location': np.zeros([0, 3]),
                'rotation_y': np.array([]),
                'score': np.array([]),
            }
            annos.append(anno)

        annos[-1]['sample_idx'] = np.array([sample_idx] * len(annos[-1]['score']), dtype=np.int64)
        returns.extend(annos)
    return returns



def convert_to_kitti_2d(results_2d: List[List[np.ndarray]],
                        img_metas: Dict[str, Any]) -> List[Dict[str, Any]]:
    
    # Check Number of Classes
    num_classes = len(results_2d[0])
    assert num_classes == len(CLASSES)
    
    if img_metas.get('scale_hw'):
        scale_hw = img_metas['scale_hw'][0]
    else:
        scale_hw = (1., 1.,)  
    scale_hw = np.array([*scale_hw[::-1], *scale_hw[::-1]])
    scale_hw = np.reciprocal(scale_hw)
    
    returns = []
    
    # 'result_2d' 
    #   --> [Class0 Bbox, Class1 Bbox, Class2 Bbox, ...]
    #   --> Shape of each bbox is (# Obj, 5)
    for batch_idx, result_2d in enumerate(results_2d):
        
        sample_idx = img_metas['sample_idx'][batch_idx]
        num_objs = sum([box.shape[0] for box in result_2d])
        
        annos = []
        anno = {
            'name': [],
            'truncated': [],
            'occluded': [],
            'alpha': [],
            'bbox': [],
            'dimensions': [],
            'location': [],
            'rotation_y': [],
            'score': []}
        
        if (num_objs == 0):
            annos.append(
                dict(
                    name=np.array([]),
                    truncated=np.array([]),
                    occluded=np.array([]),
                    alpha=np.array([]),
                    bbox=np.zeros([0, 4]),
                    dimensions=np.zeros([0, 3]),
                    location=np.zeros([0, 3]),
                    rotation_y=np.array([]),
                    score=np.array([]),
                ))
        
        else:
            for class_idx in range(len(result_2d)):
                class_bbox = result_2d[class_idx]
                
                for box_idx in range(class_bbox.shape[0]):
                    anno['name'].append(CLASSES[class_idx])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(-10)
                    anno['bbox'].append(class_bbox[box_idx, :4] * scale_hw)
                    anno['dimensions'].append(
                        np.zeros(shape=[3], dtype=np.float32))
                    anno['location'].append(
                        np.ones(shape=[3], dtype=np.float32) * (-1000.0))
                    anno['rotation_y'].append(0.0)
                    anno['score'].append(class_bbox[box_idx, 4])
                
            anno = {k: np.stack(v) for k, v in anno.items()}
            annos.append(anno)
        
        annos[-1]['sample_idx'] = np.array(
            [sample_idx] * num_objs, dtype=np.int64)
        returns.extend(annos)
        
    return returns
                