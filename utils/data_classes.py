import numpy as np
import pandas as pd

from typing import Union, List, Dict, Any

from utils.geometry_ops import points_cam2img, center_to_corner_box3d, view_points


# Calibration Utils
class KITTICalibration:
    def __init__(self, calib_file: Union[Dict[str, Any], str]):
        if isinstance(calib_file, str):
            calib = self._get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P0 = calib['P0']               # 3 x 4
        self.P1 = calib['P1']               # 3 x 4
        self.P2 = calib['P2']               # 3 x 4
        self.P3 = calib['P3']               # 3 x 4
        self.R0 = calib['R0']               # 3 x 3
        
        self.V2C = calib['Tr_velo2cam']     # 3 x 4
        self.C2V = self.inverse_rigid_trans(self.V2C)
        
        self.I2V = calib['Tr_imu2velo']     # 3 x 4
        self.V2I = self.inverse_rigid_trans(self.I2V)

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)
        
    def _get_calib_from_file(self, calib_file: str) -> Dict[str, Any]:
        with open(calib_file) as f:
            lines = f.readlines()

        obj = lines[0].strip().split(' ')[1:]
        P0 = np.array(obj, dtype=np.float32)

        obj = lines[1].strip().split(' ')[1:]
        P1 = np.array(obj, dtype=np.float32)
        
        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)
        
        obj = lines[6].strip().split(' ')[1:]
        Tr_imu_to_cam = np.array(obj, dtype=np.float32)

        return {'P0': P0.reshape(3, 4),
                'P1': P1.reshape(3, 4),
                'P2': P2.reshape(3, 4),
                'P3': P3.reshape(3, 4),
                'R0': R0.reshape(3, 3),
                'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4),
                'Tr_imu2velo': Tr_imu_to_cam.reshape(3, 4),}
        
    def _convert_to_4x4(self, mat: np.ndarray):
        view = np.eye(4)
        
        mat_l, mat_c = mat.shape
        view[:mat_l, :mat_c] = mat
        return view
    
    def get_info_dict(self) -> Dict[str, Any]:
        return {'P0': self._convert_to_4x4(self.P0),
                'P1': self._convert_to_4x4(self.P1),
                'P2': self._convert_to_4x4(self.P2),
                'P3': self._convert_to_4x4(self.P3),
                'R0_rect': self._convert_to_4x4(self.R0),
                'Tr_velo_to_cam': self._convert_to_4x4(self.V2C),
                'Tr_imu_to_velo': self._convert_to_4x4(self.I2V)}
        
    def inverse_rigid_trans(self, Tr):
        inv_Tr = np.zeros_like(Tr)
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr
    
    
    def rescale(self, scale_x: float = None, scale_y: float = None) -> None:
        
        if scale_x is None:
            scale_x = 1.0
        if scale_y is None:
            scale_y = 1.0
        
        # Rescale
        for mat in [self.P0, self.P1, self.P2, self.P3]:
            mat[0, [0, 2, 3]] *= scale_x
            mat[1, [1, 2, 3]] *= scale_y
        
        # Reassign
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)
            


# Util class for single object
class KITTISingleObject:
    def __init__(self, parsed_line: str, calib: KITTICalibration):
        
        self.calib = calib
        self.parsed_line = parsed_line
        label = parsed_line.strip().split(' ')
        
        cls_str_to_idx = {
            'DontCare': -1,
            'Pedestrian': 0,
            'Cyclist': 1,
            'Car': 2}
        
        self.cls_str = label[0]
        self.cls_num = cls_str_to_idx.get(self.cls_str, -1)
        
        self.occlusion = float(label[2])        # 0: Fully Visible, 3: Unknown
        self.truncation = float(label[1])
        
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dim = np.array((self.l, self.h, self.w), dtype=np.float32)
        self.ry = float(label[14])
        self.alpha = float(label[3])
        
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        
        self.level_str = None
        self.level = self.get_obj_level()
        
        self.base_cam = 0
        self.yaw_type = 'global'
        self.center_type = 'bottom-center'
        
    def get_obj_level(self) -> int:
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if self.truncation == -1:
            self.level_str = 'DontCare'
            return 0

        if (height >= 40) and (self.truncation <= 0.15) and (self.occlusion <= 0):
            self.level_str = 'Easy'
            return 1
        elif (height >= 25) and (self.truncation <= 0.3) and (self.occlusion <= 1):
            self.level_str = 'Moderate'
            return 2
        elif (height >= 25) and (self.truncation <= 0.5) and (self.occlusion <= 2):
            self.level_str = 'Hard'
            return 3
        else:
            self.level_str = 'UnKnown'
            return 4
    
    def translate(self, shift_x: float, shift_y: float = 0.0, shift_z: float = 0.0) -> None:
        self.loc += np.array([shift_x, shift_y, shift_z])
        
    def flip(self) -> None:
        self.loc *= np.array((-1, 1, 1), dtype=np.float32)
            
    def convert_yaw(self, src_type: str, dst_type: str) -> None:
        x, _, z = self.loc
        rot_offset = np.arctan2(x, z)
        sign = +1 if (src_type == 'local') else -1
        
        if self.yaw_type != dst_type:
            self.ry = (self.ry + sign * rot_offset)
            self.yaw_type = dst_type
            
    def convert_cam(self, src_cam: int, dst_cam: int) -> None:
        if self.base_cam != dst_cam:
            src_proj = getattr(self.calib, f'P{src_cam}')
            dst_proj = getattr(self.calib, f'P{dst_cam}')
            offset = (dst_proj[0, 3] - src_proj[0, 3]) / dst_proj[0, 0]
            
            self.translate(shift_x=offset)
            self.base_cam = dst_cam
            
    def convert_center(self, src_type: str, dst_type: str) -> None:
        h_offset = (0.5 * self.h)
        sign = -1 if (src_type == 'bottom-center') else +1
        
        if self.center_type != dst_type:
            self.translate(shift_x=0.0, shift_y=(sign * h_offset))
            self.center_type = dst_type
                
    @property
    def projected_center(self) -> np.ndarray:
        cam_flag, center_flag = False, False
        
        if self.base_cam == 2:
            self.convert_cam(src_cam=2, dst_cam=0)
            cam_flag = True
            
        if self.center_type == 'bottom-center':
            self.convert_center(src_type='bottom-center', dst_type='gravity-center')
            center_flag = True
            
        cam0_center = self.loc[np.newaxis, ...]
        proj_center = points_cam2img(cam0_center, self.calib.P2, with_depth=True)[0]
        
        if cam_flag:
            self.convert_cam(src_cam=0, dst_cam=2)
        if center_flag:
            self.convert_center(src_type='gravity-center', dst_type='bottom-center')
        
        return proj_center
    
    @property
    def projected_kpts(self) -> np.ndarray:
        
        proj_center = self.projected_center
        
        cam_flag, center_flag, yaw_flag = False, False, False
        if proj_center[-1] <= 0:
            return None
        
        if self.yaw_type == 'local':
            self.convert_yaw(src_type='local', dst_type='global')
            yaw_flag = True
        
        if self.base_cam == 2:
            self.convert_cam(src_cam=2, dst_cam=0)
            cam_flag = True
            
        if self.center_type == 'bottom-center':
            self.convert_center(src_type='bottom-center', dst_type='gravity-center')
            center_flag = True

            
        # (3, 8)
        corners_3d = center_to_corner_box3d(
            self.loc[np.newaxis, ...],
            self.dim[np.newaxis, ...],
            np.array([self.ry]),
            origin=(0.5, 0.5, 0.5),
            axis=1)[0].T
        original_corners_3d = corners_3d.copy()
        
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        valid_corners_mask = np.zeros((8, 1))
        valid_corners_mask[in_front, :] = 1
        
        all_corner_coords = view_points(original_corners_3d, self.calib.P2, normalize=True).T[:, :2]
        all_corner_coords = np.hstack([all_corner_coords, valid_corners_mask])
        
        proj_center[2] = 1
        projected_pts = np.concatenate([all_corner_coords, proj_center[np.newaxis, ...]])
        
        if center_flag:
            self.convert_center(src_type='gravity-center', dst_type='bottom-center')
        if cam_flag:
            self.convert_cam(src_cam=0, dst_cam=2)
        if yaw_flag:
            self.convert_yaw(src_type='global', dst_type='local')
            
        return projected_pts
    
    
    @property
    def is_ignored(self) -> bool:
        return True if (self.cls_num == -1) else False
     
            
# Util class for multi objects
class KITTIMultiObjects:
    def __init__(self, 
                 obj_list: List[KITTISingleObject],
                 ignore_dontcare: bool = True):
        
        self.ignore_dontcare = ignore_dontcare
        if ignore_dontcare:
            self.ori_obj_list = obj_list
            
            new_obj_list = []
            for obj in obj_list:
                if not obj.is_ignored:
                    new_obj_list.append(obj)
        
        else:
            new_obj_list = obj_list
    
        self.obj_list = new_obj_list
        self.calib = None
        if len(self.obj_list) >= 1:
            self.calib = self.obj_list[0].calib
        
    def __len__(self) -> int:
        return len(self.obj_list)
    
    def __getitem__(self, idx: int) -> KITTISingleObject:
        return self.obj_list[idx]
    
    def __repr__(self) -> str:
        return f"KITTIMultiObjects(Objects: {len(self)})"
    
    def convert_yaw(self, src_type: str, dst_type: str) -> None:
        for obj in self.obj_list:
            obj.convert_yaw(src_type, dst_type)
    
    def convert_cam(self, src_cam: int, dst_cam: int) -> None:
        for obj in self.obj_list:
            obj.convert_cam(src_cam, dst_cam)
            
    def convert_center(self, src_type: str, dst_type: str) -> None:
        for obj in self.obj_list:
            obj.convert_center(src_type, dst_type)
    
    @property
    def original_objects(self):
        if self.ignore_dontcare:
            return KITTIMultiObjects(self.ori_obj_list, ignore_dontcare=False)
        else:
            return self
            
    @property
    def data_frame(self) -> pd.DataFrame:
        key_to_attr = {
            'name': 'cls_str',
            'truncated': 'truncation',
            'occluded': 'occlusion',
            'alpha': 'alpha',
            'bbox': 'box2d',
            'dimensions': 'dim',
            'location': 'loc',
            'rotation_y': 'ry',
            'score': 'score',
        }
        
        obj_dict = {
            k: [getattr(obj, attr) for obj in self.obj_list]
            for k, attr in key_to_attr.items()
        }
        
        df = pd.DataFrame.from_dict(obj_dict)
        return df
    
    @property
    def info_dict(self) -> Dict[str, np.ndarray]:
        df = self.data_frame
        obj_dict = df.to_dict('list')
        
        info_dict = {}
        valid_keys = list(obj_dict.keys())
        
        for valid_key in valid_keys:
            value = obj_dict[valid_key]
            
            stack_flag = False
            if isinstance(value[0], np.ndarray):
                stack_flag = True

            if stack_flag:
                info_dict.update({valid_key: np.stack(value)})
            else:
                info_dict.update({valid_key: np.array(value)})
        return info_dict

    @staticmethod
    def get_objects_from_label(label_file: str, calibration: KITTICalibration):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        objects = [KITTISingleObject(line, calibration) for line in lines]
        return KITTIMultiObjects(objects)
