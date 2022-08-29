import torch
import numpy as np

from typing import Union, Tuple


def extract_corners_from_bboxes_3d(bboxes_3d: torch.Tensor) -> torch.Tensor:
    
    """
                        front z
                            /
                            /
            (x0, y0, z1) + -----------  + (x1, y0, z1)
                        /|            / |
                        / |           /  |
        (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                        |  /      .   |  /
                        | / oriign    | /
        (x0, y1, z0) + ----------- + -------> x right
                        |             (x1, y1, z0)
                        |
                        v
                down y
                
    * Args:
        bboxes_3d (torch.Tensor): (N, 7)
    
    * Returns:
        torch.Tensor with shape of (N, 8, 3)
    """
    
    bboxes_3d = bboxes_3d.detach()
    
    loc = bboxes_3d[:, :3]
    dims = bboxes_3d[:, 3:6]
    rot_y = bboxes_3d[:, 6]
    
    corners_norm = torch.from_numpy(np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1))
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - dims.new_tensor([0.5, 1, 0.5])
    corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

    corners = rotation_3d_in_axis(corners, np.array(rot_y), axis=1, get_as_tensor=True)
    corners = (corners + loc.view(-1, 1, 3))
    return corners


def points_cam2img(points_3d: np.ndarray, 
                   proj_mat: np.ndarray, 
                   with_depth: bool = False,
                   get_as_tensor: bool = False) -> Tuple[torch.Tensor, np.ndarray]:
    
    """
    Project points in camera coordinates to image coordinates.

    Args:
        points_3d (np.ndarray): Points in shape (N, 3)
        proj_mat (np.ndarray): Transformation matrix between coordinates.
        with_depth (bool): Whether to keep depth in the output.

    Returns:
        np.ndarray: Points in image coordinates with shape [N, 2] or [N, 3].
    """
    
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, 'The dimension of the projection'\
        f' matrix should be 2 instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (d1 == 4 and d2 == 4), \
        f'The shape of the projection matrix ({d1} * {d2}) is not supported.'
        
    if isinstance(proj_mat, torch.Tensor):
        proj_mat = proj_mat.detach().numpy()
        
    if d1 == 3:
        proj_mat_expanded = np.eye(4, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    points_4 = np.concatenate([points_3d, np.ones(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    final = point_2d_res

    if with_depth:
        points_2d_depth = np.concatenate([point_2d_res, point_2d[..., 2:3]], axis=-1)
        final = points_2d_depth
        
    if get_as_tensor:
        return torch.from_numpy(final)
    return final


def corners_nd(dims: np.ndarray, origin=0.5) -> np.ndarray:
    
    """
    Generate relative box corners based on length per dim and origin point.

    Args:
        dims (np.ndarray, shape=[N, ndim]): Array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1.
    """
    
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim),
        axis=1).astype(dims.dtype)

    if ndim == 2:
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners


def rotation_3d_in_axis(points: np.ndarray, 
                        angles: np.ndarray, 
                        axis: int = 0,
                        get_as_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
    
    """
    Rotate points in specific axis.

    Args:
        points (np.ndarray, shape=[N, point_size, 3]]):
        angles (np.ndarray, shape=[N]]):
        axis (int): Axis to rotate at.

    Returns:
        np.ndarray: Rotated points.
    """
    
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError('axis should in range')
    
    result = np.einsum('aij,jka->aik', points, rot_mat_T)
    if get_as_tensor:
        return torch.from_numpy(result)
    return result


def center_to_corner_box3d(centers: np.ndarray,
                           dims: np.ndarray,
                           angles: np.ndarray = None,
                           origin=(0.5, 1.0, 0.5),
                           axis: int = 1) -> np.ndarray:
    
    """
    Convert kitti locations, dimensions and angles to corners.

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 3).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 3).
        angles (np.ndarray): Rotation_y in kitti label file with shape (N).
        origin (list or array or float): Origin point relate to smallest point.
            use (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar.
        axis (int): Rotation axis. 1 for camera and 2 for lidar.

    Returns:
        np.ndarray: Corners with the shape of (N, 8, 3).
    """
    
    corners = corners_nd(dims, origin=origin)
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points
