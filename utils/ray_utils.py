import numpy as np
import torch
import torch.nn.functional as F

import mcubes
from packaging import version as pver


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def get_ray_directions(h, w, intrinsics, norm=False, device=None):
    """
    Args:
        h (int)
        w (int)
        intrinsics: (*, 4), in [fx, fy, cx, cy]

    Returns:
        directions: (*, h, w, 3), the direction of the rays in camera coordinate
    """
    batch_size = intrinsics.shape[:-1]
    x = torch.linspace(0.5, w - 0.5, w, device=device)
    y = torch.linspace(0.5, h - 0.5, h, device=device)
    # (*, h, w, 2)
    directions_xy = torch.stack(
        [((x - intrinsics[..., 2:3]) / intrinsics[..., 0:1])[..., None, :].expand(*batch_size, h, w),
         ((y - intrinsics[..., 3:4]) / intrinsics[..., 1:2])[..., :, None].expand(*batch_size, h, w)], dim=-1)
    # (*, h, w, 3)
    directions = F.pad(directions_xy, [0, 1], mode='constant', value=1.0)
    if norm:
        directions = F.normalize(directions, dim=-1)
    return directions


def get_rays(directions, c2w, norm=False):
    """
    Args:
        directions: (*, h, w, 3) precomputed ray directions in camera coordinate
        c2w: (*, 3, 4) transformation matrix from camera coordinate to world coordinate
    Returns:
        rays_o: (*, h, w, 3), the origin of the rays in world coordinate
        rays_d: (*, h, w, 3), the normalized direction of the rays in world coordinate
    """
    rays_d = directions @ c2w[..., None, :3, :3].transpose(-1, -2)  # (*, h, w, 3)
    rays_o = c2w[..., None, None, :3, 3].expand(rays_d.shape)  # (*, h, w, 3)
    if norm:
        rays_d = F.normalize(rays_d, dim=-1)
    return rays_o, rays_d


def get_cam_rays(c2w, intrinsics, h, w, norm=True):
    """Pay attention to normalization argument"""
    directions = get_ray_directions(
        h, w, intrinsics, norm=False, device=intrinsics.device)  # (num_scenes, num_imgs, h, w, 3)
    rays_o, rays_d = get_rays(directions, c2w, norm=norm)
    return rays_o, rays_d


def get_ortho_cam_rays(c2w, intrinsics, h, w):
    batch_size = intrinsics.shape[:-1]
    device = intrinsics.device

    x = torch.linspace(0.5, w - 0.5, w, device=device)
    y = torch.linspace(0.5, h - 0.5, h, device=device)
    x = x / w * intrinsics[..., 0:1] - 0.5 * intrinsics[..., 0:1]  # (*, w)
    y = y / h * intrinsics[..., 1:2] - 0.5 * intrinsics[..., 1:2]  # (*, h)
    origins_xy = torch.stack([
        x[..., None, :].expand(*batch_size, h, w),
        y[..., :, None].expand(*batch_size, h, w),
    ], dim=-1)  # (*, h, w, 2)
    origins = F.pad(origins_xy, [0, 1], mode='constant', value=0.0)
    directions = F.pad(torch.zeros_like(origins_xy), [0, 1], mode='constant', value=1.0)

    rays_d = directions @ c2w[..., None, :3, :3].transpose(-1, -2)
    rays_o = (origins @ c2w[..., None, :3, :3].transpose(-1, -2) +
              c2w[..., None, None, :3, 3].expand(rays_d.shape))
    rays_d = F.normalize(rays_d, dim=-1)

    return rays_o, rays_d


def lift_depth_image_to_pts(depth_image, intrinsics, device=None):
    """ Unproject depth image to points
    Args:
        depth_image: (*, h, w)
        intrinsics: (*, 4), in [fx, fy, cx, cy]

    Returns:
        directions: (*, h, w, 3), the un-project points in camera coordinate
    """
    h, w = depth_image.shape[-2], depth_image.shape[-1]
    batch_size = intrinsics.shape[:-1]
    u = torch.linspace(0.5, w - 0.5, w, device=device)[..., None, :].expand(*batch_size, h, w)
    v = torch.linspace(0.5, h - 0.5, h, device=device)[..., :, None].expand(*batch_size, h, w)
    z = depth_image  # (*batch_size, h, w)
    fx = intrinsics[..., 0][..., None, None]
    cx = intrinsics[..., 2][..., None, None]
    fy = intrinsics[..., 1][..., None, None]
    cy = intrinsics[..., 3][..., None, None]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = torch.stack([x, y, z], dim=-1)  # (*, h, w, 3)
    return points


def get_world_pts(pts_cam, c2w):
    """
    Args:
        pts_cam: (*, h, w, 3)
        c2w: (*, 4, 4)
    """
    # (*, h, w, 3)
    xyzs = pts_cam @ c2w[..., None, :3, :3].transpose(-1, -2) + c2w[..., None, None, :3, 3].expand(pts_cam.shape)
    return xyzs

