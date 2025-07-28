from pathlib import Path
import h5py
import os
import numpy as np
import mcubes

import torch
from torch.nn.functional import affine_grid, grid_sample


def numpy_sdf_to_mesh_mcube(sdf: np.array, level=0.02, expand_rate=1.0):
    """
    Args:
        sdf: np.array, (resolution, resolution, resolution)
    Return:
        vertex: np.array, (npts, 3)
        faces: np.array, (nfaces, 3)
    """
    assert len(sdf.shape) == 3
    resolution = sdf.shape[0]
    verts, faces = mcubes.marching_cubes(sdf, level)
    verts_norm = verts / (resolution - 1)  # range in (0 - 1)
    verts_norm = (verts_norm * 2 - 1.0) * expand_rate  # range in (-expand_rate, expand_rate)
    return verts_norm, faces


def align_sdf_pose(sdf: np.array, pose_label: str):
    """Align SDF based on labels
    Assume sdf axis order of ["x", "z", "-y"], xyz are blender canonical coordinate
    """
    label_forward_axis, label_up_axis, label_right_axis = pose_label.split(',')
    map_axis2index = {
        "x": 0, "-x": 0,
        "y": 2, "-y": 2,
        "z": 1, "-z": 1,
    }

    canonical_axis = ["x", "z", "-y"]
    label_right_axis_index = map_axis2index[label_right_axis]
    label_up_axis_index = map_axis2index[label_up_axis]
    label_forward_axis_index = map_axis2index[label_forward_axis]

    sdf = sdf.transpose(label_right_axis_index, label_up_axis_index, label_forward_axis_index)
    flip_right = label_right_axis not in canonical_axis
    flip_up = label_up_axis not in canonical_axis
    flip_forward = label_forward_axis not in canonical_axis
    if flip_right:
        sdf = sdf[::-1, :, :]
    if flip_up:
        sdf = sdf[:, ::-1, :]
    if flip_forward:
        sdf = sdf[:, :, ::-1]
    return sdf


def numpy_downsample_sdf(sdf: np.ndarray, res: int):
    h5_res = int(np.round(np.power(sdf.size, 1 / 3)))
    assert sdf.size == h5_res ** 3
    sdf = sdf.reshape(h5_res, h5_res, h5_res)

    if res != h5_res:
        # downsample SDF by an integer factor
        assert h5_res % res == 0
        res_reduce = h5_res // res
        x_ind = np.arange(0, h5_res, res_reduce).astype(np.int32)
        y_ind = np.arange(0, h5_res, res_reduce).astype(np.int32)
        z_ind = np.arange(0, h5_res, res_reduce).astype(np.int32)
        zv, yv, xv = np.meshgrid(z_ind, y_ind, x_ind, indexing="ij")
        sdf = sdf[zv, yv, xv]
    return sdf.reshape(res, res, res)


def load_h5_file(sdf_h5_file, res, pose_label=None):
    """
    Load sdf h5 file to tensor sdf of shape (res, res ,res)
    """
    assert Path(sdf_h5_file).is_file(), f"Cannot find {sdf_h5_file}"
    h5_f = h5py.File(sdf_h5_file, 'r')
    sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)

    sdf = numpy_downsample_sdf(sdf, res)

    sdf = sdf.transpose(2, 1, 0)  # align with object coordinate

    if pose_label is not None:
        sdf = align_sdf_pose(sdf, pose_label)  # align pose

    return sdf


def save_sdf_to_h5(out_file: str, sdf: np.array):
    """Write sdf to h5 file"""
    assert sdf.ndim == 3
    f1 = h5py.File(out_file, "w")
    f1.create_dataset(
        "pc_sdf_sample",
        data=sdf.astype(np.float32),
        compression="gzip",
        compression_opts=4,
    )
    f1.close()

