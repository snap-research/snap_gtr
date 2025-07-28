"""
utility functions to get transform shapes into canonical coordiantes
"""
import numpy as np


def convert_canoincal_pose_label_to_rot_mat(pose_label: str, delimiter="_"):
    """
    Return rotation matrix from canonical coordinate to model coordinate.

    While pose label is labeled in blender world coordinate, the main logic is written in meshlab world coordinate.
    Blender:
    - right: x
    - up: z
    - forward: -y
    Meshlab:
    - right: x
    - up: y
    - forward: z
    """
    # label axis in blender world coordinates
    label_forward_axis, label_up_axis, label_right_axis = pose_label.split(delimiter)

    # convert blender axis to meshlab axis
    # blender z axis corresponds to meshlab y axis ([0, 1, 0])
    map_axis2index = {
        "x": 0, "-x": 0,
        "y": 2, "-y": 2,
        "z": 1, "-z": 1,
    }
    label_right_axis_index = map_axis2index[label_right_axis]
    label_up_axis_index = map_axis2index[label_up_axis]
    label_forward_axis_index = map_axis2index[label_forward_axis]

    # conversion from canonical coordinate to model coordinate
    rot_can2model = np.eye(3)
    rot_can2model = np.stack([
        rot_can2model[:, label_right_axis_index],
        rot_can2model[:, label_up_axis_index],
        rot_can2model[:, label_forward_axis_index],
    ], axis=1)

    # check if we need flip axis
    canonical_axis = ["x", "z", "-y"]
    flip_right = label_right_axis not in canonical_axis
    flip_up = label_up_axis not in canonical_axis
    flip_forward = label_forward_axis not in canonical_axis
    if flip_right:
        rot_can2model = rot_can2model * np.array([-1, 1, 1]).reshape(1, 3)
    if flip_up:
        rot_can2model = rot_can2model * np.array([1, -1, 1]).reshape(1, 3)
    if flip_forward:
        rot_can2model = rot_can2model * np.array([1, 1, -1]).reshape(1, 3)
    return rot_can2model


def align_sdf_pose(sdf: np.array, pose_label: str, delimiter="_"):
    """Align SDF based on labels
    Assume sdf axis order of ["x", "z", "-y"], xyz are blender canonical coordinate
    """
    label_forward_axis, label_up_axis, label_right_axis = pose_label.split(delimiter)
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