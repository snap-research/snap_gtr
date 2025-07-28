from pathlib import Path
import numpy as np
from PIL import Image
import imageio
from io import BytesIO
import math
from loguru import logger
import json
import tempfile
import cv2
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import torch



def get_client_stream(client, bucket_name, file_path):
    file_bytes = client.get_object(Bucket=bucket_name, Key=file_path)['Body'].read()
    file_stream = BytesIO(file_bytes)
    return file_stream


def read_s3_img(client, bucket_name, file_path):
    try:
        file_stream = get_client_stream(client, bucket_name, file_path)
        img = Image.open(file_stream)
        return img
    except:
        logger.error(f"Fail to load {file_path}")
        raise FileNotFoundError(file_path)


def read_s3_exr(client, bucket_name, file_path):
    try:
        file_stream = get_client_stream(client, bucket_name, file_path)
        data = np.array(imageio.v2.imread(file_stream))
        return data
    except:
        logger.error(f"Fail to load {file_path}")
        raise FileNotFoundError(file_path)


def read_s3_exr2(client, bucket_name, file_path):
    file_stream = client.get_object(Bucket=bucket_name, Key=file_path)['Body'].read()
    with tempfile.NamedTemporaryFile(suffix=".exr", prefix="/dev/shm") as f:
        f.write(file_stream)
        normal_img = cv2.imread(f.name, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
    return normal_img


def read_s3_txt(client, bucket_name, file_path):
    try:
        file_stream = get_client_stream(
            client=client,
            bucket_name=bucket_name,
            file_path=file_path,
        )
        text = file_stream.readlines()
        text = [line.decode("utf-8").strip() for line in text]
        return text
    except:
        logger.error(f"Fail to load {file_path}")
        raise FileNotFoundError(file_path)


def read_s3_json(client, bucket_name, file_path):
    try:
        file_stream = get_client_stream(client, bucket_name, file_path)
        json_str = file_stream.getvalue().decode('utf-8')
        data = json.loads(json_str)
        return data
    except:
        logger.error(f"Fail to load {file_path}")
        raise FileNotFoundError(file_path)


def read_depth(depth_file):
    """
    Objaverse rendering save depth*0.1. Save infinite depth to value 0.
    """
    depth = np.array(imageio.v2.imread(depth_file))[:, :, 0]
    depth[depth == np.max(depth)] = 0
    depth *= 10
    return depth


def get_world2cam_from_blender(bcam2world, use_meshlab_world_coord=True):
    """
    Args:
        bcam2world: transformation from opengl_camera to blender_world coordinate
        use_meshlab_world_coord: convert from blender_world coordinate to meshlab world coordinate
    Returns:
        world2cam: transformation from world to opencv_camera coordinate
    """
    # Blender camera to opencv camera coordinates
    R_bcam2cv = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])

    rotation, location = bcam2world[:3, :3], bcam2world[:3, 3]
    R_world2bcam = rotation.T
    T_world2bcam = - R_world2bcam @ location

    # world to opencv camera
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    if use_meshlab_world_coord:
        # Blender world coordinate uses z-axis as up, while MeshLab uses y-axis as up.
        # Adjust the coordinate transformation matrix.
        world_meshlab2blender = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        R_world2cv = R_world2cv @ world_meshlab2blender

    RT = np.eye(4)
    RT[:3, :3] = R_world2cv
    RT[:3, 3] = T_world2cv

    return RT


def fov_to_intrinsic(fov, width, height, is_fov_degree: bool = False):
    """Convert fov to camera intrinsic matrix"""
    if is_fov_degree:
        fov_radian = math.radians(fov)
    else:
        fov_radian = fov

    # Blender – vertical vFOV or horizontal hFOV – depends on the aspect ratio of the render.
    # reference: https://b3d.interplanety.org/en/vertical-and-horizontal-camera-fov-angles/
    if width >= height:
        f = width / (2 * math.tan(fov_radian / 2))
    else:
        f = height / (2 * math.tan(fov_radian / 2))

    # Calculate the center of the image (principal point)
    cx = width / 2
    cy = height / 2

    intrinsic_matrix = np.array([[f, 0, cx],
                                 [0, f, cy],
                                 [0, 0, 1]])
    return intrinsic_matrix


def load_camera(file: str):
    assert Path(file).is_file(), f"Cannot find {file}"
    f = open(file)
    text = f.readlines()
    cam_ext = np.array([[float(y) for y in x.split()] for x in text[1:5]])
    fx, fy, cx, cy, height, width = np.array([float(y) for y in text[7].split()])
    f.close()
    return cam_ext, fx, fy, cx, cy, height, width


def load_camera_s3(file: str):
    text = file.readlines()
    text = [line.decode("utf-8").strip() for line in text]
    cam_ext = np.array([[float(y) for y in x.split()] for x in text[1:5]])
    fx, fy, cx, cy, height, width = np.array([float(y) for y in text[7].split()])
    return cam_ext, fx, fy, cx, cy, height, width


def load_ortho_camera(file: str):
    assert Path(file).is_file(), f"Cannot find {file}"
    f = open(file)
    text = f.readlines()
    cam_ext = np.array([[float(y) for y in x.split()] for x in text[1:5]])
    ortho_width, ortho_height = np.array([float(y) for y in text[7].split()])
    f.close()
    return cam_ext, ortho_width, ortho_height


def save_camera(out_cam_file, K, pose, size):
    """Export camera extrinsic and intrinsic matrix to file"""
    Path(out_cam_file).parent.mkdir(exist_ok=True, parents=True)
    with open(out_cam_file, "w") as fout:
        # write extrinsic
        fout.write("extrinsic\n")
        for rind in range(3):
            for cind in range(4):
                fout.write(f'{pose[rind][cind]:f} ')
            fout.write('\n')
        fout.write('0 0 0 1\n\n')

        # intrinsic
        width, height = size
        fout.write("intrinsic fx, fy, cx, cy, height, width \n")
        fout.write(f'{K[0][0]:f} {K[1][1]:f} {K[0][2]:f} {K[1][2]:f} {height} {width}')


def simulate_camera_shake(
    extrinsic_matrix: np.array,
    translation_stddev: float = 0.01,
    rotation_stddev_degrees: float = 1.0,
    max_translation: float = 0.05,
    max_rotation: float = 5,
    verbose=False,
):
    """Simulate camera shake in the extrinsic matrix."""
    # Extract the rotation matrix and translation vector from the extrinsic matrix
    rotation_matrix = extrinsic_matrix[:3, :3]
    translation_vector = extrinsic_matrix[:3, 3]

    # Add noise to the translation vector
    translation_noise = np.random.normal(0, translation_stddev, size=(3,))
    translation_noise = np.clip(translation_noise, -max_translation, max_translation)
    if verbose:
        print(f"translation_noise: {translation_noise}")
    noisy_translation_vector = translation_vector + translation_noise

    # Generate random rotation angles (yaw, pitch, and roll) in degrees and add noise
    yaw_degrees = np.random.normal(0, rotation_stddev_degrees)
    pitch_degrees = np.random.normal(0, rotation_stddev_degrees)
    roll_degrees = np.random.normal(0, rotation_stddev_degrees)

    clip_float_degree = lambda x: max(-max_rotation, min(x, max_rotation))
    yaw_degrees = clip_float_degree(yaw_degrees)
    pitch_degrees = clip_float_degree(pitch_degrees)
    roll_degrees = clip_float_degree(roll_degrees)
    if verbose:
        print(f"rotation noise in degree yaw: {yaw_degrees}, pitch: {pitch_degrees}, roll: {roll_degrees}")

    # Convert degrees to radians
    yaw = np.radians(yaw_degrees)
    pitch = np.radians(pitch_degrees)
    roll = np.radians(roll_degrees)

    # Create rotation matrices for the yaw, pitch, and roll
    rotation_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    rotation_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    rotation_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Combine the rotations and add noise
    noisy_rotation_matrix = np.dot(np.dot(rotation_yaw, rotation_pitch), rotation_roll) @ rotation_matrix

    # Create the noisy extrinsic matrix
    noisy_extrinsic_matrix = np.eye(4)
    noisy_extrinsic_matrix[:3, :3] = noisy_rotation_matrix
    noisy_extrinsic_matrix[:3, 3] = noisy_translation_vector

    return noisy_extrinsic_matrix


def read_world_normal(normal_file: str, normalize=False):
    assert normal_file.endswith(".exr")
    normal = np.array(imageio.v2.imread(normal_file))
    normal = np.clip(normal, 0.0, 1.0)
    normal = (normal - 0.5) * 2.0  # [H, W, 3] in range [-1, 1]

    # convert from blender coordinate to meshlab coordinate
    rot_blender2meshlab = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    normal = normal @ rot_blender2meshlab.T

    if normalize:
        normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    return normal


def invert_transform_np(mtx):
    # calculate inv(mtx) from mtx: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (mtx)^-1
    res = np.zeros((4, 4))
    R, t = mtx[:3, :3], mtx[:3, 3]
    res[:3, :3] = R.T
    res[:3, 3] = - R.T @ t
    res[3, 3] = 1.0
    return res


def get_projection_matrix_np(fovx_deg: float, fovy_deg: float, near_clip: float, far_clip: float) -> torch.Tensor:
    fovy = math.radians(fovy_deg)
    fovx = math.radians(fovx_deg)

    proj_mtx = torch.zeros(4, 4, dtype=torch.float32)
    proj_mtx[0, 0] = 1.0 / math.tan(fovx / 2.0)
    # add a negative sign here as y axis is flipped in nvdiffrast output
    proj_mtx[1, 1] = -1.0 / math.tan(fovy / 2.0)
    proj_mtx[2, 2] = (far_clip + near_clip) / (near_clip - far_clip)
    proj_mtx[2, 3] = (2 * far_clip * near_clip) / (near_clip - far_clip)
    proj_mtx[3, 2] = -1.0
    return proj_mtx



