"""
Example script to convert zero123++ image grid to input format of LRM.

For other MV generation approaches, modify the camera parameters (FOV, radius, elevation, and azimuth) accordingly.
"""
from PIL import Image, ImageOps
import rembg
import click
from loguru import logger
import math
import numpy as np
from pathlib import Path


def pad_and_remove_background(image: Image.Image, border_size: int, rembg_session) -> Image.Image:
    # overflow object need first padding before run segmentation
    padded_image = ImageOps.expand(image, border=border_size, fill="white")

    foreground_image = rembg.remove(padded_image, session=rembg_session)

    width, height = foreground_image.size
    crop_box = (border_size, border_size, width - border_size, height - border_size)
    cropped_image = foreground_image.crop(crop_box)

    return cropped_image


def fov_to_intrinsic(fov_degree, width, height):
    # Convert FOV from degrees to radians
    fov_radian = math.radians(fov_degree)

    # Calculate the focal length
    # Assuming the same focal length for both x and y axes
    f = width / (2 * math.tan(fov_radian / 2))

    # Calculate the center of the image (principal point)
    cx = width / 2
    cy = height / 2

    # Constructing the intrinsic matrix
    intrinsic_matrix = np.array([[f, 0, cx],
                                 [0, f, cy],
                                 [0, 0, 1]])
    return intrinsic_matrix


def get_cam_pose(theta, phi, radius):
    """
    Args:
        theta: angle between up axis
        phi: angle between right axis
    Note: return in world coordinate. Y is up, x is right. +z is forward.
    """
    theta, phi = np.radians(theta), np.radians(phi)
    y = radius * np.cos(theta)
    x = radius * np.sin(theta) * np.cos(phi)
    z = radius * np.sin(theta) * np.sin(phi)
    return np.array([x, y, z])


def get_c2w_opencv(eye, center, up=np.array([0.0, -1.0, 0.0])):
    # eye position in world coordinate
    # center position in world coordinate
    # up direction in world coordinate
    # return transformation from camera to world
    forward = (center - eye)
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    new_up = np.cross(forward, right)
    new_up /= np.linalg.norm(new_up)

    c2w = np.eye(4)
    c2w[:3, :3] = np.column_stack((right, new_up, forward))
    c2w[:3, 3] = eye

    return c2w


def save_camera(out_cam_file, K, pose, size=(1000, 1000)):
    """
    Export camera extrinsic and intrinsic matrix to file
    Args:
        K: 3x3 intrinsic matrix
        pose: world2 camera transformation matrix
        size: image size
    """
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
        height, width = size
        fout.write("intrinsic fx, fy, cx, cy, height, width \n")
        fout.write(f'{K[0][0]:f} {K[1][1]:f} {K[0][2]:f} {K[1][2]:f} {height} {width}')


def read_camera(file: str):
    """Return cam_ext and cam_intrinsic"""
    assert Path(file).is_file(), f"Cannot find {file}"
    f = open(file)
    text = f.readlines()
    cam_ext = np.array([[float(y) for y in x.split()] for x in text[1:5]])
    fx, fy, cx, cy, height, width = np.array([float(y) for y in text[7].split()])
    f.close()
    cam_int = np.eye(3)
    cam_int[0, 0], cam_int[1, 1], cam_int[0, 2], cam_int[1, 2] = fx, fy, cx, cy
    return cam_ext, cam_int, height, width


def prepare_cameras(camera_dir, fov, img_size, theta_list, phi_list, radius_list):
    """"""
    width, height = img_size
    K = fov_to_intrinsic(fov, width, height)

    num_camera = len(phi_list)
    for i in range(num_camera):
        theta, phi, radius = theta_list[i], phi_list[i], radius_list[i]
        cam_pos = get_cam_pose(theta, phi, radius)
        c2w = get_c2w_opencv(cam_pos, np.array([0.0, 0.0, 0.0]))
        pose = np.linalg.inv(c2w)
        out_cam_file = f"{camera_dir}/cam_{i:03d}.txt"
        save_camera(out_cam_file, K, pose, size=img_size)
        # print(cam_pos, out_cam_file)


@click.command()
@click.option("--in_dir", type=str, help="Path to input image file")
@click.option("--out_dir", type=str, help="Path to output directory")
def main(
    in_dir: str,
    out_dir: str,
):
    img_path_to_np01 = lambda img_file: np.array(Image.open(img_file)).astype('float') / 255.
    np01_to_pil = lambda x: Image.fromarray((x * 255.).astype(np.uint8))
    rembg_session = rembg.new_session()

    Path(out_dir).mkdir(exist_ok=True, parents=True)

    img = img_path_to_np01(in_dir)
    img = np.hstack(np.split(img, 3, axis=0))
    img_list = np.array_split(img, 6, axis=1)

    # corresponding camera params for zero123++ output
    theta_list = [70, 100, 70, 100, 70, 100] # 90 - elevation
    phi_list = [60, 0, -60, -120, 180, 120] # 90 - azimuth
    cam_radius = 4
    radius_list = [cam_radius] * len(phi_list)
    fov = 30
    width, height = 512, 512

    for i in range(6):
        img = img_list[i]
        pil_img = np01_to_pil(img)
        border_size = int(pil_img.size[0] * 0.15)
        view = pad_and_remove_background(pil_img, border_size, rembg_session)
        view = view.resize((width, height))

        out_file = f"{out_dir}/rgb_{i:03d}.png"
        view.save(out_file)

    prepare_cameras(out_dir, fov, (width, height), theta_list, phi_list, radius_list)
    logger.info(f"Saved processed data to {out_dir}")


if __name__ == "__main__":
    main()