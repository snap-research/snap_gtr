import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import click
from loguru import logger
from PIL import Image
import numpy as np
import nvdiffrast.torch as dr
import math
import glob

import torch
from torch.amp.grad_scaler import GradScaler

from utils.io_utils import read_yaml, EasyDict
from utils.deterministic import seed_everything
from utils.render_utils import get_cameras, NVDiffRasterizerContext, np_fov_to_intrinsic, invert_transform
from utils.gif_utils import save_images_to_gif
from utils.mesh_utils import load_mesh
from engine.optimizers import Optimizers
from builders.build_system import build_system
from data.dataset_utils import (
    load_camera, get_projection_matrix_np, invert_transform_np
)


def render_mesh_to_gif(
    gif_file: str,
    mesh_file: str,
    cameras: dict,
    ctx: NVDiffRasterizerContext,
    device: torch.device,
    input_image: Image = None,
):
    assert Path(mesh_file).is_file()
    mesh = load_mesh(mesh_file)

    vertices = np.array(mesh.vertices)
    triangles = np.array(mesh.faces)
    vertex_colors = np.array(mesh.visual.vertex_colors)
    vertex_normals = np.array(mesh.vertex_normals)
    v = torch.from_numpy(vertices.astype(np.float32)).contiguous().cuda()
    f = torch.from_numpy(triangles.astype(np.int32)).contiguous().cuda()
    vc = torch.from_numpy(vertex_colors.astype(np.float32)).contiguous().cuda() / 255.0
    vn = torch.from_numpy(vertex_normals.astype(np.float32)).contiguous().cuda()

    mvp_mtx = cameras["mvp_mtx"]
    height = cameras["height"]
    width = cameras["width"]

    v_clip = ctx.vertex_transform(v, mvp_mtx.to(device, non_blocking=True))
    rast, rast_db = ctx.rasterize(v_clip, f, (height, width))
    out, _ = ctx.interpolate(vc, rast, f)

    rgbs = out.cpu().numpy()[:, ::-1, :, :]
    alpha = rgbs[..., 3:]
    rgbs = rgbs[..., :3] * alpha + np.ones(3) * (1 - alpha)
    rgbs = (rgbs * 255.0).astype(np.uint8)

    # normal
    world2cam = cameras["w2c"].to(device, non_blocking=True)
    vn_cam = torch.matmul(vn, world2cam[:, :3, :3].permute(0, 2, 1))
    out, _ = dr.interpolate(vn_cam, rast, f)
    normal_img = out.cpu().numpy()[:, ::-1, :, :]
    normal_img = (normal_img + 1.0) / 2.0
    normal_img = normal_img[..., :3] * alpha + np.ones(3) * (1 - alpha)
    normal_img = normal_img.clip(0, 1.0)
    normal_img = (normal_img * 255.0).astype(np.uint8)

    if input_image is not None:
        cond_grid = np.array(input_image)
        expand_cond_grid = np.tile(cond_grid[None, ...], (len(rgbs), 1, 1, 1))
        imgs = np.concatenate([rgbs, normal_img, expand_cond_grid], axis=1)
    else:
        imgs = np.concatenate([rgbs, normal_img], axis=1)

    save_images_to_gif(imgs, output_file=gif_file)


def load_eval_data(in_dir, input_img_res=512, radius=1.0):
    image_paths = sorted(glob.glob(f"{in_dir}/*.png"))
    pil_to_np01 = lambda x: np.array(x).astype(np.float32) / 255.
    convert_img2camera_path = lambda x: x.replace('rgb_', 'cam_').replace('.png', '.txt')

    imgs_list = []
    masks_list = []
    poses_list = []
    intrinsics_list = []
    img_paths_list = []
    projection_list = []  # used by nvdiffrast
    m2v_list = []  # used by nvdiffrast
    for image_path in image_paths:
        camera_path = convert_img2camera_path(image_path)
        assert Path(camera_path).is_file, f"Cannot find {camera_path}"

        cam_ext, fx, fy, cx, cy, height, width = load_camera(camera_path)
        downsample_factor = input_img_res / height
        downsample_lambda = lambda x: x * downsample_factor
        fx, fy, cx, cy, new_height, new_width = list(map(downsample_lambda, [fx, fy, cx, cy, height, width]))
        new_height, new_width = int(new_height), int(new_width)

        intrinsics_single = torch.FloatTensor([fx, fy, cx, cy])
        intrinsics_list.append(intrinsics_single)

        fovx_deg = math.degrees(math.atan(width / 2 / fx) * 2)
        fovy_deg = math.degrees(math.atan(height / 2 / fy) * 2)
        proj_mtx = get_projection_matrix_np(fovx_deg, fovy_deg, near_clip=0.1, far_clip=100.0)
        projection_list.append(proj_mtx)
        
        c2w = np.linalg.inv(cam_ext)
        c2w[:3, 3] *= 1 / radius
        opengl_c2w = np.copy(c2w)
        opengl_c2w[:3, 1:3] *= -1  # opengl_cam -> world
        c2w = torch.FloatTensor(c2w)
        poses_list.append(c2w)  # (4, 4)
        m2v = torch.FloatTensor(invert_transform_np(opengl_c2w))
        m2v_list.append(m2v)

        img_paths_list.append(image_path)
        pil_img = Image.open(image_path)
        img = pil_to_np01(pil_img)
        rgb, alpha = img[:, :, :3], img[:, :, 3]
        bg_color = np.ones(3).astype(np.float32)
        aug_img = rgb * alpha[:, :, None] + bg_color * (1.0 - alpha[:, :, None])
        new_img = Image.fromarray((aug_img * 255.).astype(np.uint8)).resize((input_img_res, input_img_res))
        img = pil_to_np01(new_img)
        new_mask = Image.fromarray((alpha * 255.).astype(np.uint8)).resize((input_img_res, input_img_res))
        new_mask = pil_to_np01(new_mask)

        img = torch.from_numpy(img)
        imgs_list.append(img)
        new_mask = torch.from_numpy(new_mask)[:, :, None]
        masks_list.append(new_mask)

    cond_poses = torch.stack(poses_list, dim=0)  # (n, 4, 4)
    cond_intrinsics = torch.stack(intrinsics_list, dim=0)  # (n, 4)
    cond_imgs = torch.stack(imgs_list, dim=0)  # (n, h, w, 3)
    cond_masks = torch.stack(masks_list, dim=0)  # (n, h, w, 1)
    cond_m2vs = torch.stack(m2v_list, dim=0)  # (n, 4, 4)
    cond_projections = torch.stack(projection_list, dim=0)  # (n, 4, 4)

    cond_mvps = cond_projections @ cond_m2vs
    eval_data = {
        'cond_imgs': cond_imgs,
        'cond_poses': cond_poses,
        'cond_intrinsics': cond_intrinsics,
        'cond_masks': cond_masks,
        'cond_m2vs': cond_m2vs,
        'cond_projections': cond_projections,
        'cond_mvps': cond_mvps,
    }
    return eval_data


@click.command()
@click.option("--ckpt_path", type=str, help="Path to checkpoint")
@click.option("--in_dir", type=str, help="Path to multiview images and camera files")
@click.option("--out_dir", type=str, default="", help="Path to exported mesh and visuals")
@click.option("--seed", type=int, default=2025, help="random seed")
@click.option("--render_mesh_res", type=int, default=512, help="resolution of mesh renering")
@click.option("--render_nerf_res", type=int, default=1024, help="resolution of NeRF renering")
def main(**kwargs):
    opt = EasyDict(kwargs)
    config = "configs/config_texrefine.yaml"
    job_num = 0
    ckpt_path = opt.ckpt_path
    out_dir = opt.out_dir
    in_dir = opt.in_dir
    seed = opt.seed
    render_mesh_res = opt.render_mesh_res
    render_nerf_res = opt.render_nerf_res
    print(f"---------------options-----------------")
    for key, value in kwargs.items():
        print(f"{key}: {value}")

    seed_everything(seed)
    device = torch.device("cuda:0")

    job_description = read_yaml(config)
    config = job_description["jobs"][job_num]
    config = EasyDict(config)

    assert Path(ckpt_path).is_file(), f"Wrong ckpt path {ckpt_path}"
    state_dict = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    # build pipeline
    logger.debug(f"Building pipeline")
    model = build_system(config, device=device, world_size=1)
    model.load_state_dict(state_dict['pipeline'], strict=True)
    model.switch_eval()

    # load data
    data_batch = load_eval_data(in_dir)

    # set up cameras for mesh rendering
    num_frames, fov_deg, cam_distance = 50, 50, 3.5
    azimuth_deg = torch.from_numpy(np.linspace(0, 360, num=num_frames, endpoint=False, dtype=np.float32))
    elevation_deg = ([20] * num_frames)
    elevation_deg = torch.tensor(elevation_deg).float()

    cameras_render = get_cameras(
        azimuth_deg,
        elevation_deg,
        width=render_mesh_res,
        height=render_mesh_res,
        fov=fov_deg,
        camera_distance=cam_distance,
    )

    # set up cameras for NeRF rendering
    K = np_fov_to_intrinsic(fov_deg, render_nerf_res, render_nerf_res)
    camera_intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
    camera_intrinsics = torch.from_numpy(camera_intrinsics).to(device).unsqueeze(0).expand(num_frames, -1)

    trans_cv2blender = torch.eye(4)
    trans_cv2blender[1, 1] = -1
    trans_cv2blender[2, 2] = -1
    trans_blender2cv = trans_cv2blender.T
    w2blender_cam = cameras_render['w2c']
    w2cv_cam = trans_blender2cv[None] @ w2blender_cam
    camera_poses = invert_transform(w2cv_cam)
    # camera_poses[:, :3, 3] *= 1 / 1.3

    data_batch["poses"] = camera_poses
    data_batch["intrinsics"] = camera_intrinsics
    data_batch["imgs"] = torch.zeros((num_frames, render_nerf_res, render_nerf_res, 3), dtype=torch.float32)
    data_batch["depths"] = torch.zeros((num_frames, render_nerf_res, render_nerf_res, 1), dtype=torch.float32)
    data_batch["mvps"] = (cameras_render['mvp_mtx'])
    data_batch["m2vs"] = (cameras_render['w2c'])

    # inference
    dr_ctx = NVDiffRasterizerContext(device=device)

    param_groups = model.get_param_groups()
    optimizers = Optimizers(config["optimizers"], param_groups)

    for key, value in data_batch.items():
        if type(value) is torch.Tensor:
            data_batch[key] = value.unsqueeze(0).cuda(device, non_blocking=True)

    Path(out_dir).mkdir(exist_ok=True, parents=True)
    logger.info(f'Save to {out_dir}')

    # refine texture
    grad_scaler = GradScaler(enabled=True, init_scale=2048)
    grad_scaler.load_state_dict(state_dict["scalers"])
    model.load_state_dict(state_dict['pipeline'], strict=True)
    
    logger.info(f"Refine texture")
    code, loss = model.refine_texture(data_batch, optimizers, grad_scaler, iters=50, learn_code=True)
    logger.info(f"Refine texture done, Final loss: {loss}")

    # NeRF rendering
    outputs = model.inference_with_code(data_batch, code)[0]

    rgbs = outputs['rgb'][0]
    rgbs = (rgbs.cpu().numpy() * 255.0).astype(np.uint8)[:, ::-1, :, :]

    nerf_rgb_gif_file = f"{out_dir}/nerf.gif"
    save_images_to_gif(rgbs, output_file=nerf_rgb_gif_file)
    logger.info(f"save NeRF rgb rendering to {out_dir}")

    # mesh extraction
    mesh_file = f"{out_dir}/mesh.obj"
    logger.info(f"Provide code to extract mesh")
    mesh_list = model.extract_geometry(data_batch, resolution=512, level=10, code=code)
    logger.info(f"Extract mesh")
    mesh_list[0].export(mesh_file)         

    # render mesh
    mesh_gif_file = f"{out_dir}/mesh.gif"
    render_mesh_to_gif(
        mesh_gif_file,
        mesh_file,
        cameras_render,
        dr_ctx,
        device=device,
        # input_image=cond_grid,
    )
    logger.info(f"Save to {mesh_gif_file}")


if __name__ == "__main__":
    main()
