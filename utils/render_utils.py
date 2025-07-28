"""Create surround views"""
import math
import numpy as np

import torch
import torch.nn.functional as F

import nvdiffrast.torch as dr


def look_at(center, target, up):
    f = F.normalize(target - center, dim=-1)
    s = F.normalize(torch.cross(f, up, dim=-1), dim=-1)
    u = F.normalize(torch.cross(s, f, dim=-1), dim=-1)
    m = torch.stack([s, -u, f], dim=-1)
    return m


def surround_views(initial_pos, angle_amp=1.0, num_frames=60):
    rad = torch.from_numpy(
        np.linspace(0, 2 * np.pi, num=num_frames, endpoint=False, dtype=np.float32))

    initial_pos_dist = torch.linalg.norm(initial_pos)
    initial_pos_norm = initial_pos / initial_pos_dist
    initial_angle = torch.tensor([np.pi / 6.0])

    angles = initial_angle * (rad.sin() * angle_amp + 1)
    pos_xy = F.normalize(initial_pos_norm[:2], dim=0) @ torch.stack(
        [rad.cos(), -rad.sin(),
         rad.sin(), rad.cos()], dim=-1).reshape(-1, 2, 2)
    pos = torch.cat(
        [pos_xy * angles.cos().unsqueeze(-1), angles.sin().unsqueeze(-1)],
        dim=-1) * initial_pos_dist
    pos = torch.stack([pos[:, 0], pos[:, 2], pos[:, 1]], dim=-1).contiguous()
    rot = look_at(pos, torch.zeros_like(pos), pos.new_tensor([0, 1, 0]).expand(pos.size()))
    poses = torch.cat(
        [torch.cat([rot, pos.unsqueeze(-1)], dim=-1),
         rot.new_tensor([0, 0, 0, 1]).expand(num_frames, 1, -1)], dim=-2)

    return poses


def np_fov_to_intrinsic(fov_degree, width, height):
    fov_radian = math.radians(fov_degree)
    f = width / (2 * math.tan(fov_radian / 2))  # Assuming the same focal length for both x and y axes

    cx = width / 2
    cy = height / 2
    intrinsic_matrix = np.array(
        [[f, 0, cx],
         [0, f, cy],
         [0, 0, 1]]
    )
    return intrinsic_matrix


def np_intrinsic_to_fov(K):
    width = K[0, 2] * 2
    fx = K[0, 0]
    fov_radian = math.atan(width / 2 / fx) * 2
    fov_degree = math.degrees(fov_radian)
    return fov_degree


def np_get_projection_matrix(fov, aspect_ratio, near_clip, far_clip):
    fov_rad = np.deg2rad(fov)
    f = 1.0 / np.tan(fov_rad / 2.0)

    # Create the perspective matrix
    perspective_matrix = np.array([
        [f / aspect_ratio, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far_clip + near_clip) / (near_clip - far_clip),
         (2 * far_clip * near_clip) / (near_clip - far_clip)],
        [0.0, 0.0, -1.0, 0.0]
    ])

    return perspective_matrix


def get_projection_matrix(fovy_deg, aspect_ratio, near_clip, far_clip):
    fovy = torch.deg2rad(fovy_deg)

    batch_size = fovy.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
    proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_ratio)
    proj_mtx[:, 1, 1] = 1.0 / torch.tan(fovy / 2.0)
    proj_mtx[:, 2, 2] = (far_clip + near_clip) / (near_clip - far_clip)
    proj_mtx[:, 2, 3] = (2 * far_clip * near_clip) / (near_clip - far_clip)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx


def get_c2w_opencv(eye, center, up=np.array([0.0, -1.0, 0.0])):
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


def c2w_to_mv(c2w):
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    w2c = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    return w2c


def invert_transform(mtx):
    # calculate inv(mtx) from mtx: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (mtx)^-1
    res = torch.zeros(mtx.shape[0], 4, 4).to(mtx)
    res[:, :3, :3] = mtx[:, :3, :3].permute(0, 2, 1)
    res[:, :3, 3:] = -mtx[:, :3, :3].permute(0, 2, 1) @ mtx[:, :3, 3:]
    res[:, 3, 3] = 1.0
    return res


def get_cameras(
    azimuth_deg: torch.tensor,
    elevation_deg: torch.tensor,
    width: int = 512,
    height: int = 512,
    fov: float = 49.1,
    camera_distance: float = 2.0,
):
    azimuth = torch.deg2rad(azimuth_deg)
    elevation = torch.deg2rad(elevation_deg)
    batch_size = len(azimuth)
    camera_distances = torch.ones(batch_size) * camera_distance

    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.sin(elevation),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
        ],
        dim=-1,
    )

    # c2w matrix
    center = torch.zeros_like(camera_positions)
    # default camera up direction as +z
    up = torch.as_tensor([0, 1, 0], dtype=torch.float32)[None, :].repeat(batch_size, 1)
    lookat = F.normalize(center - camera_positions, dim=-1)  # forward
    right = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )  # (batch_size, 3, 4)
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0

    fovy_deg = torch.ones(batch_size) * fov
    proj_mtx = get_projection_matrix(fovy_deg, width/height, 0.1, 1000.0)

    mv = c2w_to_mv(c2w)  # model to view
    mvp_mtx = proj_mtx @ mv  # model to view to projection

    camera = {
        "c2w": c2w,
        "w2c": mv,
        "proj_mtx": proj_mtx,
        "mvp_mtx": mvp_mtx,
        "camera_positions": camera_positions,
        "elevation": elevation_deg,
        "azimuth": azimuth_deg,
        "camera_distances": camera_distances,
        "height": height,
        "width": width,
    }

    return camera


class NVDiffRasterizerContext:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.ctx = dr.RasterizeCudaContext(device=device)

    def vertex_transform(self, verts, mvp_mtx):
        verts_homo = torch.cat(
            [verts, torch.ones([verts.shape[0], 1]).to(verts)], dim=-1
        )
        return torch.matmul(verts_homo, mvp_mtx.permute(0, 2, 1))

    def rasterize(self, pos, tri, resolution):
        return dr.rasterize(self.ctx, pos.float(), tri.int(), resolution, grad_db=True)

    def rasterize_one(self, pos, tri, resolution):
        rast, rast_db = self.rasterize(pos[None, ...], tri, resolution)
        return rast[0], rast_db[0]

    def antialias(self, color, rast, pos, tri):
        return dr.antialias(color.float(), rast, pos.float(), tri.int())

    def interpolate(self, attr, rast, tri, rast_db=None, diff_attrs=None):
        return dr.interpolate(
            attr.float(), rast, tri.int(), rast_db=rast_db, diff_attrs=diff_attrs
        )

    def interpolate_one(self, attr, rast, tri, rast_db=None, diff_attrs=None):
        return self.interpolate(attr[None, ...], rast, tri, rast_db, diff_attrs)


def change_coordinate_system(coordinates, rule: str):
    x, y, z = coordinates.split((1, 1, 1), dim=-1)
    rules = {"x": x, "y": y, "z": z}
    for key in list(rules.keys()):
        rules["-" + key] = -rules[key]
    return torch.cat([rules[r] for r in rule.split(",")], dim=-1)


def change_coordinate_system_numpy(coordinates, rule: str):
    x, y, z = [coordinates[..., i : i + 1] for i in range(3)]
    rules = {"x": x, "y": y, "z": z}
    for key in list(rules.keys()):
        rules["-" + key] = -rules[key]
    return np.concatenate([rules[r] for r in rule.split(",")], axis=-1)

