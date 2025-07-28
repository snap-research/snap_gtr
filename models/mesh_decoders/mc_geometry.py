from dataclasses import dataclass
from collections import defaultdict
from loguru import logger

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, custom_fwd

from diso import DiffMC

from utils.render_utils import NVDiffRasterizerContext
from utils.config_utils import PrintableConfig
from utils.distributed import get_rank


def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def compute_vertex_normal(v_pos, faces):
    i0 = faces[:, 0]
    i1 = faces[:, 1]
    i2 = faces[:, 2]

    v0 = v_pos[i0, :]
    v1 = v_pos[i1, :]
    v2 = v_pos[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(v_pos)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(
        dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
    )
    v_nrm = F.normalize(v_nrm, dim=1)

    return v_nrm


class Geometry:
    def __init__(self):
        pass

    def forward(self):
        pass


@dataclass
class MCGeometryConfig(PrintableConfig):
    """Marching cube geometry config"""
    grid_res: int = 256
    """Resolution of grid points"""
    mc_level: float = 10.
    """marching cube level used to determine surface"""
    render_res: int = 1024
    """Render resolution"""
    batch_size: int = 2
    """Batch size, used by DiffMC"""


class MCGeometry(Geometry):
    def __init__(
        self,
        device,
        cfg: MCGeometryConfig,
    ):
        super().__init__()
        self.grid_res = cfg.grid_res
        self.mc_level = cfg.mc_level
        self.render_res = cfg.render_res
        self.device = device
        self.batch_size = cfg.batch_size

        # construct grid points
        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, self.grid_res),
            torch.linspace(0, 1, self.grid_res),
            torch.linspace(0, 1, self.grid_res),
        ), -1)
        samples = samples * 2.0 - 1.0  # we use points in range [-1.0, 1.0]
        self.grid_verts = samples.view(-1, 3).to(device)
        self.grid_verts.requires_grad = False
        extractors = [DiffMC(dtype=torch.float32).to(device) for _ in range(self.batch_size)]
        self.extractors = extractors

        # render context
        self.ctx = NVDiffRasterizerContext(device=device)
        logger.debug(f"Rank[{get_rank()}], build MCGeometry: {device}, {self.grid_verts.device} {self.batch_size} mc_level: {self.mc_level}")

    def get_grid_pts(self):
        return self.grid_verts

    @custom_fwd(cast_inputs=torch.float32)
    def get_geometry_from_density(self, density: torch.Tensor, batch_idx: int):
        density = density.view(self.grid_res, self.grid_res, self.grid_res)

        # use density as 
        sdf = -(density - self.mc_level)  # change to use density as sdf

        verts, faces = self.extractors[batch_idx](
            sdf,
            deform=None,
            normalize=True
        )
        verts_norm = (verts * 2 - 1.0)  # range in (-1.0, 1.0)

        return verts_norm, faces

    @custom_fwd(cast_inputs=torch.float32)
    def render_geometry(self, v, f, mvp_mtx, mv_mtx):
        # TODO: when to use antialias
        assert v.ndim == 2 and f.ndim == 2 and mvp_mtx.ndim == 3 and mv_mtx.ndim == 3
        num_views = mvp_mtx.shape[0]
        if len(v) == 0:
            #    raise Exception(f"Empty vertices")
            return {
            'hard_mask': torch.zeros((num_views, self.render_res, self.render_res, 1)).to(self.device),
            'antialias_mask': torch.zeros((num_views, self.render_res, self.render_res, 1)).to(self.device),
            'tex_pts': torch.zeros((num_views, self.render_res, self.render_res, 3)).to(self.device),
            'depth': torch.zeros((num_views, self.render_res, self.render_res, 1)).to(self.device),
            'normal': (torch.ones((num_views, self.render_res, self.render_res, 3)) * torch.tensor([0., 0., 1.0])).to(self.device),
        }

        v_clip = self.ctx.vertex_transform(v, mvp_mtx)  # (num_views, npts, 4)
        rast, rast_db = self.ctx.rasterize(v_clip, f, (self.render_res, self.render_res))  # (num_views, img_h, img_w, 4)

        # mask
        hard_mask = torch.clamp(rast[..., -1:], 0, 1)  # rast is a 4-channel with tuple (u, v, z/w, triangle_id)
        antialias_mask = self.ctx.antialias(hard_mask.clone(), rast, v_clip, f)  # (num_views, img_h, img_w, 1)

        # pixel corresponding points
        tex_pts, _ = self.ctx.interpolate(v, rast, f)  # (num_views, img_h, img_w, 3)

        # depth
        verts_view = self.ctx.vertex_transform(v, mv_mtx)
        out_depth, _ = self.ctx.interpolate(verts_view, rast, f)
        out_depth = out_depth[..., -2:-1]  # (num_views, img_h, img_w, 1)
        out_depth = - out_depth

        # normal
        vn = compute_vertex_normal(v, f)  # vertex_normal in world space
        out_normal, _ = self.ctx.interpolate(vn, rast, f)
        out_normal = self.ctx.antialias(out_normal, rast, v_clip, v)
        out_normal = F.normalize(out_normal, dim=-1)

        res = {
            'hard_mask': hard_mask,
            'antialias_mask': antialias_mask,
            'tex_pts': tex_pts,
            'depth': out_depth,
            'normal': out_normal,
        }
        return res

    @custom_fwd(cast_inputs=torch.float32)
    def batch_render_geometry(self, verts_list, faces_list, mvp_mtx_list, mv_mtx_list):
        outputs_lists = defaultdict(list)
        for verts, faces, mvp_mtx, mv_mtx in zip(verts_list, faces_list, mvp_mtx_list, mv_mtx_list):
            outputs = self.render_geometry(verts, faces, mvp_mtx, mv_mtx)
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.stack(outputs_list, dim=0)
        return outputs

