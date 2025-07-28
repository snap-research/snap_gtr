"""
Finetune triplane with mesh diff-rasterization.
Triplane -> density -> diff_marching_cube -> mesh
mesh -> diff_rasterize -> pixel_xyzs, depth, normal, mask
pixel_xyzs -> sample Triplane -> decode color
"""
import uuid
from collections import OrderedDict
from loguru import logger
import numpy as np
import trimesh
import functools

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, custom_fwd
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure

from utils.distributed import get_rank
from utils.vis_utils import make_grid
from builders.build_model import build_model

from utils.ray_utils import get_cam_rays, get_ortho_cam_rays
from models.mesh_decoders.mc_geometry import MCGeometry
from engine.optim_utils import configure_optimizers
from utils.typing import cast

from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
class MeshGen:
    def __init__(
        self,
        cfg,
        device: str,
        world_size: int = 1,
    ):
        logger.debug(f"MeshGen rank: {get_rank()} device: {device}")
        self.cfg = cfg
        self.distributed = world_size > 1
        self.device = device
        self.cond_mode = cfg.get("cond_mode", "rgb")
        self.tex_pts_detach = cfg.get("tex_pts_detach", False)
        self.gradient_checkpointing = cfg.get("gradient_checkpointing", False)
        self.num_pts_per_chunk = cfg.get("num_pts_per_chunk", 8388608)

        self.generator = build_model(cfg["models"]["generator"], device=self.device)
        self.generator.to(self.device)

        self.field = build_model(cfg["models"]["fields"], device=self.device)
        self.field.to(self.device)

        # Mesh decoder
        # TODO: do we want to refactor mesh render outside mesh_decoder
        cfg["models"]["mesh_decoder"]["args"]["batch_size"] = cfg["batch_size"]
        self.mesh_decoder = build_model(cfg["models"]["mesh_decoder"], device=self.device)

        logger.debug(f"MeshGen rank[{get_rank()}]: build mesh loss_module")
        self.mesh_loss = build_model(cfg["models"]["mesh_loss"], device=self.device)
        self.mesh_loss.to(self.device)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = structural_similarity_index_measure

        # TODO: Do we need to make loss_module distributed?
        if self.distributed:
            self.make_distributed(device=self.device)
            self.field_module = self.field.module
            self.generator_module = self.generator.module
        else:
            self.field_module = self.field
            self.generator_module = self.generator

    def get_param_groups(self, only_trainable=True, verbose=False):
        """One optimizer for whole model"""
        param_groups_list = []
        for model in [self.field, self.generator]:
            model_param_groups = configure_optimizers(
                model, only_trainable=only_trainable, verbose=verbose)
            param_groups_list += model_param_groups
        return {"model": param_groups_list}

    def make_distributed(self, device):
        logger.debug(f"Rank[{get_rank()}], device: {device}")
        self.field = nn.parallel.DistributedDataParallel(
            self.field,
            device_ids=[device],
            output_device=device,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        self.generator = nn.parallel.DistributedDataParallel(
            self.generator,
            device_ids=[device],
            output_device=device,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    def switch_train(self):
        self.field.train()
        self.generator.train()

    def switch_eval(self):
        self.field.eval()
        self.generator.eval()

    def state_dict(self) -> dict:
        """Return a dict containing a whole current state of the training """
        state = {
            "field": self.field_module.state_dict(),
            "generator": self.generator_module.state_dict(),
        }
        return state

    def load_state_dict(
        self,
        state: dict,
        strict: bool = True,
    ):
        logger.info(f"Rank[{get_rank()}] Resume pipeline from state")
        self.field_module.load_state_dict(state['field'], strict=strict)
        self.generator_module.load_state_dict(state['generator'], strict=strict)

    def train_iteration(
        self,
        data_batch,
        optimizers,
        grad_scaler,
        gradient_accumulation_steps,
        mixed_precision: str,
        chunk_size=1024,
    ):
        optimizers.zero_grad_all()
        assert mixed_precision in ["no", "fp16", "bf16"]
        enable_amp = mixed_precision != "no"
        amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16

        assert gradient_accumulation_steps > 0, f"Wrong gradient accumulation steps: {gradient_accumulation_steps}"
        for _ in range(gradient_accumulation_steps):
            with autocast(enabled=enable_amp, dtype=amp_dtype):
                loss_dict = self.get_train_loss_dict(data_batch)
                loss = functools.reduce(torch.add, loss_dict.values())
                loss /= gradient_accumulation_steps
            grad_scaler.scale(loss).backward()

        optimizers.optimizer_scaler_step_all(grad_scaler)

        scale = grad_scaler.get_scale()
        grad_scaler.update()
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= grad_scaler.get_scale():
            optimizers.scheduler_step_all()

        return loss, loss_dict

    def _gen_triplane(self, data_batch):
        cond_imgs = data_batch["cond_imgs"]  # (num_scenes, num_imgs, h, w, 3)
        cond_intrinsics = data_batch["cond_intrinsics"]  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
        cond_c2w = data_batch["cond_poses"]  # (num_scenes, num_imgs, 4, 4)
        num_scenes, num_cond_imgs, cond_h, cond_w, _ = cond_imgs.shape

        if cond_intrinsics.shape[-1] == 2:
            rays_o, rays_d = get_ortho_cam_rays(cond_c2w, cond_intrinsics, cond_h, cond_w)
        elif cond_intrinsics.shape[-1] == 4:
            rays_o, rays_d = get_cam_rays(cond_c2w, cond_intrinsics, cond_h, cond_w)
        else:
            raise NotImplementedError(f"Do not support {cond_intrinsics.shape} intrinsics")
        rays_p = torch.cross(rays_d, rays_o, dim=-1)  # (num_scenes, num_imgs, h, w, 3)
        if self.cond_mode == "rgb":
            cond_input = torch.concat([cond_imgs, rays_d, rays_p], dim=-1)
        else:
            assert "cond_masks" in data_batch
            cond_masks = data_batch["cond_masks"]  # (num_scenes, num_imgs, h, w, 1)
            cond_input = torch.concat([cond_imgs, rays_d, rays_p, cond_masks], dim=-1)

        cond_input = cond_input.permute(0, 1, 4, 2, 3)
        code = self.generator(cond_input)

        return code

    @custom_fwd(cast_inputs=torch.float32)
    def _gen_mesh(self, code):
        """Retrieve density for grid and convert to mesh"""
        num_scenes = code.shape[0]
        verts_list, faces_list = [], []
        for i in range(num_scenes):
            grid_pts = self.mesh_decoder.get_grid_pts()
            grid_pts = grid_pts.view(-1, 3)
            # logger.debug(f"get point density: {grid_pts.device} {code.device} grad: {grid_pts.grad}")

            density_list = []
            num_pts = grid_pts.shape[0]
            num_pts_per_chunk = self.num_pts_per_chunk
            for j in range(0, num_pts, num_pts_per_chunk):
                if self.gradient_checkpointing:
                    density = torch.utils.checkpoint.checkpoint(
                        self.field_module.get_point_density,
                        grid_pts[j:j+num_pts_per_chunk][None],
                        code[i][None],
                        use_reentrant=False,
                    )[0]
                else:
                    density = self.field_module.get_point_density(
                        grid_pts[j:j+num_pts_per_chunk][None],
                        code[i][None],
                    )[0]
                density_list.append(density)
            density = torch.cat(density_list, dim=0)

            verts, faces = self.mesh_decoder.get_geometry_from_density(density, i)  # density, batch_idx
            verts_list.append(verts)
            faces_list.append(faces)
        return verts_list, faces_list

    @custom_fwd(cast_inputs=torch.float32)
    def _decode_appearance(self, code, tex_pts, masks):
        """convert pixel pts to images"""
        if self.tex_pts_detach:
            tex_pts = tex_pts.detach().clone()
            masks = masks.detach().clone()
        num_scenes, num_views, height, width, _ = tex_pts.shape
        assert masks.ndim == tex_pts.ndim
        # TODO: set background color
        rgbs = torch.ones((num_scenes, num_views, height, width, 3), device=self.device)  # white background
        for i in range(num_scenes):
            cur_mask = masks[i, ..., 0] > 0.5  # (num_views, height, width)
            
            valid_pts = tex_pts[i][cur_mask]  # (npts, 3)
            if valid_pts.shape[0] == 0:
                # empty object
                continue
                
            if self.gradient_checkpointing:
                pixel_pts_color = torch.utils.checkpoint.checkpoint(
                    self.field_module.get_point_color,
                    valid_pts[None],
                    code[i][None],
                    use_reentrant=False,
                )[0]
            else:
                pixel_pts_color = self.field_module.get_point_color(valid_pts[None], code[i][None])[0]
            rgbs[i][cur_mask] = pixel_pts_color
        return rgbs

    def base_forward(self, data_batch):
        """Prepare data, feed to model, and get loss dict """
        code = self._gen_triplane(data_batch)

        verts_list, faces_list = self._gen_mesh(code)
        mvp_mtx_list = data_batch["mvps"]
        mv_mtx_list = data_batch["m2vs"]

        results = self.mesh_decoder.batch_render_geometry(
            verts_list,
            faces_list,
            mvp_mtx_list,
            mv_mtx_list
        )

        hard_masks = results["hard_mask"]
        tex_pts = results["tex_pts"]
        pred_imgs = self._decode_appearance(code, tex_pts, hard_masks)
        results["rgb"] = pred_imgs

        # ray opacity loss
        with torch.no_grad():
            imgs = data_batch["imgs"]
            intrinsics = data_batch["intrinsics"]
            c2ws = data_batch["poses"]
            num_scenes, num_imgs, h, w, _ = imgs.shape
            gt_depth = data_batch["depths"].view(num_scenes, -1, 1)  # num_scenes, npts 1
            
            
            rays_o, rays_d = get_cam_rays(c2ws, intrinsics, h, w, norm=False)
            rays_o = rays_o.reshape(num_scenes, -1, 3)  # num_scenes, npts, 3
            rays_d = rays_d.reshape(num_scenes, -1, 3)  # num_scenes, npts, 3
            nears, fars = self.field_module.get_rays_near_far_raw(rays_o, rays_d)
            assert nears.ndim == 3  # num_scenes, npts, 1
            assert fars.ndim == 3  # num_scenes, npts, 1
            is_ray_valid = fars > nears  # num_scenes, npts, 1  # rays that intersect with bounding box
            depth_mask = gt_depth > 0  # foreground mask

            # modify invalid rays near far to avoid too large values
            if torch.any(is_ray_valid).item():
                nears[~is_ray_valid] = nears[is_ray_valid].min()
                fars[~is_ray_valid] = fars[is_ray_valid].max()

            # specify fars
            fars[depth_mask] = gt_depth[depth_mask]  # for foreground pixel rays, set far to surface pts
            t_rand = torch.rand((num_scenes, (num_imgs * h * w), 1), device=rays_o.device)
            samples = t_rand * (fars - nears) + nears  # num_scenes, npts, 1
            ray_pts = rays_o + samples * rays_d  # num_scenes, npts, 3
            ray_len = torch.abs(samples - fars) * torch.norm(rays_d, dim=-1, keepdim=True)  # num_scenes, npts, 1 TODO: should we take rays_dir norm into consideration
            # there are possible points outside [-1, 1] range

        ray_opacities = torch.zeros((num_scenes, num_imgs * h * w, 1), device=code.device)
        for i in range(num_scenes):
            chunk_pts_density = self.field_module.get_point_density(
                ray_pts[i][None],
                code[i][None],
            )[0]  # npts, 1
            chunk_opacities = 1.0 - torch.exp(- chunk_pts_density * ray_len[i])
            cur_valid_ray_mask = is_ray_valid[i]
            ray_opacities[i][cur_valid_ray_mask] = chunk_opacities[cur_valid_ray_mask]
        results["ray_opacities"] = ray_opacities

        return results

    def get_train_loss_dict(self, data_batch):
        preds = self.base_forward(data_batch)

        loss_dict = self.mesh_loss(preds, data_batch)
        return loss_dict

    @torch.no_grad()
    def inference(self, data_batch):
        self.switch_eval()

        outputs = self.base_forward(data_batch)
        
        imgs = data_batch['imgs']  # (num_scenes, num_imgs, h, w, 3)
        num_scenes, num_imgs, h, w, _ = imgs.shape
        pred_image = outputs['rgb'].reshape(num_scenes * num_imgs, h, w, 3).contiguous().permute(0, 3, 1, 2)
        gt_image = imgs.reshape(num_scenes * num_imgs, h, w, 3).contiguous().permute(0, 3, 1, 2)
        psnr = self.psnr(gt_image, pred_image)
        ssim = cast(torch.Tensor, self.ssim(gt_image, pred_image))
        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
        }

        self.switch_train()
        return outputs, metrics_dict

    @torch.no_grad()
    def prepare_geo(self, data_batch):
        code = self._gen_triplane(data_batch)
        verts_list, faces_list = self._gen_mesh(code)
        
        mvp_mtx_list = data_batch["cond_mvps"]
        mv_mtx_list = data_batch["cond_mvps"]

        results = self.mesh_decoder.batch_render_geometry(
            verts_list,
            faces_list,
            mvp_mtx_list,
            mv_mtx_list
        )
        
        hard_masks = results["hard_mask"]
        tex_pts = results["tex_pts"]
        
        return code, tex_pts, hard_masks
    
    def texture_forward(self, code, text_pts, hard_masks):
        """Prepare data, feed to model, and get loss dict """
        pred_imgs = self._decode_appearance(code, text_pts, hard_masks)
        return pred_imgs
    
    def refine_texture(self, data_batch, optimizers, grad_scaler, iters=10, mixed_precision='fp16', learn_code=False):
        code, text_pts, hard_masks = self.prepare_geo(data_batch)

        self.switch_train()
        # make it traininable
        if learn_code:
            code = torch.nn.parameter.Parameter(code, requires_grad=True)
        else:
            code = torch.nn.parameter.Parameter(code, requires_grad=False)
            
        cond_imgs = data_batch["cond_imgs"]
        enable_amp = mixed_precision != "no"
        amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
        import torch.optim as optim
        
        optimizers.optimizers['model'].add_param_group({'params': [code], 'lr': .15})

        for i in tqdm(range(iters)):
            # num_scene, num_imgs, h, w, 3
            pred_imgs = self.texture_forward(code, text_pts, hard_masks)
            
            # compute loss
            optimizers.zero_grad_all()
            if pred_imgs.shape[2] != 512:
                pred_imgs = F.upsample(pred_imgs[0].permute(0,3,1,2), (512,512), mode='bilinear').permute(0,2,3,1)
                pred_imgs = pred_imgs.unsqueeze(0)
            with autocast(enabled=enable_amp, dtype=amp_dtype):
                loss = F.l1_loss(pred_imgs, cond_imgs)

            grad_scaler.scale(loss).backward()
            optimizers.optimizer_scaler_step_all(grad_scaler)
            scale = grad_scaler.get_scale()
            grad_scaler.update()
            # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
            if scale <= grad_scaler.get_scale():
                optimizers.scheduler_step_all()
        param_group_to_remove = optimizers.optimizers['model'].param_groups[-1]
        # Remove the identified parameter group
        optimizers.optimizers['model'].param_groups = [pg for pg in optimizers.optimizers['model'].param_groups if pg is not param_group_to_remove]

        return code, loss.item()

    def base_forward_with_code(self, data_batch, code, compute_ray_opacity=True):
        """Prepare data, feed to model, and get loss dict """
        code_geo = self._gen_triplane(data_batch)
        # verts_list, faces_list = self._gen_mesh(code)
        verts_list, faces_list = self._gen_mesh(code_geo)

        mvp_mtx_list = data_batch["mvps"]
        mv_mtx_list = data_batch["m2vs"]

        results = self.mesh_decoder.batch_render_geometry(
            verts_list,
            faces_list,
            mvp_mtx_list,
            mv_mtx_list
        )

        hard_masks = results["hard_mask"]
        tex_pts = results["tex_pts"]
        pred_imgs = self._decode_appearance(code, tex_pts, hard_masks)
        results["rgb"] = pred_imgs

        # ray opacity loss
        if compute_ray_opacity:
            with torch.no_grad():
                imgs = data_batch["imgs"]
                intrinsics = data_batch["intrinsics"]
                c2ws = data_batch["poses"]
                num_scenes, num_imgs, h, w, _ = imgs.shape
                gt_depth = data_batch["depths"].view(num_scenes, -1, 1)  # num_scenes, npts 1
                
                rays_o, rays_d = get_cam_rays(c2ws, intrinsics, h, w, norm=False)
                rays_o = rays_o.reshape(num_scenes, -1, 3)  # num_scenes, npts, 3
                rays_d = rays_d.reshape(num_scenes, -1, 3)  # num_scenes, npts, 3
                nears, fars = self.field_module.get_rays_near_far_raw(rays_o, rays_d)
                assert nears.ndim == 3  # num_scenes, npts, 1
                assert fars.ndim == 3  # num_scenes, npts, 1
                is_ray_valid = fars > nears  # num_scenes, npts, 1  # rays that intersect with bounding box
                depth_mask = gt_depth > 0  # foreground mask

                # modify invalid rays near far to avoid too large values
                if torch.any(is_ray_valid).item():
                    nears[~is_ray_valid] = nears[is_ray_valid].min()
                    fars[~is_ray_valid] = fars[is_ray_valid].max()

                # specify fars
                fars[depth_mask] = gt_depth[depth_mask]  # for foreground pixel rays, set far to surface pts
                t_rand = torch.rand((num_scenes, (num_imgs * h * w), 1), device=rays_o.device)
                samples = t_rand * (fars - nears) + nears  # num_scenes, npts, 1
                ray_pts = rays_o + samples * rays_d  # num_scenes, npts, 3
                ray_len = torch.abs(samples - fars) * torch.norm(rays_d, dim=-1, keepdim=True)  # num_scenes, npts, 1 TODO: should we take rays_dir norm into consideration
                # there are possible points outside [-1, 1] range

            ray_opacities = torch.zeros((num_scenes, num_imgs * h * w, 1), device=code.device)
            for i in range(num_scenes):
                chunk_pts_density = self.field_module.get_point_density(
                    ray_pts[i][None],
                    code[i][None],
                )[0]  # npts, 1
                chunk_opacities = 1.0 - torch.exp(- chunk_pts_density * ray_len[i])
                cur_valid_ray_mask = is_ray_valid[i]
                ray_opacities[i][cur_valid_ray_mask] = chunk_opacities[cur_valid_ray_mask]
            results["ray_opacities"] = ray_opacities

        return results

    @torch.no_grad()
    def inference_with_code(self, data_batch, code):
        self.switch_eval()

        outputs = self.base_forward_with_code(data_batch, code, compute_ray_opacity=False)

        metrics_dict = {}
        self.switch_train()
        return outputs, metrics_dict
  
    @torch.no_grad()
    def get_image_metrics_and_images(self, data_batch):
        def to_img(x):
            x = x.cpu().detach().float().numpy()
            num_scenes, num_imgs, h, w, _ = x.shape
            x = x.reshape(num_scenes * num_imgs, h, w, -1)
            grid = make_grid(x, nx=num_imgs)
            grid = (grid * 255.).astype(np.uint8)
            return grid

        outputs, metrics_dict = self.inference(data_batch)

        vis_tensor_names = [
            'img_cond',
            'img_gt',
            'img_pred',
            'mask_gt',
            'mask_pred',
            'normal_pred',
        ]
        vis_imgs = []
        vis_imgs.append(to_img(data_batch['cond_imgs']))
        vis_imgs.append(to_img(data_batch['imgs']))
        vis_imgs.append(to_img(outputs['rgb']))
        vis_imgs.append(to_img(data_batch['masks'].expand(-1, -1, -1, -1, 3)))
        vis_imgs.append(to_img(outputs['antialias_mask'].expand(-1, -1, -1, -1, 3)))
        vis_imgs.append(to_img(outputs['normal']*0.5 + 0.5))
        visuals = zip(vis_tensor_names, vis_imgs)
        return OrderedDict(visuals), metrics_dict

    @torch.no_grad()
    def extract_geometry(self, data_batch, resolution=128, level=5, scene_name=None, save_density=False, code=None):
        self.switch_eval()
        
        code_geo = self._gen_triplane(data_batch)
        verts_list, faces_list = self._gen_mesh(code_geo)
        
        if code is None:
            code = code_geo
            print("CODE IS NONE!")
        else:
            print('Using existing code')
        

        mesh_list = []
        num_scenes = code.shape[0]
        for i in range(num_scenes):
            pts = verts_list[i]
            faces = faces_list[i]
            logger.debug(pts.shape)
            pts = pts.reshape(-1, 3)
            _, pts_color = self.field_module.get_point_density_color(pts[None], code[i][None])
            pts_color = pts_color.clamp(min=0, max=1)
            vertex_colors = (pts_color[0].cpu().numpy() * 255.0).astype(np.uint8)

            out_mesh = trimesh.Trimesh(
                vertices=pts.detach().cpu().numpy(),
                faces=faces.cpu().numpy(),
                vertex_colors=vertex_colors
            )
            mesh_list.append(out_mesh)

        self.switch_train()
        return mesh_list



