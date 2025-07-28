from dataclasses import dataclass
from collections import defaultdict
from loguru import logger
import functools

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio

from utils.ray_samples import RayBundle, UniformSampler
from utils.ray_utils import get_cam_rays, get_ortho_cam_rays
from utils.math import intersect_aabb
from utils.colors import get_color

from models.volume_rendering.renderer import RGBRenderer, AccumulationRenderer, DepthRenderer, NormalsRenderer
from models.decoders.triplane_decoder import TriplaneDecoder
from models.losses import MSELoss, tv_loss, LPIPS, normal_cos_loss, normal_l2_loss, normal_l1_loss
from models.volume_rendering.deferred_render import deferred_bp, forward_chunk

from utils.config_utils import to_immutable_dict, PrintableConfig
from utils.typing import *
from utils.modeling_util import requires_grad, scale_dict
from utils.distributed import get_rank

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


@dataclass
class BaseNeRFModelConfig(PrintableConfig):
    """BaseNeRF model config"""
    num_samples_per_ray: int = 128
    """Number of samples per ray"""
    single_jitter: bool = False
    """Use a same random jitter for all samples along a ray. Defaults to True"""
    background_color: Literal["black", "white"] = "black"
    """background color"""
    decoder_type: Literal["TriPlaneDecoder"] = "TriPlaneDecoder"
    """Decoder type"""
    decoder_args: Dict[str, Union[int, str]] = to_immutable_dict(
        {
            "in_dim": 3 * 80,
            "num_layers": 4,
            "layer_width": 128,
            "out_dim": 4,
            "sigma_activation": "softplus",
        }
    )
    min_near: float = 0.2
    """min value of ray nears"""
    bound: float = 1.0
    """Scene bound """
    apply_forward_chunk: bool = False
    """Set true to chunk rays in forward function"""
    eval_num_rays_per_chunk: int = 10240
    """Number of rays per evaluation"""
    apply_ray_mask: bool = False
    """Set true to mask rays do not intersect with AABB"""
    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
            "loss_rgb": 1.0,
            "loss_lpips": 0.0,
            "loss_normal_l1": 0.0,
            "loss_normal_l2": 0.0,
            "loss_normal_cos": 0.0,
        }
    )
    gradient_checkpointing: bool = False
    """Set True to apply gradient checkpointing to point_decoder"""


class BaseNeRF(nn.Module):
    """Represent scenes with latent encodings and decode latent encodings to color and density with NeRF"""

    def __init__(
        self,
        cfg: BaseNeRFModelConfig,
        device,
    ):
        super().__init__()
        self.num_samples_per_ray = cfg.num_samples_per_ray
        self.eval_num_rays_per_chunk = cfg.eval_num_rays_per_chunk
        self.min_near = cfg.min_near
        self.background_color = get_color(cfg.background_color)
        self.apply_ray_mask = cfg.apply_ray_mask
        self.single_jitter = cfg.single_jitter
        self.apply_forward_chunk = cfg.apply_forward_chunk
        self.loss_coefficients = cfg.loss_coefficients
        self.device = device

        # field
        if cfg.decoder_type == "TriPlaneDecoder":
            self.decoder = TriplaneDecoder(**cfg.decoder_args)
        else:
            raise NotImplementedError(f"Not found decoder {self.cfg.decoder_type}")
        self.use_pred_normal = self.decoder.use_pred_normal

        # ray samplers
        # TODO: eval_sampler
        self.train_sampler = UniformSampler(
            num_samples=self.num_samples_per_ray,
            train_stratified=True,
            single_jitter=self.single_jitter,
        )

        # renders
        self.renderer_rgb = RGBRenderer()
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()

        # losses
        self.rgb_loss = MSELoss()
        if 'loss_lpips' in self.loss_coefficients and self.loss_coefficients['loss_lpips'] > 0.0:
            # TODO: !!! remove lpips from model parameter and state_dict and training. Current set training will cause lpips training
            # https://github.com/Lightning-AI/lightning/issues/2824
            lpips_loss = LPIPS().to(device)
            requires_grad(lpips_loss, False)
            lpips_loss.eval()
            self.lpips_loss = [lpips_loss]  # hack to remove lpips from model parameters, child modules.

                                                                
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure

        # colliders
        bound = cfg.bound
        aabb = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        self.register_buffer('aabb', aabb)

        # Add gradient checkpointing here
        self.gradient_checkpointing = cfg.gradient_checkpointing

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def dp_render_wrap(self, xyzs, code):
        density, color = deferred_bp(xyzs, code, self.eval_num_rays_per_chunk, self.decoder)
        outputs = {"density": density, "rgb": color}
        return outputs

    @torch.no_grad()
    def get_rays_near_far_raw(self, rays_o, rays_d):
        num_scenes, num_rays, _ = rays_o.shape
        nears, fars = intersect_aabb(
            rays_o.reshape(-1, 3),
            rays_d.reshape(-1, 3),
            self.aabb,
            min_near=self.min_near
        )
        nears = nears.view(num_scenes, num_rays, 1)
        fars = fars.view(num_scenes, num_rays, 1)
        return nears, fars

    def forward(self, rays_o, rays_d, code, background_color=None):
        """Given rays, output color and density
        Args:
            rays_o: torch.Tensor, (num_scenes, num_rays, 3)
            rays_d: torch.Tensor, (num_scenes, num_rays, 3)
            code: torch.Tensor, (num_scenes, *code_size)
            background_color: background color for rays
        """
        assert rays_o.shape[0] == code.shape[0]
        num_scenes, num_rays, _ = rays_o.shape

        with torch.no_grad():
            nears, fars = intersect_aabb(
                rays_o.reshape(-1, 3),
                rays_d.reshape(-1, 3),
                self.aabb,
                min_near=self.min_near
            )
            is_ray_valid = fars > nears
            if torch.any(is_ray_valid).item():
                nears[~is_ray_valid] = nears[is_ray_valid].min()
                fars[~is_ray_valid] = fars[is_ray_valid].max()
            nears = nears.view(num_scenes, num_rays, 1)
            fars = fars.view(num_scenes, num_rays, 1)
            is_ray_valid = is_ray_valid.view(num_scenes, num_rays)

            # ray samples
            ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
            ray_samples = self.train_sampler.generate_ray_samples(ray_bundle)
            xyzs = ray_samples.get_positions()

        num_scenes, num_rays, num_samples, _ = xyzs.shape
        xyzs = xyzs.reshape(num_scenes, -1, 3)
        if self.gradient_checkpointing:
            outputs = self.dp_render_wrap(xyzs, code)
        else:
            outputs = forward_chunk(xyzs, code, self.eval_num_rays_per_chunk, self.decoder)
        sigmas = outputs["density"].reshape(num_scenes, num_rays, num_samples, -1)
        rgbs = outputs["rgb"].reshape(num_scenes, num_rays, num_samples, -1)

        if self.apply_ray_mask:
            sigmas[~is_ray_valid] = 0.0
        weights = ray_samples.get_weights(sigmas)  # (num_scenes, num_rays, num_samples, 1)

        # TODO: move background color to dataloader
        if background_color is None:
            background_color = self.background_color

        # check dimension of background color
        if background_color.ndim == 1:
            pass
        elif background_color.ndim == 2:
            background_color = background_color.unsqueeze(1)  # (num_scenes, num_ray, c)
        else:
            raise ValueError(f'Wrong dimension of background_color tensor: {background_color.ndim}')

        rgb = self.renderer_rgb(rgb=rgbs, weights=weights, background_color=background_color)  # (num_scenes, num_rays, 3)
        accumulation = self.renderer_accumulation(weights)
        accum_mask = torch.clamp((torch.nan_to_num(accumulation, nan=0.0)), min=0.0, max=1.0)
        results = {
            "rgb": rgb,
            "accumulation": accumulation,
            "accum_mask": accum_mask,
        }

        if self.use_pred_normal and "pred_normal" in outputs:
            pred_normal = outputs["pred_normal"].reshape(num_scenes, num_rays, num_samples, -1)
            pred_normals_img = self.renderer_normals(
                normals=pred_normal, weights=weights, normalize=True
            )  # (num_scenes, num_rays, 3)
            results["pred_normal"] = pred_normals_img

        # depth = self.renderer_depth(weights, ray_samples)
        return results

    def get_point_density(self, xyzs, code):
        outputs = self.decoder.point_decode(xyzs, code, density_only=True)
        return outputs["density"]

    def get_point_density_color(self, xyzs, code):
        outputs = self.decoder.point_decode(xyzs, code)
        return outputs["density"], outputs["rgb"]

    def get_point_color(self, xyzs, code):
        rgbs = self.decoder.point_decode_color(xyzs, code)
        return rgbs

    def get_loss_dict(self, outputs, batch, patch_size) -> Dict[str, torch.Tensor]:
        """Return loss_dict"""
        loss_dict = {}

        num_scenes, num_imgs, h, w, _ = batch["imgs"].shape
        assert 'rgb' in outputs and 'target_rgb' in batch
        target_rgb = batch["target_rgb"]
        pred_rgb = outputs["rgb"]

        rgb_loss = self.rgb_loss(target_rgb, pred_rgb)
        loss_dict['loss_rgb'] = rgb_loss
        if patch_size is not None and "loss_lpips" in self.loss_coefficients:
            if self.loss_coefficients["loss_lpips"] > 0.0:
                pred_patch = pred_rgb.reshape(
                    num_scenes * num_imgs,
                    patch_size,
                    patch_size,
                    3
                )
                gt_patch = target_rgb.reshape(num_scenes * num_imgs, patch_size, patch_size, 3)
                pred_patch = pred_patch.contiguous().permute(0, 3, 1, 2)
                gt_patch = gt_patch.contiguous().permute(0, 3, 1, 2)
                lpips_loss = self.lpips_loss[0](pred_patch, gt_patch)
                
                loss_dict['loss_lpips'] = lpips_loss

        if self.use_pred_normal:
            target_normal = batch["target_normal"]
            pred_normal = outputs["pred_normal"]
            weights = outputs["accum_mask"]  # TODO: replace accum_mask with gt mask ???
            if self.loss_coefficients.get("loss_normal_l1", 0.0) > 0.0:
                loss_normal = normal_l1_loss(
                    pred_normal,
                    target_normal,
                    weights.detach()
                )
                loss_dict['loss_normal_l1'] = loss_normal
            if self.loss_coefficients.get("loss_normal_l2", 0.0) > 0.0:
                loss_normal = normal_l1_loss(
                    pred_normal,
                    target_normal,
                    weights.detach()
                )
                loss_dict['loss_normal_l2'] = loss_normal
            if self.loss_coefficients.get("loss_normal_cos", 0.0) > 0.0:
                loss_normal = normal_cos_loss(
                    pred_normal,
                    target_normal,
                    weights.detach()
                )
                loss_dict['loss_normal_cos'] = loss_normal

        loss_dict = scale_dict(loss_dict, self.loss_coefficients)
        return loss_dict

    @torch.no_grad()
    def get_metric_dict(self, outputs, batch, patch_size) -> Dict[str, float]:
        """Return metric dict"""
        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        num_scenes, num_imgs, h, w, _ = batch["imgs"].shape

        assert 'rgb' in outputs and 'target_rgb' in batch
        target_rgb = batch["target_rgb"]
        pred_rgb = outputs["rgb"]

        pred_patch = pred_rgb.reshape(num_scenes * num_imgs, patch_size, patch_size, 3)
        gt_patch = target_rgb.reshape(num_scenes * num_imgs, patch_size, patch_size, 3)
        pred_patch = pred_patch.contiguous().permute(0, 3, 1, 2)  # (batch_size, C, H ,W)
        gt_patch = gt_patch.contiguous().permute(0, 3, 1, 2)  # (batch_size, C, H ,W)
        psnr = self.psnr(gt_patch, pred_patch)
        ssim = cast(torch.Tensor, self.ssim(gt_patch, pred_patch))
        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
        }
        return metrics_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, data, code, return_metrics=True):
        """For inference, return render images for camera"""
        # TODO: refactor h, w to data['intrinsics']
        num_rays_per_chunk = 65536  # 128 * 128 patch_size

        _, _, h, w, _ = data['imgs'].size()  # (num_scenes, num_imgs, h, w, 3)
        intrinsics = data['intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
        c2w = data['poses']  # (num_scenes, num_imgs, 4, 4)
        num_scenes, num_imgs, _ = intrinsics.size()

        if "background_color" in data:
            background_color = data["background_color"]
        else:
            background_color = None

        total_outputs_list = defaultdict(list)
        for batch_idx in range(num_scenes):
            cur_c2w = c2w[batch_idx][None]
            cur_intrinsics = intrinsics[batch_idx][None]
            cur_code = code[batch_idx][None]
            if intrinsics.shape[-1] == 2:
                rays_o, rays_d = get_ortho_cam_rays(cur_c2w, cur_intrinsics, h, w)
            else:
                rays_o, rays_d = get_cam_rays(cur_c2w, cur_intrinsics, h, w)  # (num_scenes, num_imgs, h, w, 3)
            rays_o = rays_o.reshape(1, -1, 3)
            rays_d = rays_d.reshape(1, -1, 3)
            num_rays = rays_o.shape[1]

            cur_outputs_lists = defaultdict(list)
            for i in range(0, num_rays, num_rays_per_chunk):
                start_idx = i
                end_idx = i + num_rays_per_chunk
                slice_rays_o = rays_o[:, start_idx:end_idx]
                slice_rays_d = rays_d[:, start_idx:end_idx]
                outputs = self.forward(
                    slice_rays_o,
                    slice_rays_d,
                    cur_code,
                    background_color=background_color,
                )
                for output_name, output in outputs.items():  # type: ignore
                    if not torch.is_tensor(output):
                        continue
                    cur_outputs_lists[output_name].append(output)
            cur_outputs = {}
            for output_name, outputs_list in cur_outputs_lists.items():
                cur_outputs[output_name] = torch.cat(outputs_list, dim=1).reshape(1, num_imgs, h, w, -1)

            for output_name, output in cur_outputs.items():
                total_outputs_list[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in total_outputs_list.items():
            outputs[output_name] = torch.cat(outputs_list, dim=0)

        if return_metrics:
            imgs = data['imgs']  # (num_scenes, num_imgs, h, w, 3)
            pred_image = outputs['rgb'].reshape(num_scenes * num_imgs, h, w, 3).contiguous().permute(0, 3, 1, 2)
            gt_image = imgs.reshape(num_scenes * num_imgs, h, w, 3).contiguous().permute(0, 3, 1, 2)
            psnr = self.psnr(gt_image, pred_image)
            ssim = cast(torch.Tensor, self.ssim(gt_image, pred_image))
            metrics_dict = {
                "psnr": float(psnr.item()),
                "ssim": float(ssim.item()),
            }
        else:
            metrics_dict = None

        return outputs, metrics_dict
