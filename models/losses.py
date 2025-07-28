from dataclasses import dataclass
from einops import rearrange, repeat
from loguru import logger

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from utils.typing import *
from utils.modeling_util import requires_grad, scale_dict
from utils.config_utils import to_immutable_dict, PrintableConfig
import lpips
from torchvision.utils import save_image
import imageio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

L1Loss = nn.L1Loss
MSELoss = nn.MSELoss

LOSSES = {"L1": L1Loss, "MSE": MSELoss}

EPS = 1.0e-7


def tv_loss(grids: Float[Tensor, "grids feature_dim row column"]) -> Float[Tensor, ""]:
    """
    https://github.com/apchenstu/TensoRF/blob/4ec894dc1341a2201fe13ae428631b58458f105d/utils.py#L139

    Args:
        grids: stacks of explicit feature grids (stacked at dim 0)
    Returns:
        average total variation loss for neighbor rows and columns.
    """
    number_of_grids = grids.shape[0]
    h_tv_count = grids[:, :, 1:, :].shape[1] * grids[:, :, 1:, :].shape[2] * grids[:, :, 1:, :].shape[3]
    w_tv_count = grids[:, :, :, 1:].shape[1] * grids[:, :, :, 1:].shape[2] * grids[:, :, :, 1:].shape[3]
    h_tv = torch.pow((grids[:, :, 1:, :] - grids[:, :, :-1, :]), 2).sum()
    w_tv = torch.pow((grids[:, :, :, 1:] - grids[:, :, :, :-1]), 2).sum()
    return 2 * (h_tv / h_tv_count + w_tv / w_tv_count) / number_of_grids


class LPIPS(torch.nn.Module):
    def __init__(self, net="vgg"):
        super().__init__()
        self.model = lpips.LPIPS(lpips=True, net=net)

    def forward(self, batch1: torch.Tensor, batch2: torch.Tensor):
        loss = self.model(batch1, batch2, normalize=True).mean()
        return loss


def normal_cos_loss(
    pred_normals: Float[Tensor, "*bs 3"],
    normals: Float[Tensor, "*bs 3"],
    weights: Optional[Float[Tensor, "*bs 1"]] = None,
):
    """Angular loss between normals and pred_normals"""
    assert pred_normals.ndim == normals.ndim
    similarity = torch.sum(normals * pred_normals, dim=-1).abs()
    if weights is not None:
        loss_normal = 1 - similarity[weights[...,0]>0].mean()
        return loss_normal
    else:
        loss_normal = 1 - similarity.mean()
        return loss_normal.mean()


def normal_l1_loss(
    pred_normals: Float[Tensor, "*bs 3"],
    normals: Float[Tensor, "*bs 3"],
    weights: Optional[Float[Tensor, "*bs 1"]] = None,
):
    """L1 Loss between normals and pred_normals"""
    assert pred_normals.ndim == normals.ndim
    l1 = torch.abs(pred_normals - normals).sum(dim=-1)
    if weights is not None:
        return (weights[..., 0] * l1).mean()
    else:
        return l1.mean()


def normal_l2_loss(
    pred_normals: Float[Tensor, "*bs 3"],
    normals: Float[Tensor, "*bs 3"],
    weights: Optional[Float[Tensor, "*bs 1"]] = None,
):
    """L2 Loss between normals and pred_normals"""
    assert pred_normals.ndim == normals.ndim
    l2 = torch.mean(torch.pow(pred_normals - normals, 2), dim=-1)
    if weights is not None:
        return (weights[..., 0] * l2).mean()
    else:
        return l2.mean()


def orientation_loss(
    normals: Float[Tensor, "*bs num_samples 3"],
    view_dirs: Float[Tensor, "*bs 3"],
    weights: Float[Tensor, "*bs num_samples 1"],
):
    """Orientation loss proposed in Ref-NeRF.
    Loss that encourages that all visible normals are facing towards the camera.
    Returns: Float[Tensor, "*bs"]
    """
    w = weights
    n = normals
    v = view_dirs * -1
    n_dot_v = (n * v[..., None, :]).sum(dim=-1)
    return (w[..., 0] * torch.fmin(torch.zeros_like(n_dot_v), n_dot_v) ** 2).sum(dim=-1)


@dataclass
class MeshRenderLossConfig(PrintableConfig):
    """Loss model config"""
    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
            "loss_rgb": 1.0,
            "loss_lpips": 0.0,
        }
    )
    """Dict of loss coefficients"""
    lpips_res: int = 256
    """Resolution of lpips model"""


class MeshRenderLoss(nn.Module):
    def __init__(
        self,
        cfg: MeshRenderLossConfig,
        device,
    ):
        super().__init__()
        self.device = device
        self.loss_coefficients = cfg.loss_coefficients
        self.lpips_res = cfg.lpips_res

        self.rgb_loss = MSELoss()
        self.mask_loss = L1Loss()
        self.ct = 0
        if 'loss_lpips' in self.loss_coefficients and self.loss_coefficients['loss_lpips'] > 0.0:
            self.lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, preds, targets):
        loss_dict = {}

        num_scenes, num_imgs, h, w, _ = targets["imgs"].shape
        target_rgb = targets["imgs"]
        pred_rgb = preds["rgb"]


        rgb_loss = self.rgb_loss(target_rgb, pred_rgb)
        loss_dict['loss_rgb'] = rgb_loss

        if 'loss_mask' in self.loss_coefficients and self.loss_coefficients['loss_mask'] > 0.0:
            pred_masks = (preds['antialias_mask']).float()
            target_masks = (targets['masks']).float()
            mask_loss = F.mse_loss(pred_masks, target_masks)
            loss_dict['loss_mask'] = mask_loss

        if 'loss_depth' in self.loss_coefficients and self.loss_coefficients['loss_depth'] > 0.0:
            pred_depth = preds['depth']
            gt_depth = targets['depths']
            depth_masks = (gt_depth > 0).float()
            depth_loss = torch.pow(pred_depth - gt_depth, 2) * depth_masks
            loss_dict['loss_depth'] = depth_loss.mean()            

        if 'loss_ray_opacity' in self.loss_coefficients and self.loss_coefficients['loss_ray_opacity'] > 0.0:
            assert "ray_opacities" in preds
            loss_dict['loss_ray_opacity'] = torch.abs(preds["ray_opacities"]).mean()

        if 'loss_normal' in self.loss_coefficients and self.loss_coefficients['loss_normal'] > 0.0:
            pred_normal = preds['normal']
            gt_normal = targets['normals']  
            target_masks = (targets['masks']).float()
            pred_masks = (preds['antialias_mask']).float()
            mask = target_masks * pred_masks

            bg_normal = torch.ones_like(pred_normal) * torch.tensor([0.0, 0.0, 1.0], device=pred_normal.device)
            pred_normal = pred_normal * mask   + (1.0 - mask) * bg_normal
            gt_normal = gt_normal * mask  + (1.0 - mask) * bg_normal
            
            loss_dict['loss_normal'] = normal_cos_loss(pred_normal, gt_normal, mask)

        if 'loss_normal' in self.loss_coefficients and self.loss_coefficients['loss_normal_l2'] > 0.0:
            pred_normal = preds['normal']
            gt_normal = targets['normals']
            target_masks = targets['masks']
            pred_masks = (preds['antialias_mask']).float()

            
            mask = target_masks * pred_masks

            bg_normal = torch.ones_like(pred_normal) * torch.tensor([0.0, 0.0, 1.0], device=pred_normal.device)
            pred_normal = pred_normal * mask + (1.0 - mask) * bg_normal
            gt_normal = gt_normal * mask + (1.0 - mask) * bg_normal
            
            loss_dict['loss_normal_l2'] = normal_l2_loss(pred_normal, gt_normal, mask)


        if 'loss_lpips' in self.loss_coefficients and self.loss_coefficients['loss_lpips'] > 0.0:
            pred_rgb, target_rgb = map(lambda t: rearrange(t, 'b n h w c -> (b n) c h w').contiguous(), (pred_rgb, target_rgb))
            if pred_rgb.shape[-1] != self.lpips_res:
                pred_rgb = F.interpolate(pred_rgb, (self.lpips_res, self.lpips_res), mode='bilinear', align_corners=False)
                target_rgb = F.interpolate(target_rgb, (self.lpips_res, self.lpips_res), mode='bilinear', align_corners=False)
            
            pred_rgb = pred_rgb.clamp(0,1)
            lpips_loss = self.lpips_loss(pred_rgb* 2 - 1, target_rgb* 2 - 1)
            
            loss_dict['loss_lpips'] = lpips_loss.mean()

        loss_dict = scale_dict(loss_dict, self.loss_coefficients)
        return loss_dict
