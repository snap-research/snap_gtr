import numpy as np
from loguru import logger
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def collate_pixel_sample(rays_o, rays_d, cond_imgs, sample_inds):
    """ Collate rays info based on sample_inds
    Args:
        rays_o (torch.Tensor): (num_scenes, num_imgs, h, w, 3)
        rays_d (torch.Tensor): (num_scenes, num_imgs, h, w, 3)
        cond_imgs (torch.Tensor): (num_scenes, num_imgs, h, w, 3)
        sample_inds (torch.Tensor): (num_scenes, n_samples), value in range [0, num_imgs * h * w -1 ]

    Returns:
        rays_o (torch.Tensor): (num_scenes, n_samples, 3)
        rays_d (torch.Tensor): (num_scenes, n_samples, 3)
        target_rgbs (torch.Tensor): (num_scenes, n_samples, 3)
    """
    device = rays_o.device
    num_scenes, num_imgs, h, w, _ = rays_o.size()
    num_scene_pixels = num_imgs * h * w
    rays_o = rays_o.reshape(num_scenes, num_scene_pixels, 3)
    rays_d = rays_d.reshape(num_scenes, num_scene_pixels, 3)
    target_rgbs = cond_imgs.reshape(num_scenes, num_scene_pixels, 3)

    scene_arange = torch.arange(num_scenes, device=device)[:, None]
    rays_o = rays_o[scene_arange, sample_inds]
    rays_d = rays_d[scene_arange, sample_inds]
    target_rgbs = target_rgbs[scene_arange, sample_inds]
    return rays_o, rays_d, target_rgbs


def collate_pixel_sample_single(rays_attr, sample_inds):
    """
    Args:
        rays_attr: (torch.Tensor): (num_scenes, num_imgs, h, w, 3)
        sample_inds (torch.Tensor): (num_scenes, n_samples), value in range [0, num_imgs * h * w -1 ]
            index array for (num_imgs * h * w)
    """
    device = rays_attr.device
    num_scenes, num_imgs, h, w, _ = rays_attr.size()
    num_scene_pixels = num_imgs * h * w
    rays_attr = rays_attr.reshape(num_scenes, num_scene_pixels, -1)

    scene_arange = torch.arange(num_scenes, device=device)[:, None]
    select_rays_attr = rays_attr[scene_arange, sample_inds]  # (num_scenes, num_select_rays, C)
    return select_rays_attr


def np_sample_index(weight_matrix: np.ndarray):
    """
    Randomly sample an index from a numpy array based on weight matrix.

    Args:
        weight_matrix: numpy array representing the weights for sampling.

    Returns:
        Tuple of indices sampled based on the weight matrix.
    """
    # how to make the number small
    # print(f"weight_matrix: {np.max(weight_matrix)} {np.min(weight_matrix)}, {np.sum(weight_matrix)}")
    if np.sum(weight_matrix) == 0.0:
        # incase we have zero weight_matrix
        # TODO: why it happens
        weight_matrix = np.ones_like(weight_matrix)
    normalized_weights = weight_matrix / np.sum(weight_matrix)
    if np.isnan(normalized_weights).any():
        print(f"Found NaNs in weight matrix: {np.sum(weight_matrix)} {np.max(weight_matrix)} {np.min(weight_matrix)}")
    flattened_weights = normalized_weights.flatten()
    # print(f"flatten_weight: {flattened_weights.sum()}")
    index = np.random.choice(np.arange(len(flattened_weights)), p=flattened_weights)
    coord = np.unravel_index(index, weight_matrix.shape)
    return coord


def torch_sample_index(weight_matrix):
    """
    Randomly sample an index from a 2D array based on weight matrix.

    Args:
        weight_matrix: 2D tensor representing the weights for sampling.

    Returns:
        Tuple of indices (row, column) sampled based on the weight matrix.
    """
    normalized_weights = weight_matrix / torch.sum(weight_matrix)
    flattened_weights = normalized_weights.flatten()
    index = torch.multinomial(flattened_weights, 1).item()
    row_index, col_index = index // weight_matrix.size(1), index % weight_matrix.size(1)
    return row_index, col_index


def patch_pixel_weighted_sample(
    imgs: torch.Tensor,
    masks: torch.Tensor,
    patch_size: int,
    sample_option: str,
    verbose=False,
):
    """
    Sample square patches from images randomly.
    Use masks to increase the likelihood of patches that covers foreground pixels.

    Args:
        imgs: torch.Tensor, (num_scenes, num_imgs, h, w, 3)
        masks: torch.Tensor, (num_scenes, num_imgs, h, w, 1)
        patch_size: int
        sample_option, str, one from ['base', 'weight']
        verbose: bool

    Returns:
        sample_inds: torch.Tensor, (num_scenes, n_samples), keep the flattened index of (img_idx, y_idx, x_idx)
    """
    device = imgs.device
    num_scenes, num_imgs, h, w, _ = imgs.size()
    assert h > patch_size and w > patch_size
    assert masks.ndim == imgs.ndim

    if sample_option == 'weight':
        kernel = torch.ones((1, 1, patch_size, patch_size), dtype=imgs.dtype, device=device) / patch_size
        masks = masks.reshape(num_scenes * num_imgs, 1, h, w)
        weights = F.conv2d(masks, kernel)  # (batch_size, channel, height, width)
        weights = weights.clamp(min=0.0)  # numerical precision
        weights = weights / patch_size  # numerical precision
        if verbose:
            print(f"Input mask: {masks.shape}, output weights: {weights.shape}")
        weight_h, weight_w = weights.shape[-2:]
        if weights.dtype == torch.bfloat16:
            weights = weights.float()
        np_weights = weights.reshape(num_scenes, num_imgs, weight_h, weight_w).detach().cpu().numpy()

    indices = []
    for i in range(num_scenes):
        for img_idx in range(num_imgs):
            if sample_option == 'base':
                y_idx = random.randint(0, h - patch_size - 1)
                x_idx = random.randint(0, w - patch_size - 1)
            elif sample_option == 'weight':
                cur_weight = np_weights[i, img_idx]  # (h, w)
                y_idx, x_idx = np_sample_index(cur_weight)
            else:
                raise NotImplementedError(f"Do not support sample_option {sample_option}")

            indices.append([img_idx, y_idx, x_idx])
    indices = torch.tensor(indices).to(device=device)
    if verbose:
        logger.debug(f"Select top-left corner: {indices}")

    bs = num_scenes * num_imgs
    indices_expand = indices.reshape(bs, 1, 1, 3).expand(-1, patch_size, patch_size, 3).clone()
    yys, xxs = torch.meshgrid(
        torch.arange(patch_size, device=device), torch.arange(patch_size, device=device)
    )
    indices_expand[:, ..., 1] += yys  # [img_idx, height_idx, width_idx]
    indices_expand[:, ..., 2] += xxs

    # flatten indices
    offset_img = h * w
    indices_expand = indices_expand[..., 0] * offset_img + indices_expand[..., 1] * w + indices_expand[..., 2]
    indices_expand = indices_expand.reshape(num_scenes, -1)
    return indices_expand


def pixel_sample(
    imgs: torch.tensor,
    num_rays_per_batch: int,
    verbose=False,
):
    """Sample pixel_batch from image_batch.

    Args:
        imgs: (torch.Tensor): (num_scenes, num_imgs, h, w, 3)
        num_rays_per_batch: (int)

    Returns:
        sample_inds (torch.Tensor): (num_scenes, n_samples), value in range [0, num_imgs * h * w -1 ]
    """
    device = imgs.device
    num_scenes, num_imgs, h, w, _ = imgs.size()
    num_scene_pixels = num_imgs * h * w
    assert num_rays_per_batch <= num_scene_pixels
    sample_inds = [torch.randperm(num_scene_pixels, device=device)[:num_rays_per_batch] for _ in range(num_scenes)]
    sample_inds = torch.stack(sample_inds, dim=0)
    return sample_inds
