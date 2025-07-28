"""
BaseNeRF:
    xyzs: torch.Size([8, 8388608, 3])
    code: torch.Size([8, 3, 40, 256, 256])
BaseNeRF
    output density: torch.Size([8, 8388608, 1]),
    output color: torch.Size([8, 8388608, 3])
"""
import torch
from collections import defaultdict
from loguru import logger

from utils.modeling_util import _get_autocast_kwargs



class DeferredBP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, code, chunk_size, decoder_model):
        """
        Forward rendering.
        """
        assert (xyz.dim() == 3) and (code.dim() == 5)
        ctx.save_for_backward(xyz, code)  # save tensors for backward
        ctx.decoder_model = decoder_model
        ctx.chunk_size = chunk_size

        ctx.gpu_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs()
        ctx.manual_seeds = []
        # logger.info(f"DeferredBP: xyz {xyz.dtype} ")

        with torch.no_grad(), torch.cuda.amp.autocast(
            **ctx.gpu_autocast_kwargs
        ), torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
            device = code.device
            batch_size, num_points = xyz.shape[:2]
            assert num_points % chunk_size == 0, f'chunk_size {chunk_size} must be divided by num_points {num_points}!'
            densities = torch.zeros(
                batch_size, num_points, 1, device=device, dtype=code.dtype,
            )
            colors = torch.zeros(
                batch_size, num_points, 3, device=device, dtype=code.dtype,
            )
            for i in range(batch_size):
                ctx.manual_seeds.append([])
                slice_code = code[i:(i + 1)]
                for j in range(0, num_points // ctx.chunk_size):
                    seed = torch.randint(0, 2 ** 32, (1,)).long().item()
                    ctx.manual_seeds[-1].append(seed)
                    # implement to get ray
                    slice_xyz = xyz[i:(i+1), j * ctx.chunk_size:(j + 1) * ctx.chunk_size]
                    # print(f"[{i}][{j}]: slice_xyz: {slice_xyz.dtype} {slice_code.dtype}")

                    results = ctx.decoder_model.point_decode(slice_xyz, slice_code)
                    densities[i:(i+1), j * ctx.chunk_size:(j + 1) * ctx.chunk_size] = results["density"]
                    colors[i:(i+1), j * ctx.chunk_size:(j + 1) * ctx.chunk_size] = results["rgb"]
                    # print(f"{results['density'].dtype} {results['rgb'].dtype} {densities.dtype} {colors.dtype}")

            # logger.info(f"DeferredBP forward: xyz {xyz.dtype} {densities.dtype} {colors.dtype}")
            return densities, colors

    @staticmethod
    def backward(ctx, grad_densities, grad_colors):
        """
        Backward process
        """
        xyz, code = ctx.saved_tensors

        xyz_nosync = xyz.detach().clone()
        xyz_nosync.requires_grad = True
        xyz_nosync.grad = None

        code_nosync = code.detach().clone()
        code_nosync.requires_grad = True
        code_nosync.grad = None
        # logger.info(f"DeferredBP backward start ")

        with torch.enable_grad(), torch.cuda.amp.autocast(
            **ctx.gpu_autocast_kwargs
        ), torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
            batch_size, num_points = xyz.shape[:2]

            for i in range(batch_size):
                ctx.manual_seeds.append([])
                slice_code = code_nosync[i:(i + 1)]

                for j in range(0, num_points // ctx.chunk_size):
                    grad_densities_slice = grad_densities[i:(i+1), j * ctx.chunk_size:(j + 1) * ctx.chunk_size]
                    grad_colors_slice = grad_colors[i:(i+1), j * ctx.chunk_size:(j + 1) * ctx.chunk_size]

                    seed = torch.randint(0, 2 ** 32, (1,)).long().item()
                    ctx.manual_seeds[-1].append(seed)
                    slice_xyz = xyz_nosync[i:(i + 1), j * ctx.chunk_size:(j + 1) * ctx.chunk_size]
                    results = ctx.decoder_model.point_decode(slice_xyz, slice_code)
                    density_slice = results["density"]
                    color_slice = results["rgb"]

                    render_slice = torch.cat([density_slice, color_slice], dim=-1)
                    grad_slice = torch.cat([grad_densities_slice, grad_colors_slice], dim=-1)
                    render_slice.backward(grad_slice)
        # logger.info(f"DeferredBP backward finish ")

        return xyz_nosync.grad, code_nosync.grad, None, None


def deferred_bp(
    xyz, code, chunk_size, decoder_model,
):
    return DeferredBP.apply(
        xyz, code, chunk_size, decoder_model,
    )


def forward_chunk_v1(xyzs, code, chunk_size, decoder_model):
    outputs_list = defaultdict(list)
    num_points = xyzs.shape[1]
    for i in range(0, num_points, chunk_size):
        slice_xyzs = xyzs[:, i:i+chunk_size]
        slice_outputs = decoder_model.point_decode(slice_xyzs, code)
        for key, value in slice_outputs.items():
            outputs_list[key].append(value)
    outputs = {key: torch.cat(value, dim=1) for key, value in outputs_list.items()}
    return outputs


def forward_chunk(xyz, code, chunk_size, decoder_model):
    batch_size, num_points = xyz.shape[:2]
    outputs_list = defaultdict(list)
    for i in range(batch_size):
        slice_code = code[i:(i + 1)]
        for j in range(0, num_points, chunk_size):
            slice_xyz = xyz[i:(i + 1), j: j + chunk_size]
            results = decoder_model.point_decode(slice_xyz, slice_code)
            for key, value in results.items():
                outputs_list[key].append(value)
    outputs = {key: torch.cat(value, dim=1) for key, value in outputs_list.items()}
    return outputs

