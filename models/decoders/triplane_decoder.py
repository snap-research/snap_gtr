"""
Triplane decoder implementation
"""
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fields import BaseField, SeparateField

from utils.typing import *
from utils.config_utils import to_immutable_dict, PrintableConfig
from torch.utils.checkpoint import _get_autocast_kwargs


class TriplaneDecoder(nn.Module):
    """
    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
    """

    def __init__(
        self,
        in_dim: int = 120,
        num_layers: int = 8,
        layer_width: int = 64,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[str] = 'relu',
        sigma_activation: Optional[str] = 'softplus',
        interp_mode: Optional[str] = 'bilinear',
        sigmoid_saturation: Optional[float] = 0.001,
        hidden_dim_normal: Optional[int] = 64,
        use_pred_normal: Optional[bool] = False,
        field_type: str = "BaseField",
        field_args: Dict[str, Any] = to_immutable_dict(
            {
                "num_layers_sigma": 3
            }
        ),
    ) -> None:
        super().__init__()
        self.interp_mode = interp_mode
        self.use_pred_normal = use_pred_normal
        if field_type == "BaseField":
            self.field = BaseField(
                in_dim=in_dim,
                num_layers=num_layers,
                layer_width=layer_width,
                out_dim=out_dim,
                skip_connections=skip_connections,
                activation=activation,
                sigma_activation=sigma_activation,
                sigmoid_saturation=sigmoid_saturation,
                hidden_dim_normal=hidden_dim_normal,
                use_pred_normal=use_pred_normal,
            )
        elif field_type == "SeparateField":
            self.field = SeparateField(**field_args)
        else:
            raise NotImplementedError(f"Not Implemented: {field_type}")

    def xyz_transform(self, xyz):
        """Transform xyz 3d points to uv coordinates in each plane"""
        xy = xyz[..., :2]
        xz = xyz[..., ::2]
        yz = xyz[..., 1:]

        assert xyz.ndim == 3
        num_scenes, num_points, _ = xyz.size()
        out = torch.stack([xy, xz, yz], dim=1).reshape(num_scenes * 3, 1, num_points, 2)

        return out

    def point_decode(self, xyzs, code, density_only=False):
        """ Convert latent feature to RGB + density.
        Args:
            xyzs: Shape (num_scenes, (num_points_per_scene, 3))
            code: Shape (num_scenes, 3, n_channels, h, w)

        Returns:
            sigmas: Shape (num_scenes, (num_points_per_scene, 1))
            rgbs: Shape (num_scenes, (num_points_per_scene, 3))
        """
        num_scenes, _, n_channels, h, w = code.size()

        assert isinstance(xyzs, torch.Tensor), f"Only support xyzs as torch.tensor, same points across scenes"
        assert xyzs.dim() == 3  # (num_scenes, num_pts, 3)
        num_points = xyzs.size(-2)
        point_code = F.grid_sample(
            code.reshape(num_scenes * 3, -1, h, w),
            self.xyz_transform(xyzs),
            mode=self.interp_mode, padding_mode='border', align_corners=False
        ).reshape(num_scenes, 3, -1, num_points)
        point_code = point_code.permute(0, 3, 2, 1).reshape(num_scenes * num_points, -1)  #

        outputs = self.field(point_code, density_only=density_only)
        if "density" in outputs:
            outputs["density"] = outputs["density"].reshape(num_scenes, num_points, 1)
        if "rgb" in outputs:
            outputs["rgb"] = outputs["rgb"].reshape(num_scenes, num_points, 3)
        if "pred_normal" in outputs:
            outputs["pred_normal"] = outputs["pred_normal"].reshape(num_scenes, num_points, 3)
        return outputs

    def point_decode_color(self, xyzs, code):
        """ Convert latent feature to RGB + density.
                Args:
                    xyzs: Shape (num_scenes, (num_points_per_scene, 3))
                    code: Shape (num_scenes, 3, n_channels, h, w)

                Returns:
                    sigmas: Shape (num_scenes, (num_points_per_scene, 1))
                    rgbs: Shape (num_scenes, (num_points_per_scene, 3))
                """
        num_scenes, _, n_channels, h, w = code.size()

        assert isinstance(xyzs, torch.Tensor), f"Only support xyzs as torch.tensor, same points across scenes"
        assert xyzs.dim() == 3  # (num_scenes, num_pts, 3)
        num_points = xyzs.size(-2)
        
        point_code = F.grid_sample(
            code.reshape(num_scenes * 3, -1, h, w),
            self.xyz_transform(xyzs),
            mode=self.interp_mode, padding_mode='border', align_corners=False
        ).reshape(num_scenes, 3, -1, num_points)
        point_code = point_code.permute(0, 3, 2, 1).reshape(num_scenes * num_points, -1)  #

        rgbs = self.field.forward_color(point_code)
        return rgbs.reshape(num_scenes, num_points, 3)

    def point_density_decode(self, xyzs, code):
        outputs = self.point_decode(xyzs,  code, density_only=True)
        return outputs["density"]

    def visualize(self, code, scene_name, viz_dir, code_range=None):
        if code_range is None:
            code_range = [-1, 1]
        num_scenes, _, num_chn, h, w = code.size()
        code_viz = code.cpu().numpy()
        code_viz = code_viz.transpose(0, 1, 3, 2, 4).reshape(num_scenes, 3 * h, num_chn * w)
        for code_viz_single, scene_name_single in zip(code_viz, scene_name):
            out_file = f"{viz_dir}/scene_{scene_name_single}.jpg"
            plt.imsave(
                out_file,
                code_viz_single,
                vmin=code_range[0],
                vmax=code_range[1]
            )
