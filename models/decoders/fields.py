"""
Fields convert point features to density and color
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import TruncExp
from utils.typing import *


def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class MLP(nn.Module):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Use layer width if None
        skip_connections: Add skip connections to input at specific layers
        activation: intermediate layer activation function
        out_activation: output activation function
    """
    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation

        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.in_dim, self.out_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self._skip_connections, "Skip connection at layer 0 doesn't make sense."
                    layers.append(nn.Linear(self.in_dim, self.layer_width))
                elif i in self._skip_connections:
                    layers.append(nn.Linear(self.layer_width + self.in_dim, self.layer_width))
                else:
                    layers.append(nn.Linear(self.layer_width, self.layer_width))
            layers.append(nn.Linear(self.layer_width, self.out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        """Process input with a multilayer perceptron"""
        # TODO: refactor BaseField with base_mlp, mlp_color(ray_dir as input), mlp_normal.
        x = in_tensor
        for i, layer in enumerate(self.layers):
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x


class BaseField(nn.Module):
    """ Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        sigma_activation: output activation function.
        sigmoid_saturation: color sigmoid saturation
    """
    activation_dict = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'sigmoid': nn.Sigmoid,
        'silu': nn.SiLU,
        'softplus': nn.Softplus,
        'tanh': nn.Tanh,
        'trunc_exp': TruncExp,
    }

    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[str] = 'relu',
        sigma_activation: Optional[str] = 'softplus',
        sigmoid_saturation: Optional[float] = 0.001,
        hidden_dim_normal: int = 64,
        use_pred_normal: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.activation = self.activation_dict[activation.lower()]()
        self.sigmoid_saturation = sigmoid_saturation
        self.use_pred_normal = use_pred_normal
        self.hidden_dim_normal = hidden_dim_normal

        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.in_dim, self.out_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self._skip_connections, "Skip connection at layer 0 doesn't make sense."
                    layers.append(nn.Linear(self.in_dim, self.layer_width))
                elif i in self._skip_connections:
                    layers.append(nn.Linear(self.layer_width + self.in_dim, self.layer_width))
                else:
                    layers.append(nn.Linear(self.layer_width, self.layer_width))
            layers.append(nn.Linear(self.layer_width, self.out_dim))
        self.layers = nn.ModuleList(layers)
        self.sigma_activation = self.activation_dict[sigma_activation.lower()]()
        self.color_activation = nn.Sigmoid()

        if self.use_pred_normal:
            self.mlp_normal = MLP(
                in_dim=self.layer_width,
                num_layers=3,
                layer_width=self.hidden_dim_normal,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=None,
            )
            self.normal_activation = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"], density_only=False):
        outputs = {}

        geo_embedding = None
        x = in_tensor
        for i, layer in enumerate(self.layers):
            # as checked  0 should not be in `_skip_connections`
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
            if self.use_pred_normal and i == len(self.layers) - 2:
                geo_embedding = x
        raw_sigma, raw_color = x.split([1, 3], dim=-1)
        sigmas = self.sigma_activation(raw_sigma)
        outputs["density"] = sigmas

        if not density_only:
            rgbs = self.color_activation(raw_color)
            if self.sigmoid_saturation > 0:
                rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation
            outputs["rgb"] = rgbs

        if not density_only and self.use_pred_normal:
            raw_normal = self.mlp_normal(geo_embedding)
            pred_normals = self.normal_activation(raw_normal)
            pred_normals = torch.nn.functional.normalize(pred_normals, dim=-1)
            outputs["pred_normal"] = pred_normals
        return outputs


class BaseFactoField(nn.Module):
    """ Slightly modified BaseField.
    MLP_base -> density + geometry_embedding
    MLP_head -> rgb
    MLP_normal -> normal

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        geo_feat_dim: output geo feature dimension
        hidden_dim_color: dimension of hidden layers for color mlp
        hidden_dim_normal: dimension of hidden layers for normal mlp
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        sigma_activation: output activation function.
        sigmoid_saturation: color sigmoid saturation
    """
    activation_dict = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'sigmoid': nn.Sigmoid,
        'silu': nn.SiLU,
        'softplus': nn.Softplus,
        'tanh': nn.Tanh,
        'trunc_exp': TruncExp,
    }
    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        geo_feat_dim: int,
        out_dim: Optional[int] = None,
        hidden_dim_color: int = 64,
        hidden_dim_normal: int = 64,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[str] = 'relu',
        sigma_activation: Optional[str] = 'softplus',
        sigmoid_saturation: Optional[float] = 0.001,
        use_pred_normal: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.geo_feat_dim = geo_feat_dim
        self.hidden_dim_color = hidden_dim_color
        self.hidden_dim_normal = hidden_dim_normal
        self.activation = self.activation_dict[activation.lower()]()
        self.sigmoid_saturation = sigmoid_saturation
        self.sigma_activation = self.activation_dict[sigma_activation.lower()]()
        self.use_pred_normal = use_pred_normal

        
        # MLP_base for density + geometry_embedding
        self.mlp_base = MLP(
            in_dim=in_dim,
            num_layers=num_layers,
            layer_width=layer_width,
            out_dim=1+geo_feat_dim,
            skip_connections=skip_connections,
            activation=self.activation,
            out_activation=None,
        )

        # mlp_head for rgb
        self.mlp_head = MLP(
            in_dim=self.geo_feat_dim,
            num_layers=3,
            layer_width=self.hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=None,
        )
        self.color_activation = nn.Sigmoid()

        # mlp_normal for normal
        if self.use_pred_normal:
            self.mlp_normal = MLP(
                in_dim=geo_feat_dim,
                num_layers=3,
                layer_width=self.hidden_dim_normal,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=None,
            )
            self.normal_activation = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"], density_only=False):
        outputs = {}

        h = self.mlp_base(in_tensor)
        raw_sigma, base_mlp_out = h.split([1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = raw_sigma
        sigmas = self.sigma_activation(raw_sigma)
        outputs["density"] = sigmas

        if not density_only:
            raw_rgb = self.mlp_head(base_mlp_out)
            rgbs = self.color_activation(raw_rgb)
            if self.sigmoid_saturation > 0:
                rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation
            outputs["rgb"] = rgbs

        if self.use_pred_normal:
            raw_normal = self.mlp_normal(base_mlp_out)
            pred_normals = self.normal_activation(raw_normal)
            pred_normals = torch.nn.functional.normalize(pred_normals, dim=-1)
            outputs["pred_normal"] = pred_normals

        return outputs


class SeparateField(nn.Module):
    """
    Separate MLP for each property
    MLP_density
    MLP_color

    Args:
        in_dim: Input layer dimension
        num_layers_sigma: Number of linear layers for sigma
        num_layers_color: Number of linear layers for color
        layer_width: hidden width of each MLP layer
        activation: intermediate layer activation function.
        sigma_activation: output activation function for sigma
        sigmoid_saturation: by color sigmoid saturation
    """
    activation_dict = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'sigmoid': nn.Sigmoid,
        'silu': nn.SiLU,
        'softplus': nn.Softplus,
        'tanh': nn.Tanh,
        'trunc_exp': TruncExp,
        'gelu': nn.GELU,
    }
    def __init__(
        self,
        in_dim: int,
        num_layers_sigma: int,
        num_layers_color: int,
        layer_width: int,
        activation: Optional[str] = 'relu',
        sigma_activation: Optional[str] = 'softplus',
        sigmoid_saturation: Optional[float] = 0.001,
    ):
        super().__init__()
        print(f"SeparateField: \n"
              f"\t in_dim: {in_dim}\n"
              f"\t num_layers_sigma: {num_layers_sigma}\n"
              f"\t num_layers_color: {num_layers_color}\n"
              f"\t layer_width: {layer_width}\n"
              f"\t activation: {activation}\n"
              f"\t sigma_activation: {sigma_activation}\n"
              f"\t sigmoid_saturation: {sigmoid_saturation}")
        self.activation = self.activation_dict[activation.lower()]()
        self._density_before_activation = None

        self.mlp_sigma = MLP(
            in_dim=in_dim,
            num_layers=num_layers_sigma,
            layer_width=layer_width,
            out_dim=1,
            skip_connections=None,
            activation=self.activation,
            out_activation=None,
        )
        self.sigma_activation = self.activation_dict[sigma_activation.lower()]()

        self.mlp_color = MLP(
            in_dim=in_dim,
            num_layers=num_layers_color,
            layer_width=layer_width,
            out_dim=3,
            activation=self.activation,
            out_activation=None,
        )
        self.color_activation = nn.Sigmoid()
        self.sigmoid_saturation = sigmoid_saturation

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"], density_only=False):
        outputs = {}

        raw_sigma = self.mlp_sigma(in_tensor)
        self._density_before_activation = raw_sigma
        sigmas = self.sigma_activation(raw_sigma)
        outputs["density"] = sigmas

        if not density_only:
            raw_rgb = self.mlp_color(in_tensor)
            rgbs = self.color_activation(raw_rgb)
            if self.sigmoid_saturation > 0:
                rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation
            outputs["rgb"] = rgbs

        return outputs

    def forward_density(self, in_tensor: Float[Tensor, "*bs in_dim"]):
        raw_sigma = self.mlp_sigma(in_tensor)
        self._density_before_activation = raw_sigma
        sigmas = self.sigma_activation(raw_sigma)
        return sigmas

    def forward_color(self, in_tensor: Float[Tensor, "*bs in_dim"]):
        raw_rgb = self.mlp_color(in_tensor)
        rgbs = self.color_activation(raw_rgb)
        if self.sigmoid_saturation > 0:
            rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation
        return rgbs


