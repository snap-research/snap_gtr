from dataclasses import dataclass
from loguru import logger

import torch
import torch.nn as nn

from utils.typing import *
from utils.config_utils import PrintableConfig, to_immutable_dict

from transformers import ViTModel

@dataclass
class CustomDinoConfig(PrintableConfig):
    model_name: str = "facebook/dino-vitb16"
    """Model name """
    input_channels: int = 3
    """Number of channels of model input """
    gradient_checkpointing: bool = True
    """If true, apply gradient checkpointing"""
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = to_immutable_dict(
        {"use_reentrant": False}
    )
    """Gradient checkpointing arguments"""
    freeze: bool = False
    """If true, freeze model"""


class DinoWrapper(torch.nn.Module):
    """
    Dinov1 wrapper using huggingface transformer implementation
    """
    def __init__(
        self,
        cfg: CustomDinoConfig,
    ):
        super().__init__()
        model_name = cfg.model_name
        input_channels = cfg.input_channels
        gradient_checkpointing = cfg.gradient_checkpointing
        gradient_checkpointing_kwargs = cfg.gradient_checkpointing_kwargs
        freeze = cfg.freeze

        # project input image channel to 3
        self.in_layers = nn.Sequential()
        embed_dim = 3
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, embed_dim, kernel_size=3, padding=1)
        )

        model = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        else:
            model.gradient_checkpointing_disable()
        self.gradient_checkpointing = gradient_checkpointing
        self.model = model

        if freeze:
            self._freeze()

    def _freeze(self):
        print(f"======== Freezing DinoWrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @property
    def device(self):
        """torch.device where model is placed"""
        return next(self.dino.parameters()).device

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self.gradient_checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        outputs = self.model(x, interpolate_pos_encoding=True)

        return outputs['last_hidden_state']
