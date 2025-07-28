from dataclasses import dataclass
from loguru import logger

import torch
from torchvision import transforms
import torch.nn as nn

from utils.typing import *
from utils.config_utils import PrintableConfig, to_immutable_dict

from transformers import Dinov2Model


@dataclass
class CustomDinoConfig(PrintableConfig):
    model_name: str = "facebook/dinov2-base"
    """Model name """
    input_channels: int = 3
    """Number of channels of model input """
    gradient_checkpointing: bool = True
    """If true, apply gradient checkpointing"""
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = to_immutable_dict(
        {"use_reentrant": False}
    )
    """Gradient checkpointing arguments"""
    crop_size: int = 224
    """Default center crop size. Dinov2 used a patch of 14x14. The crop size is suggested to be integer times of 14."""


class CustomDino(torch.nn.Module):
    def __init__(
        self,
        cfg: CustomDinoConfig,
    ):
        super().__init__()
        model_name = cfg.model_name
        input_channels = cfg.input_channels
        gradient_checkpointing = cfg.gradient_checkpointing
        gradient_checkpointing_kwargs = cfg.gradient_checkpointing_kwargs
        crop_size = cfg.crop_size

        # project input image channel to 3
        self.in_layers = nn.Sequential()
        embed_dim = 3
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, embed_dim, kernel_size=3, padding=1)
        )

        # !!! Rescale need change camera pucker accordingly. Here we avoid rescaling.
        # DINOV2 naively support interpolate positional encoding
        self.preprocess = transforms.Compose(
            [
                transforms.CenterCrop(crop_size),
            ]
        )

        # pretrained dinov2 model
        model = Dinov2Model.from_pretrained(model_name)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        else:
            model.gradient_checkpointing_disable()
        self.gradient_checkpointing = gradient_checkpointing
        self.dino = model

    @property
    def device(self):
        """torch.device where model is placed"""
        return next(self.dino.parameters()).device

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self.gradient_checkpointing

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """
        # TODO: refactor num_parameters to be generic

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]
            total_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
        else:
            total_parameters = list(self.parameters())

        total_numel = []
        for param in total_parameters:
            if param.requires_grad or not only_trainable:
                total_numel.append(param.numel())

        return sum(total_numel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            output: torch.tensor (B, 1+256, C), class_feature + patch_feature
        """
        x = self.preprocess(x)  # center crop to (224, 224)
        x = self.input_proj(x)

        output = self.dino(x)['last_hidden_state']

        return output
