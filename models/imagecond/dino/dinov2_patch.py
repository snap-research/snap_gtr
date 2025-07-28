from dataclasses import dataclass

import torch
from torchvision import transforms

from utils.typing import *
from utils.config_utils import PrintableConfig


DINOv2_REPO = "facebookresearch/dinov2"

@dataclass
class DINOv2Config(PrintableConfig):
    model_name: str = "dinov2_vitb14"
    input_pixel_range: Tuple[float, float] = (0.0, 1.0)


class DINOv2(torch.nn.Module):
    def __init__(
        self,
        cfg: DINOv2Config,
    ):
        """
        DINOv2 model can be loaded by the model_name.
        See all available models by calling `DINOv2Model.get_available_clip_models()`.

        Args:
            model_name (str): architecture for DINOv2 model. Default: 'dinov2_vitl14'
            input_pixel_range (Tuple[float, float]): (min_value, max_value) input image range for proper normalization

        Example:
        >>> dino = DINOv2("dinov2_vitb14", input_pixel_range=(0.0, 1.0))
        >>> dino = dino.cpu()
        >>> images = torch.rand(2, 3, 512, 512)
        >>> features = dino(images)
        >>> features = dino.get_image_features(images) # Equivalent to the above
        """
        super().__init__()
        model_name = cfg.model_name
        input_pixel_range = cfg.input_pixel_range

        min_val, max_val = input_pixel_range

        self.preprocess = transforms.Compose(
            [
                transforms.Normalize(mean=[min_val] * 3, std=[max_val - min_val] * 3),
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
                ),
            ]
        )

        self.dino = torch.hub.load(DINOv2_REPO, model_name)
        self.dino.eval()

    @staticmethod
    def get_available_models() -> List[str]:
        """Returns list of all available DINOv2 models"""
        return torch.hub.list(DINOv2_REPO)

    @property
    def device(self):
        """torch.device where model is placed"""
        return next(self.dino.parameters()).device

    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Returns image features tensor of shape [batch_size, 1, C]"""
        images = self.preprocess(images)
        return self.dino(images).unsqueeze(dim=1)

    def get_image_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Returns image features tensor of shape [batch_size, 256, C]"""
        images = self.preprocess(images)
        res = self.dino.forward_features(images)
        return res['x_norm_patchtokens']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns image features tensor.
        For patch embedding_type, return (B, 256, C)
        """
        return self.get_image_patch_features(x)

