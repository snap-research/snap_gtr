"""
Build mdoels
"""
from utils.io_utils import EasyDict
from utils.distributed import get_rank


def build_model(cfg, device):
    cfg = EasyDict(cfg)
    if cfg.name == "BaseNeRF":
        from models.base_nerf import BaseNeRF, BaseNeRFModelConfig
        model_config = BaseNeRFModelConfig(**cfg.args)
        model = BaseNeRF(model_config, device=device)
    elif cfg.name == "TriPlaneEncoding":
        from models.decoders.encodings import TriPlaneEncoding, TriPlaneEncodingConfig
        model_config = TriPlaneEncodingConfig(**cfg.args)
        model = TriPlaneEncoding(model_config)
    elif cfg.name == "VoxelEncoding":
        from models.decoders.encodings import VoxelEncoding, VoxelEncodingConfig
        model_config = VoxelEncodingConfig(**cfg.args)
        model = VoxelEncoding(model_config)
    elif cfg.name == "TransformerUNetV1":
        from models.transformers.networks import TransformerUNetV1Config, TransformerUNetV1
        model_config = TransformerUNetV1Config(**cfg.args)
        model = TransformerUNetV1(model_config)
    elif cfg.name == "TransformerV1":
        # pure attention layers
        from models.transformers.networks import TransformerV1Config, TransformerV1
        model_config = TransformerV1Config(**cfg.args)
        model = TransformerV1(model_config)
    elif cfg.name == "TransformerV2":
        from models.transformers.models import TransformerV2Config, TransformerV2
        model_config = TransformerV2Config(**cfg.args)
        model = TransformerV2(model_config)
    elif cfg.name == "transformer_dinov2":
        from models.imagecond.dino.transformer_dinov2 import CustomDino, CustomDinoConfig
        model_config = CustomDinoConfig(**cfg.args)
        model = CustomDino(model_config)
    elif cfg.name == "transformer_dinov1":
        from models.imagecond.dino.transformer_dinov1 import DinoWrapper, CustomDinoConfig
        model_config = CustomDinoConfig(**cfg.args)
        model = DinoWrapper(model_config)
    elif cfg.name == "dinov2_patch":
        from models.imagecond.dino.dinov2_patch import DINOv2, DINOv2Config
        model_config = DINOv2Config(**cfg.args)
        model = DINOv2(cfg=model_config)
    elif cfg.name == "dinov2_custom":
        from models.imagecond.dino.dinov2_custom import DINOv2, DINOv2Config
        model_config = DINOv2Config(**cfg.args)
        model = DINOv2(cfg=model_config)
    elif cfg.name == "GeneratorV2":
        from models.transformers.generator_v2 import ModelConfig, GeneratorV2
        model_config = ModelConfig(**cfg.args)
        model = GeneratorV2(cfg=model_config)
    elif cfg.name == "GeneratorV3":
        from models.transformers.generator_v3 import ModelConfig, GeneratorV3
        model_config = ModelConfig(**cfg.args)
        model = GeneratorV3(cfg=model_config)
    elif cfg.name == "MarchingCubeGeo":
        from models.mesh_decoders.mc_geometry import MCGeometryConfig, MCGeometry
        model_config = MCGeometryConfig(**cfg.args)
        model = MCGeometry(device=device, cfg=model_config)
    elif cfg.name == "MeshRenderLoss":
        from models.losses import MeshRenderLoss, MeshRenderLossConfig
        model_config = MeshRenderLossConfig(**cfg.args)
        model = MeshRenderLoss(device=device, cfg=model_config)
    else:
        raise NotImplementedError(f"Cannot find model {cfg.anme}")

    if get_rank() == 0:
        print(model_config)
    return model

