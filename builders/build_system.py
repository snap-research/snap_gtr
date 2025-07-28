from termcolor import cprint

from utils.distributed import get_rank


def build_system(config, device, world_size: int):
    if config.system_type == "MeshGen":
        from systems.MeshGen import MeshGen
        system = MeshGen(cfg=config, device=device, world_size=world_size)
    else:
        raise NotImplementedError(f"Do not support system {config.name}")
    return system
