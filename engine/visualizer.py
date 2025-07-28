from pathlib import Path
from termcolor import colored
from PIL import Image
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from utils.typing import *


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


class Visualizer:
    def __init__(self, expr_dir, job_name, world_size):
        self.job_name = job_name
        self.num_gpu = world_size

        # images
        self.img_dir = f"{expr_dir}/images"
        Path(self.img_dir).mkdir(exist_ok=True, parents=True)
        # tensorboard
        self.tb_dir = f"{expr_dir}/tboard"
        Path(self.tb_dir).mkdir(exist_ok=True, parents=True)
        self.writer = SummaryWriter(log_dir=self.tb_dir)
        # logs
        self.log_name = f"{expr_dir}/loss_log.txt"
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(f'================ Training Loss ({now}) ================\n')

    def put_visual_dict(self, name: str, visual_dict: Dict[str, Any], step: int):
        """Assume img_np in RGB, range [0, 255]"""
        for label, img_np in visual_dict.items():
            assert len(img_np.shape) == 3
            img_np = img_np[:, :, :3]
            self.writer.add_image(f"{name}/{label}", img_np, global_step=step, dataformats='HWC')

    def put_visual_dict_to_file(self, name: str, visual_dict: Dict[str, Any], step: int):
        for label, img_np in visual_dict.items():
            img_path = f"{self.img_dir}/{name}_step{step:05d}_{label}.jpg"
            img_pil = Image.fromarray(img_np)
            img_pil.save(img_path)

    def put_scalar(self, name: str, scalar: Any, step: int):
        self.writer.add_scalar(name, scalar, step)

    def put_dict(self, name: str, scalar_dict: Dict[str, Any], step: int):
        for key, value in scalar_dict.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
                value = value.float()  # bf16 -> numpy is not supported
            self.writer.add_scalar(f"{name}/{key}", value, step)

    def put_messsage_to_file(self, message):
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
