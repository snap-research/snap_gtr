import os
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from loguru import logger

import torch

from utils.distributed import get_rank


TrainerState = Dict[str, Any]


class CheckPointer:
    """
    Main class for checkpointing during training. Saves / loads checkpoint to disk
    If training job restarted, thanks to this callback training will continue from the last saved checkpoint
    Can resume from other training job partially
    Saves the following files:
    - "full_checkpoint.pth" - full state of trainer to restore training (models, optimizers, schedulers, etc)
    - "generator.pth" - only generator state
    - "generator.pth.ema" - only ema generator state (if no ema generator - this file is not saved)
    """
    CHECKPOINT_DIR = "ckpt"
    BACKUP_DIR = "backup"
    CKPT_FULL_STATE = "full_checkpoint.pth"

    def __init__(
        self,
        output_dir: Optional[str] = None,
        resume_path: Optional[str] = None,
        strict=True,
        resume_step_counter=True,
        resume_optimizer=True
    ):
        """ Manage load and save ckpt and auto-resume from lastest checkpoint.
        Args:
            output_dir (Optional[str]): if specified, root directory where to save checkpoints.
            resume_path (Optional[str]): if specified, should point to training checkpoint (full_checkpoint.pth) to
                load weights from on initialization stage. Useful if you want to resume from another training job.
                If this parameter is specified but job is trying to recover after failure the initialization will work
                from the latest backup checkpoint (i.e. this parameter is ignored if checkpoint directory is not empty)
            strict (bool): used in conjunction with resume_path only. if True checks that all keys in the loaded state
                are same as in the current trainer state and raises error if check did not pass
            resume_step_counter (bool): used in conjunction with resume_path only. if True continues step count, otherwise resets count
        """
        super().__init__()
        self.checkpoint_dir = Path(output_dir) / self.CHECKPOINT_DIR if output_dir is not None else None
        self.backup_dir = Path(output_dir) / self.BACKUP_DIR if output_dir is not None else None
        self.resume_path = Path(resume_path) if resume_path is not None else None
        self.strict = strict
        self.resume_step_counter = resume_step_counter
        self.resume_optimizer = resume_optimizer
        logger.info(f"Rank[{get_rank()}] init Checkpointer {self.resume_path} {self.resume_optimizer}")

    def on_train_start(self, trainer):
        resume_state, strict = self.load_initial_checkpoint()
        if resume_state is not None:
            trainer.load_state_dict(resume_state, strict=strict, resume_optimizer=self.resume_optimizer)

    def load_initial_checkpoint(self) -> Tuple[Optional[TrainerState], bool]:
        resume_path, strict = self._resolve_resume_path()
        state = None
        if resume_path is not None:
            state = torch.load(resume_path, map_location="cpu")  # cpu placement should prevent potential OOM
        return state, strict

    def _resolve_resume_path(self):
        resume_path = None
        strict = True

        if self.resume_path is not None:
            resume_path = self.resume_path
            strict = self.strict
            logger.info(f"Rank[{get_rank()}] Resume from another training run: {resume_path} strict {strict}")
        elif self.checkpoint_dir is not None and (self.checkpoint_dir / self.CKPT_FULL_STATE).exists():
            resume_path = self.checkpoint_dir / self.CKPT_FULL_STATE
            self.resume_step_counter = True
            logger.info(f"Rank[{get_rank()}] resume earlier training (broken / shutdown job) {resume_path}, strict {strict}, resume step_counter")
        else:
            logger.info(f"Rank[{get_rank()}] training from scratch")
        return resume_path, strict

    def _save_checkpoint_to(self, state: TrainerState, cp_dir: Path):
        cp_dir.mkdir(parents=True, exist_ok=True)
        self._safe_save_ckpt_to_file(state, str(cp_dir / self.CKPT_FULL_STATE))

    def _safe_save_ckpt_to_file(self, state: TrainerState, ckpt_file: str):
        """Try to save to a temporary file. If success, move to ckpt_file"""
        temp_file = f"{Path(ckpt_file).parent}/{Path(ckpt_file).stem}_temp.pth"
        assert not Path(temp_file).is_file()
        try:
            torch.save(state, temp_file)
            os.rename(temp_file, ckpt_file)
        except Exception as e:
            print("An unexpected error occurred:", str(e))
            if Path(temp_file).is_file():
                os.remove(temp_file)
            raise IOError(f"Fail to save file {ckpt_file}")

    def save_checkpoint(self, state: TrainerState, backup=True):
        logger.debug(f"Rank[{get_rank()}] save checkpoint to {self.checkpoint_dir}")
        if self.checkpoint_dir is not None:
            self._save_checkpoint_to(state=state, cp_dir=self.checkpoint_dir)
            # backup intermediate iterations
            if backup:
                logger.debug(f"Rank[{get_rank()}] save backup checkpoint to {self.backup_dir / str(state['step'])}")
                self._save_checkpoint_to(state=state, cp_dir=self.backup_dir / str(state["step"]))