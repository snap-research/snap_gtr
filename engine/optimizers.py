"""
Optimizers class.
"""
from __future__ import annotations

from dataclasses import dataclass
from loguru import logger

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parameter import Parameter

from .schedulers import MultiStepSchedulerConfig, ExponentialDecaySchedulerConfig, CosineDecaySchedulerConfig
from utils.config_utils import PrintableConfig
from utils.io_utils import EasyDict
from utils.typing import *
from utils.distributed import get_rank


# Optimizer related configs
@dataclass
class OptimizerConfig(PrintableConfig):
    """Basic optimizer config with RAdam"""

    _target: Type = torch.optim.Adam
    """The optimizer class to use."""
    lr: float = 0.0005
    """The learning rate to use."""
    eps: float = 1e-08
    """The epsilon value to use."""
    max_norm: Optional[float] = None
    """The max norm to use for gradient clipping."""

    def setup(self, params) -> torch.optim.Optimizer:
        """Returns the instantiated object using the config."""
        if get_rank() == 0:
            print(f"Setup optimizer ", self)
        kwargs = vars(self).copy()
        kwargs.pop("_target")
        kwargs.pop("max_norm")

        return self._target(params, **kwargs)


@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with Adam"""

    _target: Type = torch.optim.Adam
    """The optimizer class to use."""
    weight_decay: float = 0
    """The weight decay to use."""
    betas: Tuple[float, ...] = (0.9, 0.999)
    """The beta value to use"""


@dataclass
class AdamWOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with Adam"""

    _target: Type = torch.optim.AdamW
    """The optimizer class to use."""
    weight_decay: float = 0
    """The weight decay to use."""
    betas: Tuple[float, ...] = (0.9, 0.999)
    """The beta value to use"""


def parse_optimizer_config(config):
    config = EasyDict(config)
    if config.name == "AdamOptimizerConfig":
        optim_cfg = AdamOptimizerConfig(**config.args)
    elif config.name == "AdamWOptimizerConfig":
        optim_cfg = AdamWOptimizerConfig(**config.args)
    else:
        raise NotImplementedError(f"Do not support optimizer {config.name}")
    return optim_cfg


def parse_scheduler_config(config):
    config = EasyDict(config)
    if config.name == "MultiStepSchedulerConfig":
        sched_cfg = MultiStepSchedulerConfig(**config.args)
    elif config.name == "ExponentialDecaySchedulerConfig":
        sched_cfg = ExponentialDecaySchedulerConfig(**config.args)
    elif config.name == "CosineDecaySchedulerConfig":
        sched_cfg = CosineDecaySchedulerConfig(**config.args)
    else:
        raise NotImplementedError(f"Fail to find scheduler {config.name}")
    return sched_cfg


def check_param_list_type(input_list: List[Union[Parameter, Dict]]) -> str:
    """
    Optimizers support
     - a list of Parameter [var1, var2]
     - a list of dict, [{'parmas': }, {'params': }], each dict should contain a 'params' key
    """
    assert isinstance(input_list, list)
    if all(isinstance(param, Parameter) for param in input_list):
        return "param_list"
    elif all(isinstance(param_group, dict) for param_group in input_list):
        assert all("params" in param_group for param_group in input_list)
        return "param_group_list"
    return "Unknown type"


def get_params(input_list: List[Union[Parameter, Dict]]) -> List[Parameter]:
    input_type = check_param_list_type(input_list)
    if input_type == "param_list":
        # logger.debug(f"Input param is param_list, return as it is ")
        return input_list
    elif input_type == "param_group_list":
        # logger.debug(f"Input param is param_group, convert to param_list ")
        parameters = []
        for param_group in input_list:
            parameters += param_group['params']
        return parameters
    else:
        raise NotImplementedError(f"Unknown input type")


def log_optim_info(optim):
    for group_idx, param_group in enumerate(optim.param_groups):
        message = f"param_group[{group_idx}], contain {len(param_group['params'])} params \n "
        for key, value in param_group.items():
            if key == 'params':
                continue
            message += f"\t {key}: {str(value)} \n"
        print(message)


class Optimizers:
    """Optimizers handles optimizer, schedulers and parameters

    Args:
        config: The optimizer configuration.
        param_groups: A dictionary of parameter groups to optimize.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        param_groups: Dict[str, List[Union[Parameter, Dict]]]
    ) -> None:
        self.config = config
        # Optimize
        self.optimizers = {}
        self.schedulers = {}
        self.parameters = {}
        for param_group_name, params in param_groups.items():
            # Print some nice warning messages if the user forgot to specify an optimizer
            if param_group_name not in config:
                raise RuntimeError(
                    f"""Optimizer config for '{param_group_name}' not found in config file. Make sure you specify an optimizer for each parameter group. Provided configs were: {config.keys()}"""
                )
            # build optimizer
            
            optimizer_config = parse_optimizer_config(config[param_group_name]["optimizer"])
            
            self.optimizers[param_group_name] = optimizer_config.setup(params=params)
            self.config[param_group_name]["optimizer"] = optimizer_config  # update default parameter to config

            # get list of parameters
            self.parameters[param_group_name] = get_params(params)  # support both list of param and list of param_group

            # build scheduler
            if config[param_group_name]["scheduler"]:
                lr_init = optimizer_config.lr
                scheduler_config = parse_scheduler_config(config[param_group_name]["scheduler"])
                self.schedulers[param_group_name] = (
                    scheduler_config
                    .setup()
                    .get_scheduler(optimizer=self.optimizers[param_group_name], lr_init=lr_init)
                )
                # update default parameters to config
                self.config[param_group_name]["scheduler"] = scheduler_config

    def zero_grad_all(self) -> None:
        """Zero the gradients for all optimizer parameters."""
        for _, optimizer in self.optimizers.items():
            optimizer.zero_grad()

    def optimizer_scaler_step_all(self, grad_scaler: GradScaler) -> None:
        """Take an optimizer step using a grad scaler.

        Args:
            grad_scaler: GradScaler to use
        """
        for param_group, optimizer in self.optimizers.items():
            max_norm = self.config[param_group]["optimizer"].max_norm
            if max_norm is not None:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters[param_group], max_norm)
            if any(any(p.grad is not None for p in g["params"]) for g in optimizer.param_groups):
                grad_scaler.step(optimizer)

    def optimizer_step_all(self) -> None:
        """Run step for all optimizers."""
        for param_group, optimizer in self.optimizers.items():
            # note that they key is the parameter name
            max_norm = self.config[param_group]["optimizer"].max_norm
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters[param_group], max_norm)
            optimizer.step()

    def scheduler_step_all(self) -> None:
        """Run step for all schedulers.

        Args:
            step: the current step
        """
        for param_group_name, scheduler in self.schedulers.items():
            scheduler.step()

    def get_lr_all(self) -> Dict[str, float]:
        """Return learning rates for all"""
        lr_dict = {}
        for param_group_name, optimizer in self.optimizers.items():
            lr = optimizer.param_groups[0]['lr']
            lr_dict[param_group_name] = lr
        return lr_dict

    def load_optimizers(self, loaded_state: Dict[str, Any]) -> None:
        """Helper to load the optimizer state from previous checkpoint

        Args:
            loaded_state: the state from the previous checkpoint
        """
        for k, v in loaded_state.items():
            self.optimizers[k].load_state_dict(v)
        logger.debug(f"Rank[{get_rank()}] success load optimizer state_dict")

    def optimizers_state_dict(self, ) -> Dict[str, Any]:
        state_dict = {}
        for param_group_name, optimizer in self.optimizers.items():
            logger.debug(f"save_optimizers param_group_name: {param_group_name}")
            state_dict[param_group_name] = optimizer.state_dict()
        return state_dict

    def load_schedulers(self, loaded_state: Dict[str, Any]) -> None:
        """Helper to load the scheduler state from previous checkpoint

        Args:
            loaded_state: the state from the previous checkpoint
        """
        for k, v in loaded_state.items():
            self.schedulers[k].load_state_dict(v)

    def schedulers_state_dict(self,) -> Dict[str, Any]:
        state_dict = {}
        for param_group_name, scheduler in self.schedulers.items():
            logger.debug(f"save_schedulers param_group_name: {param_group_name}")
            state_dict[param_group_name] = scheduler.state_dict()
        return state_dict

