from termcolor import cprint
from loguru import logger
from abc import abstractmethod

from utils.distributed import get_rank


class BaseModel:
    def name(self):
        return 'BaseModel'

    @abstractmethod
    def initialize(self, device):
        self.device = device

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    @abstractmethod
    def get_current_visuals(self):
        pass

    @abstractmethod
    def get_current_errors(self):
        return {}

    def get_current_metrics(self):
        return None

    def set_optimizers(self):
        pass

    @abstractmethod
    def switch_eval(self):
        pass

    @abstractmethod
    def switch_train(self):
        pass

    @abstractmethod
    def state_dict(self) -> dict:
        pass

    @abstractmethod
    def load_state_dict(
        self,
        state: dict,
        strict: bool = True,
        resume_optimizer: bool = True
    ):
        pass

