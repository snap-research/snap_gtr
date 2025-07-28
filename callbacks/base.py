from abc import abstractmethod
from typing import Iterable


class Callback:
    """
    Base class for custom manipulations during the training

    Overall methods are called during training loop in the following scheme:
    ```
    class Trainer:
        def train():
            callback.on_train_start()
            for step in ...:
                state: State = self._train_step(step)
                callback.on_step(self, step, state)
                if is_validation_step(step):
                    callback.on_valid_start(self, step)
                    for valid_step in ...:
                        callback.on_valid_step(self, valid_step, valid_state)
                    callback.on_valid_end(self, step)
            callback.on_train_end()
    ```
    """

    def on_train_start(self, trainer):
        pass

    def on_step(self, trainer, step):
        pass

    def on_train_end(self, trainer):
        pass

    def on_valid_start(self, trainer, step):
        pass

    def on_valid_step(self, trainer, valid_step):
        pass

    def on_valid_end(self, trainer, step):
        pass


class PeriodCallback(Callback):
    def __init__(self, period, early_sanity_iteration=3):
        if period < 1:
            raise ValueError(f"Expected positive value for period, got {period}")
        self.period = period
        self.early_sanity_iteration = early_sanity_iteration

    def is_step_selected(self, step, initial_step=0):
        return (step > 0 and step % self.period == 0) or (step == initial_step + self.early_sanity_iteration)

    def on_step(self, trainer, step):
        if self.is_step_selected(step=step, initial_step=trainer.initial_step):
            self._on_step(trainer, step)

    @abstractmethod
    def _on_step(self, trainer, step):
        pass

