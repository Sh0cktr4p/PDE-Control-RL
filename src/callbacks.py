import sys
from typing import Callable, List

from stable_baselines3.common.utils import safe_mean
from one_line_output_format import OneLineOutputFormat
from stable_baselines3.common.callbacks import BaseCallback, EventCallback, EveryNTimesteps
from stable_baselines3.common import logger
import time
import numpy as np


class EveryNRolloutsFunctionCallback(EventCallback):
    def __init__(self, n_rollouts, callback_fn):
        super().__init__(FunctionCallback(callback_fn))
        self.rollout_idx = 0
        self.n_rollouts = n_rollouts

    def _on_rollout_end(self):
        if self.rollout_idx % self.n_rollouts == self.n_rollouts - 1:
            self._on_event()
        self.rollout_idx += 1


class EveryNRolloutsPlusStartFinishFunctionCallback(EveryNRolloutsFunctionCallback):
    def __init__(self, n_rollouts, callback_fn):
        super().__init__(n_rollouts, callback_fn)

    def _on_training_start(self):
        self._on_event()

    def _on_training_end(self):
        self._on_event()


class EveryNTimestepsFunctionCallback(EveryNTimesteps):
    def __init__(self, n_steps, callback_fn):
        super().__init__(n_steps, FunctionCallback(callback_fn))


class FunctionCallback(BaseCallback):
    def __init__(self, callback_fn):
        super().__init__()
        self.callback_fn = callback_fn

    def _on_step(self):
        self.callback_fn(self.n_calls)
        return True


class TimeConsumptionMonitorCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        print('Called super constructor')
        self.fwd_times = []
        self.bwd_times = []
        self.rollout_start_timestamp = 0
        self.rollout_end_timestamp = 0

    def _on_training_start(self):
        self.rollout_start_timestamp = time.time()
        self.rollout_end_timestamp = time.time()

    def _on_rollout_start(self):
        print(self.rollout_end_timestamp - self.rollout_start_timestamp)
        first_rollout = self.rollout_end_timestamp - self.rollout_start_timestamp < 1e-4
        self.rollout_start_timestamp = time.time()
        if not first_rollout:
            self.bwd_times.append(self.rollout_start_timestamp - self.rollout_end_timestamp)
        else:
            print("not storing time data on first rollout")

    def _on_rollout_end(self):
        self.rollout_end_timestamp = time.time()
        self.fwd_times.append(self.rollout_end_timestamp - self.rollout_start_timestamp)
        self.print_statistics()

    def _on_step(self):
        return True

    def print_statistics(self):
        avg_fwd_time = np.average(self.fwd_times)
        avg_bwd_time = np.average(self.bwd_times)
        fwd_to_bwd_ratio = avg_fwd_time / avg_bwd_time
        print("average rollout collection time: %f" % avg_fwd_time)
        print("average update time: %f" % avg_bwd_time)
        print("ratio forward to backward: %f" % fwd_to_bwd_ratio)
        logger.record("rollout collection time average", avg_fwd_time)
        logger.record("learning time average", avg_bwd_time)
        logger.record("ratio collection to learning", fwd_to_bwd_ratio)


class CustomLoggerInjectionCallback(BaseCallback):
    def _on_training_start(self):
        # Check if output format is already present
        for format in self.logger.output_formats:
            if isinstance(format, OneLineOutputFormat):
                return

        self.logger.output_formats.append(OneLineOutputFormat(sys.stdout))

    def _on_step(self) -> bool:
        return True


class RecordScalarCallback(BaseCallback):
    def __init__(self, scalar_name: str, scalar_fn: Callable[[], float]):
        self.scalar_name = scalar_name
        self.scalar_fn = scalar_fn
        super().__init__()
    
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        value = self.scalar_fn()
        self.logger.record(self.scalar_name, value)


class RecordInfoScalarsCallback(BaseCallback):
    def __init__(self, *scalar_names):
        self.scalar_names = scalar_names
        super().__init__()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            for scalar_name in self.scalar_names:
                scalar_value = safe_mean([ep_info[scalar_name] for ep_info in self.model.ep_info_buffer])
                self.logger.record("rollout/" + scalar_name, scalar_value)
