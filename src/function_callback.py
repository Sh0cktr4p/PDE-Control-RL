from stable_baselines3.common.callbacks import BaseCallback, EventCallback, EveryNTimesteps


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
