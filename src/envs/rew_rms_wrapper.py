from typing import Optional
import numpy as np

from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper, VecEnvObs, VecEnvStepReturn

class RewRmsWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, rew_rms: Optional[RunningMeanStd]):
        super().__init__(venv)
        self.rew_rms = rew_rms

    def step_wait(self) -> VecEnvStepReturn:
        obs, rew, done, info = self.venv.step_wait()

        self.rew_rms.update(rew)
        norm_rew = (rew - self.rew_rms.mean) / np.sqrt(self.rew_rms.var)

        return obs, norm_rew, done, info

    def reset(self) -> VecEnvObs:
        return self.venv.reset()