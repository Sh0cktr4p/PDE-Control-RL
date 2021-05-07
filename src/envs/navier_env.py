import numpy as np
from typing import Any, List, Optional, Type

import gym
from phi.tf.flow import box, Domain
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn

class NavierEnv(VecEnv):
    metadata = {'render.modes': ['live', 'gif', 'png']}

    def __init__(
        self,
        num_envs: int,
        step_count: int=32,
        domain: Domain=Domain((32, 32), box=box[0:1]),
        dt: float=0.03,
        exp_name: str='v0',
    ):
        self.step_idx = 0


    def reset(self) -> VecEnvObs:
        self.step_idx = 0
        self.gt_forces = self._get_gt_forces()

    def step_async(self, actions: np.ndarray) -> None:
        pass

    def step_wait(self) -> VecEnvStepReturn:
        pass

    def close(self) -> None:
        pass

    def render(self, mode: str='live') -> None:
        pass

    def seed(self, seed: Optional[int]=None) -> List[Optional[int]]:
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices=None) -> Any:
        pass
    
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices=None):
        pass

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices=None):
        pass

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices=None) -> List[bool]:
        pass