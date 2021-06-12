from matplotlib.pyplot import step
import numpy as np
from typing import Any, List, Optional, Tuple, Type

import gym
from phi.tf.flow import box, Domain, DomainState, FieldEffect, Physics
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from src.envs.pde_env import PDEEnv

class PDEReconstructEnv(PDEEnv):
    metadata = {'render.modes': ['live', 'gif', 'png']}

    def __init__(
        self,
        physics_type: Type[Physics],
        physics_kwargs: dict,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        num_envs: int,
        domain: Domain,
        step_cnt: int,
        dt: float,
        final_reward_factor: float,
        exp_name: str='v0',
    ):
        super().__init__(
            physics_type,
            physics_kwargs,
            observation_space,
            action_space,
            num_envs,
            domain,
            step_cnt,
            dt,
            final_reward_factor,
            exp_name,
        )

        self.gt_forces = None
        self.gt_state = None

    def reset(self) -> VecEnvObs:
        self.gt_forces = self._get_gt_forces()
        return super().reset()

    def _get_goal_state(self) -> DomainState:
        state = self.init_state.copied_with()
        for _ in range(self.step_cnt):
            state = self._step_sim(state, (self.gt_forces,))
        return state

    def _init_ref_states(self):
        super()._init_ref_states()
        self.gt_state = self.init_state.copied_with()

    def _step_ref_states(self):
        super()._step_ref_states()
        self.gt_state = self._step_gt()
    
    def _step_gt(self) -> DomainState:
        return self._step_sim(self.gt_state, (self.gt_forces,))

    def _get_gt_forces(self) -> FieldEffect:
        raise NotImplementedError()


def vec_env_indices_to_list(raw_indices: VecEnvIndices) -> List[int]:
    if raw_indices is None:
        return []
    if isinstance(raw_indices, int):
        return [raw_indices]
    return list(raw_indices)