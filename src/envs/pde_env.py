import numpy as np
from typing import Any, List, Optional, Tuple, Type

import gym
from phi.tf.flow import box, Domain, DomainState, FieldEffect, Physics
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn

from src.visualization import Viz

class PDEEnv(VecEnv):
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
        super().__init__(num_envs, observation_space, action_space)

        self.reward_range = (-float('inf'), float('inf'))
        self.spec = None
        self.exp_name = exp_name
        self.domain = domain
        self.dt = dt

        self.physics = physics_type(**physics_kwargs)

        self.step_cnt = step_cnt
        self.step_idx = 0
        self.epis_idx = 0

        self.final_reward_factor = final_reward_factor

        self.test_mode = False

        self.actions = None
        self.init_state = None
        self.cont_state = None
        self.goal_state = None
        self.pass_state = None

        self.viz = None

    def reset(self) -> VecEnvObs:
        self.step_idx = 0
        self.init_state = self._get_init_state()
        self.cont_state = self.init_state.copied_with()
        self.goal_state = self._get_goal_state()

        if self.test_mode:
            self._init_ref_states()

        return self._build_obs()

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = np.array(actions)

    def step_wait(self) -> VecEnvStepReturn:
        self.step_idx += 1
        trajectory_finished = self.step_idx == self.step_cnt

        forces = self._reshape_actions_to_forces()
        forces_field_effect = self._forces_to_field_effect(forces)

        self.cont_state = self._step_sim(self.cont_state, (forces_field_effect,))

        if self.test_mode:
            self._step_ref_states()

        obs = self._build_obs()
        rew = self._build_rew(forces)
        done = np.full((self.num_envs,), trajectory_finished)
        info = [{'rew_unnormalized': rew[i], 'forces': np.abs(forces[i]).sum()} for i in range(self.num_envs)]

        if trajectory_finished:
            self.epis_idx += 1
            rew += self._get_final_reward() * self.final_reward_factor
            obs = self.reset()

        return obs, rew, done, info

    def close(self) -> None:
        pass

    def render(self, mode: str='live') -> None:
        if not self.test_mode and self.viz is None:
            assert self.step_idx == 0, "Step idx is %i" % self.step_idx
            self.test_mode = True
            self._init_ref_states()
            self.viz = self.get_viz_for_mode(mode)

        self.viz.render(*self._get_fields_labels_colors())

    def seed(self, seed: Optional[int]=None) -> List[Optional[int]]:
        return [None for _ in range(self.num_envs)]

    def get_attr(self, attr_name: str, indices: VecEnvIndices=None) -> Any:
        return [getattr(self, attr_name) for _ in vec_env_indices_to_list(indices)]
    
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices=None, **method_kwargs) -> List[Any]:
        getattr(self, method_name)(*method_args, **method_kwargs)

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices=None) -> List[bool]:
        return [False for _ in vec_env_indices_to_list(indices)]

    def _init_ref_states(self):
        self.pass_state = self.init_state.copied_with()

    def _step_ref_states(self):
        self.pass_state = self._step_sim(self.pass_state, ())

    def _build_rew(self, forces):
        reshaped_forces = forces.reshape(forces.shape[0], -1)
        return -np.sum(reshaped_forces ** 2, axis=-1)

    def get_viz_for_mode(self, mode: str) -> Viz:
        if mode == 'live':
            return self._get_live_viz()
        elif mode == 'gif':
            return self._get_gif_viz()
        elif mode == 'png':
            return self._get_png_viz()
        else:
            raise NotImplementedError()

    def _get_init_state(self) -> DomainState:
        raise NotImplementedError()

    def _get_goal_state(self) -> DomainState:
        raise NotImplementedError()

    def _step_sim(self, in_state: DomainState, effects: Tuple[FieldEffect]) -> DomainState:
        raise NotImplementedError()

    def _build_obs(self):
        raise NotImplementedError()

    def _reshape_actions_to_forces(self) -> np.ndarray:
        raise NotImplementedError()

    def _forces_to_field_effect(self, forces: np.ndarray) -> FieldEffect:
        raise NotImplementedError()

    def _get_final_reward(self):
        raise NotImplementedError()

    def _get_live_viz(self):
        raise NotImplementedError()

    def _get_gif_viz(self):
        raise NotImplementedError()

    def _get_png_viz(self):
        raise NotImplementedError()

    def _get_fields_labels_colors(self):
        raise NotImplementedError()


def vec_env_indices_to_list(raw_indices: VecEnvIndices) -> List[int]:
    if raw_indices is None:
        return []
    if isinstance(raw_indices, int):
        return [raw_indices]
    return list(raw_indices)
