import phi.tf.flow as phiflow
import gym
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.running_mean_std import RunningMeanStd
import numpy as np
from typing import Any, List, Optional, Tuple, Type, Union
from gym_phiflow.envs.visualization import *

from gym_phiflow.envs.burgers_util import *

class BurgersEnv(VecEnv):
    metadata = {'render.modes': ['l', 'f']}

    def __init__(
        self, num_envs: int, step_count: int=32, field_shape: Tuple[int, ...]=(32,), 
        dt: float=0.03, viscosity: float=0.003, diffusion_substeps: int=1,
        final_reward_factor: float=32, exp_name: str='v0'
    ):
        act_shape = self._get_act_shape(field_shape)
        obs_shape = self._get_obs_shape(field_shape)
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        action_space = gym.spaces.Box(-np.inf, np.inf, shape=act_shape, dtype=np.float32)

        super().__init__(num_envs, observation_space, action_space)

        self.reward_range = (-float('inf'), float('inf'))
        self.spec = None
        self.exp_name = exp_name
        self.domain = phiflow.Domain(field_shape, box=phiflow.box[0:1])
        self.step_count = step_count
        self.step_idx = 0
        self.ep_idx = 0
        self.dt = dt
        self.viscosity = viscosity
        self.physics = phiflow.Burgers(diffusion_substeps=diffusion_substeps)
        self.final_reward_factor = final_reward_factor
        self.reward_rms = RunningMeanStd()
        self.actions = None
        self.test_mode = False
        self.init_state = None
        self.goal_state = None
        self.cont_state = None
        self.pass_state = None
        self.gt_state = None
        self.gt_forces = None
        self.lviz = None
        self.fviz = None

    def reset(self) -> VecEnvObs:
        self.step_idx = 0

        self.gt_forces = phiflow.FieldEffect(GaussianForce(self.num_envs), ['velocity'])

        self.init_state = self._get_init_state()
        self.cont_state = self.init_state.copied_with()
        self.goal_state = self._get_goal_state(self.init_state, self.gt_forces)

        if self.test_mode:
            self._init_ref_states()

        return self._build_obs()

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions.reshape(self.cont_state.velocity.data.shape)

    def step_wait(self) -> VecEnvStepReturn:
        self.step_idx += 1

        forces = phiflow.FieldEffect(phiflow.CenteredGrid(self.actions, box=self.domain.box), ['velocity'])

        self.cont_state = self._step_sim(self.cont_state, (forces,))

        # Perform reference simulation only when evaluating results -> after render was called once
        if self.test_mode:
            self.pass_state = self._step_sim(self.pass_state, ())
            self.gt_state = self._step_sim(self.gt_state, (self.gt_forces,))

        obs = self._build_obs()

        rew = self._build_rew(forces)

        done = np.full((self.num_envs,), self.step_idx == self.step_count)

        info = [{} for _ in range(self.num_envs)]

        if self.step_idx == self.step_count:
            self.ep_idx += 1

            missing_forces_field = (self.goal_state.velocity.data - self.cont_state.velocity.data) / self.dt
            missing_forces = phiflow.FieldEffect(phiflow.CenteredGrid(missing_forces_field, box=self.domain.box), ['velocity'])
            
            self.cont_state = self._step_sim(self.cont_state, (missing_forces,)) 

            add_rew = self._build_rew(missing_forces) * self.final_reward_factor

            rew = add_rew
            #rew += add_rew


            obs = self.reset()
        else:
            rew = np.zeros(rew.shape)

        self.reward_rms.update(rew)

        #print('%f - %f' % (self.reward_rms.mean, np.sqrt(self.reward_rms.var)))

        norm_rew = rew / np.sqrt(self.reward_rms.var)

        #print(norm_rew)
        
        return obs, rew, done, info

    def close(self) -> None:
        pass

    def render(self, mode: str='l') -> None:
        if not self.test_mode:
            self.test_mode = True
            self._init_ref_states()
            self.lviz = LivePlotter()
            self.fviz = FilePlotter('StableBurger-%s' % self.exp_name)

        fields, labels = self._get_fields_and_labels()

        if mode == 'l':
            self.lviz.render(fields, labels, 2, True, 15)
        elif mode == 'f':
            self.fviz.render(fields, labels, 2, True, 'Velocity', self.ep_idx, self.step_idx, self.num_steps, True)
        else:
            raise NotImplementedError()

    def seed(self, seed: Optional[int]=None) -> List[Union[None, int]]:
        return [None for _ in range(self.num_envs)]

    def get_attr(self, attr_name: str, indices: VecEnvIndices=None):
        return [getattr(self, attr_name) for _ in self._vec_env_indices_to_list(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices=None, **method_kwargs) -> List[Any]:
        getattr(self, method_name)(*method_args, **method_kwargs)

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices=None) -> List[bool]:
        return [False for _ in self._vec_env_indices_to_list(indices)]

    def _step_sim(self, in_state: phiflow.BurgersVelocity, effects: Tuple[phiflow.FieldEffect, ...]) -> phiflow.BurgersVelocity:
        return self.physics.step(in_state, dt=self.dt, effects=effects)

    def _get_init_state(self) -> phiflow.BurgersVelocity:
        return phiflow.BurgersVelocity(domain=self.domain, velocity=GaussianClash(self.num_envs), viscosity=self.viscosity)

    def _get_goal_state(self, init_state: phiflow.BurgersVelocity, forces: phiflow.FieldEffect) -> phiflow.BurgersVelocity:
        state = init_state.copied_with()
        for _ in range(self.step_count):
            state = self._step_sim(state, (forces,))
        return state

    def _init_ref_states(self) -> None:
        self.pass_state = self.init_state.copied_with()
        self.gt_state = self.init_state.copied_with()

    def _build_obs(self) -> List[np.ndarray]:
        curr_data = self.cont_state.velocity.data
        goal_data = self.goal_state.velocity.data

        # Preserve the spacial dimensions, cut off batch dim and use only one channel
        time_shape = curr_data.shape[1:-1] + (1,)
        time_data = np.full(curr_data.shape[1:], self.step_idx / self.step_count)
        # Channels last
        return [np.concatenate(obs + (time_data,), axis=-1) for obs in zip(curr_data, goal_data)]

    def _build_rew(self, forces: phiflow.FieldEffect) -> np.ndarray:
        reduced_shape = (forces.field.data.shape[0], -1)
        reshaped_forces = forces.field.data.reshape(reduced_shape)
        return -np.sum(reshaped_forces ** 2, axis=-1)

    # The whole field with one parameter in each direction, flattened out
    def _get_act_shape(self, field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        act_dim = np.prod(field_shape) * len(field_shape)
        return (act_dim,)

    # Current and goal field with one parameter in each direction and one time channel
    def _get_obs_shape(self, field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return field_shape + (2 * len(field_shape) + 1,)

    def _vec_env_indices_to_list(self, raw_indices: VecEnvIndices) -> List[int]:
        if raw_indices is None:
            return []
        if isinstance(raw_indices, int):
            return [raw_indices]
        return list(raw_indices)
    
    def _get_fields_and_labels(self) -> Tuple[List[np.ndarray], List[str]]:
        # Take the simulation of the first env
        fields = [f.velocity.data[0].reshape(-1) for f in [
            self.init_state,
            self.goal_state,
            self.cont_state,
            self.pass_state,
            self.gt_state,
        ]]

        labels = [
            'Initial state',
            'Goal state',
            'Controlled simulation',
            'Uncontrolled simulation',
            'Ground truth simulation',
        ]

        return fields, labels