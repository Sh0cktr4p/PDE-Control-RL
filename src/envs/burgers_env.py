import numpy as np

import gym
from phi.tf.flow import box, Burgers, BurgersVelocity, CenteredGrid, Domain, FieldEffect

from .pde_env import PDEEnv
from .burgers_util import GaussianClash, GaussianForce

class BurgersEnv(PDEEnv):
    metadata = {'render.modes': ['live', 'gif', 'png']}

    def __init__(
        self, 
        num_envs: int, 
        step_cnt: int=32, 
        domain: Domain=Domain((32,), box=box[0:1]), 
        dt: float=0.03, 
        viscosity: float=0.003, 
        diffusion_substeps: int=1,
        final_rew_factor: float=32, 
        exp_name: str='v0',
    ):
        act_shape = self._get_act_shape(domain.resolution)
        obs_shape = self._get_obs_shape(domain.resolution)
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        action_space = gym.spaces.Box(-np.inf, np.inf, shape=act_shape, dtype=np.float32)

        super().__init__(
            Burgers, 
            dict(diffusion_substeps=diffusion_substeps),
            observation_space,
            action_space,
            num_envs,
            domain,
            step_cnt,
            dt,
            final_rew_factor,
            exp_name)

        self.viscosity = viscosity

    def _step_sim(self, in_state: DomainState, effect: FieldEffect) -> DomainState:
        return self.physics.step(in_state, dt=self.dt, effects=effects)

    def _get_gt_forces(self) -> FieldEffect:
        return FieldEffect(GaussianForce(self.num_envs), ['velocity'])

    def _get_init_state(self) -> DomainState:
        return BurgersVelocity(domain=self.domain, velocity=GaussianClash(self.num_envs), ['velocity'])

    def _build_obs(self):
        curr_data = self.cont_state.velocity.data
        goal_data = self.goal_state.velocity.data

        # Preserve the spacial dimensions, cut off batch dim and use only one channel
        time_shape = curr_data.shape[1:-1] + (1,)
        time_data = np.full(curr_data.shape[1:], self.step_idx / self.step_count)
        # Channels last
        return [np.concatenate(obs + (time_data,), axis=-1) for obs in zip(curr_data, goal_data)]

    def _reshape_actions_to_forces(self) -> np.ndarray:
        return self.actions.reshape(self.cont_state.velocity.data.shape)

    def _forces_to_field_effect(self, forces: np.ndarray) -> FieldEffect:
        return FieldEffect(CenteredGrid(self.actions, box=self.domain.box), ['velocity'])

    def _get_final_reward(self):
        missing_forces = (self.goal_state.velocity.data - self.cont_state.velocity.data) / self.dt
        return self._build_rew(missing_forces) * self.final_rew_factor

    def _get_live_viz(self):
        raise NotImplementedError()

    def _get_gif_viz(self):
        raise NotImplementedError()

    def _get_png_viz(self):
        raise NotImplementedError()

    def _get_fields_and_labels(self):
        # Take the simulation of the first env
        fields = [f.velocity.data[0].reshape(-1) for f in [
            self.init_state,
            self.goal_state,
            self.pass_state,
            self.gt_state,
            self.cont_state,
        ]]

        labels = [
            'Initial state',
            'Goal state',
            'Uncontrolled simulation',
            'Ground truth simulation',
            'Controlled simulation',
        ]

        return fields, labels

    # The whole field with one parameter in each direction, flattened out
    def _get_act_shape(self, field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        act_dim = np.prod(field_shape) * len(field_shape)
        return (act_dim,)

    # Current and goal field with one parameter in each direction and one time channel
    def _get_obs_shape(self, field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(field_shape) + (2 * len(field_shape) + 1,)