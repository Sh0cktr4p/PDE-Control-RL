import numpy as np
from typing import Tuple
import gym
from phi.tf.flow import box, Burgers, BurgersVelocity, CenteredGrid, Domain, DomainState, FieldEffect

from src.envs.pde_env import PDEEnv
from src.envs.burgers_util import GaussianClash, GaussianForce
from src.visualization import LiveViz, GifViz, PngViz

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
        final_reward_factor: float=32, 
        exp_name: str='v0',
    ):
        act_shape = self._get_act_shape(domain.resolution)
        obs_shape = self._get_obs_shape(domain.resolution)
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        action_space = gym.spaces.Box(-np.inf, np.inf, shape=act_shape, dtype=np.float32)

        super().__init__(
            physics_type=Burgers, 
            physics_kwargs=dict(
                diffusion_substeps=diffusion_substeps,
            ),
            observation_space=observation_space,
            action_space=action_space,
            num_envs=num_envs,
            domain=domain,
            step_cnt=step_cnt,
            dt=dt,
            final_reward_factor=final_reward_factor,
            exp_name=exp_name
        )

        self.viscosity = viscosity

    def _step_sim(self, in_state: DomainState, effects: Tuple[FieldEffect]) -> BurgersVelocity:
        return self.physics.step(in_state, dt=self.dt, effects=effects)

    def _get_gt_forces(self) -> FieldEffect:
        return FieldEffect(GaussianForce(self.num_envs, self.domain.rank), ['velocity'])

    def _get_init_state(self) -> BurgersVelocity:
        return BurgersVelocity(domain=self.domain, velocity=GaussianClash(self.num_envs, self.domain.rank), viscosity=self.viscosity)

    def _build_obs(self):
        curr_data = self.cont_state.velocity.data
        goal_data = self.goal_state.velocity.data

        # Preserve the spacial dimensions, cut off batch dim and use only one channel
        time_shape = curr_data.shape[1:-1] + (1,)
        time_data = np.full(time_shape, self.step_idx / self.step_cnt)
        # Channels last
        return [np.concatenate(obs + (time_data,), axis=-1) for obs in zip(curr_data, goal_data)]

    def _reshape_actions_to_forces(self) -> np.ndarray:
        return self.actions.reshape(self.cont_state.velocity.data.shape)

    def _forces_to_field_effect(self, forces: np.ndarray) -> FieldEffect:
        return FieldEffect(CenteredGrid(forces, box=self.domain.box), ['velocity'])

    def _get_final_reward(self):
        missing_forces = (self.goal_state.velocity.data - self.cont_state.velocity.data) / self.dt
        return self._build_rew(missing_forces) * self.final_reward_factor

    def _get_live_viz(self):
        return LiveViz(self.domain.rank, 2, True)

    def _get_gif_viz(self):
        return GifViz("StableBurger-%s" % self.exp_name, "Velocity", self.domain.rank, self.step_cnt, 2, True, True)

    def _get_png_viz(self):
        return PngViz("StableBurger-%s" % self.exp_name, "Velocity", self.domain.rank, self.step_cnt, 2, True)

    def _get_fields_labels_colors(self):
        # Take the simulation of the first env
        fields = [f.velocity.data[0] for f in [
            self.cont_state,
            self.gt_state,
            self.pass_state,
            self.init_state,
            self.goal_state,
        ]]

        labels = [
            'Controlled simulation',
            'Ground truth simulation',
            'Uncontrolled simulation',
            'Initial state',
            'Goal state',
        ]

        colors = [
            'orange',
            'blue',
            'green',
            'magenta',
            'red',
        ]

        return fields, labels, colors

    # The whole field with one parameter in each direction, flattened out
    def _get_act_shape(self, field_shape: Tuple[int]) -> Tuple[int]:
        act_dim = np.prod(field_shape) * len(field_shape)
        return (act_dim,)

    # Current and goal field with one parameter in each direction and one time channel
    def _get_obs_shape(self, field_shape: Tuple[int]) -> Tuple[int]:
        return tuple(field_shape) + (2 * len(field_shape) + 1,)
