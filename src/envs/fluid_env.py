from typing import Tuple
import numpy as np
import gym

from phi.tf.flow import box, Domain, FieldEffect, Fluid, IncompressibleFlow, StaggeredGrid
from phi.tf.tf_cuda_pressuresolver import CUDASolver

from src.envs.pde_env import PDEEnv


class FluidEnv(PDEEnv):
    metadata = {'render.modes': ['live', 'gif']}

    def __init__(
        self,
        num_envs: int,
        step_cnt: int=32,
        domain: Domain=Domain((32, 32), box=box[0:1]),
        dt: float=0.03,
        final_reward_factor: float=32,
        exp_name: str='v0',
    ):
        act_shape = self._get_act_shape(domain.resolution)
        obs_shape = self._get_obs_shape(domain.resolution)

        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        action_space = gym.spaces.Box(-np.inf, np.inf, shape=act_shape, dtype=np.float32)

        super().init(
            physics_type=IncompressibleFlow,
            physics_kwargs=dict(
                pressure_solver=CUDASolver,
            ),
            observation_space=observation_space,
            action_space=action_space,
            num_envs=num_envs,
            domain=domain,
            step_cnt=step_cnt,
            dt=dt,
            final_reward_factor=final_reward_factor,
            exp_name=exp_name,
        )
    
    def _get_init_state(self) -> Fluid:
        return Fluid(self.domain, density=self._generate_density_field())

    def _get_goal_state(self) -> Fluid:
        pass

    def _step_sim(self, in_state: Fluid, effects: Tuple[FieldEffect]) -> Fluid:
        return self.physics.step(in_state, dt=self.dt, velocity_effects=effects)

    def _build_obs(self):
        pass

    def _reshape_actions_to_forces(self) -> np.ndarray:
        return self.actions.reshape(self.cont_state.velocity.data.shape)

    def _forces_to_field_effect(self, forces: np.ndarray) -> FieldEffect:
        return FieldEffect(StaggeredGrid(forces, box=self.domain.box), ['velocity'])

    def _get_final_reward(self):
        field_difference = np.sum((self.goal_state.density.data - self.cont_state.density.data) ** 2)
        return -field_difference * self.final_reward_factor

    def _get_live_viz(self):
        pass

    def _get_gif_viz(self):
        pass

    def _get_png_viz(self):
        pass

    def _get_fields_labels_colors(self):
        pass

    def _generate_density_field(self):
        pass

    def _get_act_shape(self, domain: Domain)-> Tuple[int]:
        return np.prod(self._get_vel_shape(domain)),

    def _get_obs_shape(self, domain: Domain)-> Tuple[int]:
        vel_shape = self._get_vel_shape(domain)
        # Add channels for current density, goal density, time
        return vel_shape[:-1] + (vel_shape[-1] + 3)

    def _get_vel_shape(self, domain: Domain) -> Tuple[int]:
        den_shape = domain.resolution
        vel_shape = (den_dim + 1 for den_dim in den_shape) + (len(den_shape),)
        return vel_shape