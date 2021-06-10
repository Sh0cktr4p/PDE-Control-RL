from typing import Tuple
import numpy as np

from phi.tf.flow import box, Domain, FieldEffect, Fluid, IncompressibleFlow
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
    
    def _step_sim(self, in_state: Fluid, effects: Tuple[FieldEffect]) -> Fluid:
        return self.physics.step(in_state, dt=self.dt, velocity_effects=effects)

    def _get_init_state(self) -> Fluid:
        return Fluid(self.domain, density=self._generate_density_field())

    def _generate_density_field(self) -> 

    def _get_vel_shape(self, domain: Domain) -> Tuple[int]:
        den_shape = domain.resolution
        vel_shape = (den_dim + 1 for den_dim in den_shape) + (len(den_shape),)
        return vel_shape

    def _get_act_shape(self, domain: Domain)-> Tuple[int]:
        return np.prod(self._get_vel_shape(domain)),

    def _get_obs_shape(self, domain: Domain)-> Tuple[int]:
        vel_shape = self._get_vel_shape(domain)
        # Add channels for current density, goal density, time
        return vel_shape[:-1] + (vel_shape[-1] + 3)