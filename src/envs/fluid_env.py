from src.visualization import LiveViz
from typing import Tuple
import numpy as np
import gym

from phi.tf.flow import box, Domain, FieldEffect, Fluid, IncompressibleFlow, StaggeredGrid
#from phi.tf.tf_cuda_pressuresolver import CUDASolver

from src.envs.pde_reconstruct_env import PDEReconstructEnv
from src.util.burgers_util import GaussianForce
from src.util.shape_util import ShapeField

class FluidEnv(PDEReconstructEnv):
    metadata = {'render.modes': ['live', 'gif']}

    def __init__(
        self,
        num_envs: int,
        step_cnt: int=32,
        domain: Domain=Domain((32, 32), box=box[0:1, 0:1]),
        dt: float=0.03,
        final_reward_factor: float=32,
        exp_name: str='v0',
        total_density: float=10,
    ):
        act_shape = self._get_act_shape(domain)
        obs_shape = self._get_obs_shape(domain)

        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        action_space = gym.spaces.Box(-np.inf, np.inf, shape=act_shape, dtype=np.float32)

        super().__init__(
            physics_type=IncompressibleFlow,
            physics_kwargs=dict(
                pressure_solver=None,
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

        self.total_density = total_density
    
    def _get_init_state(self) -> Fluid:
        return Fluid(self.domain, density=ShapeField(self.num_envs, self.domain.rank))

    def _get_gt_forces(self) -> FieldEffect:
        return FieldEffect(GaussianForce(self.num_envs, self.domain.rank) * 8, ['velocity'])

    def _step_sim(self, in_state: Fluid, effects: Tuple[FieldEffect]) -> Fluid:
        return self.physics.step(in_state, dt=self.dt, velocity_effects=effects)

    def _build_obs(self):
        curr_vel = self.cont_state.velocity.staggered_tensor()
        curr_den = self._pad_den_to_vel(self.cont_state.density.data)
        goal_den = self._pad_den_to_vel(self.goal_state.density.data)
        time = np.full(curr_den.shape[1:], self.step_idx / self.step_cnt)
        return [np.concatenate(obs + (time,), axis=-1) for obs in zip(curr_vel, curr_den, goal_den)]

    def _reshape_actions_to_forces(self) -> np.ndarray:
        return self.actions.reshape(self.cont_state.velocity.staggered_tensor().shape)

    def _forces_to_field_effect(self, forces: np.ndarray) -> FieldEffect:
        return FieldEffect(StaggeredGrid(forces, box=self.domain.box), ['velocity'])

    def _get_final_reward(self):
        field_difference = np.sum((self.goal_state.density.data - self.cont_state.density.data) ** 2)
        return -field_difference * self.final_reward_factor

    def _get_live_viz(self):
        return LiveViz(self.domain.rank, 1, False)

    def _get_gif_viz(self):
        raise NotImplementedError()

    def _get_png_viz(self):
        raise NotImplementedError()

    def _get_fields_labels_colors(self):
        fields = [f.velocity.staggered_tensor()[0] for f in [
            #self.cont_state,
            #self.pass_state,
            self.gt_state,
            #self.init_state,
            #self.goal_state,
        ]]

        #print(fields[0].shape)

        labels = [
            #'Controlled simulation',
            #'Uncontrolled simulation',
            'Ground truth simulation'
            #'Initial state',
            #'Goal state',
        ]

        colors = [
            'orange',
            #'blue',
            #'green',
            #'magenta',
            #'red',
        ]

        return fields, labels, colors

    def _get_act_shape(self, domain: Domain)-> Tuple[int]:
        return np.prod(self._get_vel_shape(domain)),

    def _get_obs_shape(self, domain: Domain)-> Tuple[int]:
        vel_shape = self._get_vel_shape(domain)
        # Add channels for current density, goal density, time
        return vel_shape[:-1] + (vel_shape[-1] + 3,)

    def _get_vel_shape(self, domain: Domain) -> Tuple[int]:
        den_shape = domain.resolution
        vel_shape = tuple(den_dim + 1 for den_dim in den_shape) + (len(den_shape),)
        return vel_shape

    def _pad_den_to_vel(self, density_tensor: np.ndarray) -> np.ndarray:
        # Add padding to all field dimensions to match velocity field
        # Ignore first (batch) and last (field value dimensionality) dimensions
        return np.pad(density_tensor, [(0,0)] + [(0,1) for _ in density_tensor.shape[1:-1]] + [(0,0)])