import phi.tf.flow as phiflow
import gym
import numpy as np

import util
import phiflow_util

class BurgersEnv(gym.Env):
    metadata = {'render.modes': ['l', 'f']}

    def __init__(self, step_count=32, dt=0.03, viscosity=0.003, field_shape=(32,)):
        act_shape = self.get_act_shape(field_shape)
        obs_shape = self.get_obs_shape(field_shape)
        self.action_space = gym.spaces.box(-np.inf, np.inf, shape=act_shape, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)



        self.domain = phiflow.Domain([128], box=phiflow.box[0:1])
        self.viscosity = viscosity
        self.step_count = 32
        self.dt = dt

        self.step_idx = 0

        
        self.cont_world = phiflow.World()
        self.prec_world = phiflow.World()
        self.pass_world = phiflow.World()

        self.physics = phiflow.Burgers(diffusion_substeps=4)

        self.cont_state = None
        self.prec_state = None
        self.pass_world = None

    def reset(self):
        self.step_idx = 0

        self.cont_world.reset()
        self.prec_world.reset()
        self.pass_world.reset()

        self.cont_state = create_state(self.cont_world)
        self.prec_state = create_state(self.prec_world)
        self.pass_state = create_state(self.pass_world)

        self.force = create_forces(self.prec_world)

        for _ in range(self.step_count):
            self.prec_world.step(dt=self.dt)

        return build_obs()

    def step(self, action):
        

    def create_state(self, world):
        state = phiflow.BurgersVelocity(domain=self.domain, velocity=phiflow_util.GaussianClash(1), viscosity=self.viscosity)
        return world.add(state, self.physics)

    def create_forces(self, world):
        forces = phiflow.FieldEffect(phiflow_util.GaussianForce(1), ['velocity'])
        return world.add(forces)

    def build_observation(self):
        return np.concatenate()


    # The whole field with one parameter in each direction, flattened out
    def get_act_shape(self, field_shape):
        act_dim = np.prod(field_shape) * len(field_shape)
        return (act_dim,)

    # Current and goal field with one parameter in each direction and one time channel
    def get_obs_shape(self, field_shape):
        return tuple(list(field_shape).append(2 * len(field_shape) + 1)
