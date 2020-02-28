import gym
import time
import phi.flow

class BurgerAdvectEnv(gym.Env):
	metadata = {'render.modes': ['human', 'file', 'live_plot']}

	@property
	def size(self):
		return (16,)

	def __init__(self):
		self.step_index = 0
		self.episode_index = 0
		self.physics = phi.flow.BurgerPhysics()
		self.burger = phi.flow.Burger(Domain(self.size), math.randn(levels=[0, 0, self.value_velocity_scale]), viscosity=0.2)
		self.adv_field = 


	def step(self, action):
		pass

	def reset(self):
		pass

	def render(self, mode='human'):
		pass

	def close(self):
		pass

	def seed(self):
		pass