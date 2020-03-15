from gym_phiflow.envs import util, visualization
import phi.flow
import gym
import numpy as np

default_act_points = util.act_points((16,), 0)

class NavierEnv(gym.Env):
	# Visualization, Plot, File
	metadata = {'render.modes': ['v', 'p', 'f']}

	def get_state_with(self, value):
		return phi.flow.Smoke(phi.flow.Domain(self.den_shape), density=value)

	def get_random_state(self):
		return phi.flow.Smoke(phi.flow.Domain(self.den_shape), density=phi.flow.math.randn(levels=[self.den_scale]))

	def step_sim(self, state, forces):
		staggered_forces = phi.flow.math.StaggeredGrid(forces.reshape(state.velocity.staggered.shape))
		controlled_state = state.copied_with(velocity=state.velocity + staggered_forces * self.delta_time)
		return self.physics.step(controlled_state, self.delta_time)

	def __init__(self, epis_len=32, dt=0.5, den_scale=1.0, use_time=False, 
			name='v0', act_type=util.ActionType.DISCRETE_2, act_points=default_act_points, 
			goal_type=util.GoalType.ZERO, rew_type=util.RewardType.ABSOLUTE, rew_force_factor=1, 
			init_field_gen=None, goal_field_gen=None):
		act_params = util.get_all_act_params(act_points)	# Multi-dimensional support
		self.step_idx = 0
		self.epis_idx = 0
		self.epis_len = epis_len
		self.delta_time = dt
		self.den_scale = den_scale
		self.exp_name = name
		self.den_shape = tuple(d-1 for d in act_points.shape)	# Act points refers to staggered velocity grid
		self.physics = phi.flow.SmokePhysics()
		self.action_space = util.get_action_space(act_type, np.sum(act_params))
		self.observation_space = util.get_observation_space(self.den_shape, goal_type, use_time)
		self.force_gen = util.get_force_gen(act_type, act_params, self.get_random_state().velocity.staggered.shape)
		self.init_gen = (lambda: self.get_state_with(init_field_gen())) if init_field_gen else self.get_random_state
		self.goal_gen = util.get_goal_gen(self.force_gen, self.step_sim,
			lambda s: np.squeeze(s.density), self.get_random_state, act_type, goal_type, 
			self.den_shape, np.sum(act_params), epis_len, goal_field_gen)
		self.obs_gen = util.get_obs_gen(goal_type, use_time, epis_len)
		self.rew_gen = util.get_rew_gen(rew_type, rew_force_factor)
		self.cont_state = None
		self.pass_state = None
		self.init_state = None
		self.goal_obs = None
		self.renderer = None
		self.live_plotter = None
		self.file_plotter = None

	def reset(self):
		self.init_state = self.init_gen()
		self.cont_state = self.init_state.copied_with()
		self.pass_state = self.init_state.copied_with()
		self.goal_obs = self.goal_gen(self.init_state.copied_with())
		self.step_idx = 0
		return self.obs_gen(np.squeeze(self.cont_state.density), self.goal_obs, self.step_idx)

	def step(self, action):
		self.step_idx += 1

		old_obs = np.squeeze(self.cont_state.density)

		forces = self.force_gen(action)

		self.cont_state = self.step_sim(self.cont_state, forces)
		self.pass_state = self.physics.step(self.pass_state, self.delta_time)

		new_obs = np.squeeze(self.cont_state.density)

		mse_old = np.sum((self.goal_obs - old_obs) ** 2)
		mse_new = np.sum((self.goal_obs - new_obs) ** 2)

		obs = self.obs_gen(np.squeeze(self.cont_state.density), self.goal_obs, self.step_idx)
		reward = self.rew_gen(mse_old, mse_new, forces)
		done = self.step_idx == self.epis_len

		if done:
			print(self.epis_idx)
			self.epis_idx += 1

		return obs, reward, done, {}

	def render(self, mode='v'):
		fields = [self.cont_state.density.reshape(-1),
					self.pass_state.density.reshape(-1),
					self.init_state.density.reshape(-1),
					self.goal_obs]

		labels = ['Controlled Simulation',
					'Uncontrolled Simulation',
					'Initial Density Field',
					'Goal Density Field']

		if mode == 'v':
			if self.renderer is None:
				self.renderer = visualization.Renderer()
			self.renderer.render(self.cont_state.density, 15, 0.1, 500, 500)
		elif mode == 'p':
			if self.live_plotter is None:
				self.live_plotter = visualization.LivePlotter()
			self.live_plotter.render(fields, labels)
		elif mode == 'f':
			if self.file_plotter is None:
				self.file_plotter = visualization.FilePlotter('SpinningNavier-%s' % self.exp_name)
			self.file_plotter.render(fields, labels, 'Velocity', self.epis_idx, self.step_idx, self.epis_len)
		else:
			raise NotImplementedError()

	def close(self):
		pass

	def seed(self):
		pass
