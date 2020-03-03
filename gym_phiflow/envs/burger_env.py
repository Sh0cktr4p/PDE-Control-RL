from gym_phiflow.envs import util, visualization
import phi.flow
import numpy as np
import gym
import time


default_act_points = util.act_points((16,), 0)

def get_all_act_params(points):
	points = np.squeeze(points)
	return np.squeeze(np.stack([points for _ in range(points.ndim)], points.ndim))

class BurgerEnv(gym.Env):
	# Visualization, Plot, File
	metadata = {'render.modes': ['v', 'p', 'f']}

	def get_random_state(self):
		return phi.flow.Burger(phi.flow.Domain(self.shape), phi.flow.math.randn(levels=[0, 0, self.vel_scale]), viscosity=0.2)

	def step_sim(self, state, forces):
		controlled_state = state.copied_with(velocity=state.velocity + forces.reshape(state.velocity.shape) * self.delta_time)
		return self.physics.step(controlled_state, self.delta_time)

	def __init__(self, epis_len=32, dt=0.5, vel_scale=1.0, use_time=False,
			name='v0', act_type=util.ActionType.DISCRETE_2, 
			act_points=default_act_points, goal_type=util.GoalType.ZERO, 
			rew_type=util.RewardType.ABSOLUTE, rew_force_factor=1):
		act_params = get_all_act_params(act_points)	# Important for multi-dimensional cases
		self.step_idx = 0
		self.epis_idx = 0
		self.epis_len = epis_len
		self.shape = act_points.shape
		self.delta_time = dt
		self.vel_scale = vel_scale
		self.exp_name = name
		self.physics = phi.flow.BurgerPhysics()
		self.action_space = util.get_action_space(act_type, np.sum(act_params))
		self.observation_space = util.get_observation_space(act_params.size, goal_type, use_time)
		self.force_gen = util.get_force_gen(act_type, act_params, self.get_random_state().velocity.shape)
		self.goal_gen = util.get_goal_gen(self.force_gen, self.step_sim, 
			lambda s: np.real(s.velocity).reshape(-1), self.get_random_state,
			act_type, goal_type, act_params.size, np.sum(act_params), epis_len)
		self.obs_gen = util.get_obs_gen(goal_type, use_time, epis_len)
		self.rew_gen = util.get_rew_gen(rew_type, rew_force_factor)
		self.cont_state = None
		self.pass_state = None
		self.init_obs = None
		self.goal_obs = None
		self.renderer = None
		self.live_plotter = None
		self.file_plotter = None

	def reset(self):
		self.cont_state = self.get_random_state()
		self.pass_state = self.cont_state.copied_with()
		self.init_state = self.cont_state.copied_with()
		self.goal_obs = self.goal_gen(self.init_state.copied_with())
		self.step_idx = 0

		return self.obs_gen(np.real(self.cont_state.velocity).reshape(-1), self.goal_obs, self.step_idx)

	def step(self, action):
		self.step_idx += 1

		v_old = np.real(self.cont_state.velocity).reshape(-1)

		forces = self.force_gen(action)

		self.cont_state = self.step_sim(self.cont_state, forces)
		self.pass_state = self.physics.step(self.pass_state, self.delta_time)

		v_new = np.real(self.cont_state.velocity).reshape(-1)

		mse_old = np.sum((self.goal_obs - v_old) ** 2)
		mse_new = np.sum((self.goal_obs - v_new) ** 2)

		obs = self.obs_gen(np.real(self.cont_state.velocity).reshape(-1), self.goal_obs, self.step_idx)
		reward = self.rew_gen(mse_old, mse_new, forces)
		done = self.step_idx == self.epis_len

		if done:
			self.epis_idx += 1

		return obs, reward, done, {}

	def render(self, mode='p'):
		fields = [np.real(self.cont_state.velocity).reshape(-1), 
					np.real(self.pass_state.velocity).reshape(-1),
					np.real(self.init_state.velocity).reshape(-1),
					self.goal_obs]
		
		labels = ['Controlled Simulation',
					'Uncontrolled Simulation',
					'Initial Field',
					'Goal Field']
		
		if mode == 'v':
			if self.renderer is None:
				self.renderer = visualization.Renderer()
			self.renderer.render(self.cont_state.velocity, 15, 1, 500, 500)
		elif mode == 'p':
			if self.live_plotter is None:
				self.live_plotter = visualization.LivePlotter()
			self.live_plotter.render(fields, labels)
		elif mode == 'f':
			if self.file_plotter is None:
				self.file_plotter = visualization.FilePlotter('SpinningBurger%s' % self.exp_name)
			self.file_plotter.render(fields, labels, 'Velocity', self.epis_idx, self.step_idx, self.epis_len)
		else:
			raise NotImplementedError()

	def close(self):
		pass

	def seed(self):
		pass
