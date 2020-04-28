from gym_phiflow.envs import util, visualization
import phi.flow
import numpy as np
import gym
import time


default_act_points = util.act_points((16,), 0)


class BurgerEnv(gym.Env):
	# Live, File
	metadata = {'render.modes': ['l', 'f']}

	def get_fields_and_labels(self):
		ndim = len(self.shape)

		if ndim == 1:
			fields = [np.real(self.cont_state.velocity.data).reshape(-1),
					np.real(self.pass_state.velocity.data).reshape(-1),
					np.real(self.init_state.velocity.data).reshape(-1),
					self.goal_obs.reshape(-1)]

			labels = ['Controlled Simulation',
					'Uncontrolled Simulation',
					'Initial Density Field',
					'Goal Density Field']
		elif ndim == 2:
			fields = [self.cont_state.velocity.data,
					self.init_state.velocity.data,
					self.goal_obs.reshape([1] + list(self.goal_obs.shape))]
			labels = ['Controlled Simulation',
					'Initial Density Field',
					'Goal Density Field']
		else:
			raise NotImplementedError()
		
		return fields, labels

	def get_random_state(self):
		domain = phi.flow.Domain(self.shape)
		return phi.flow.BurgersVelocity(domain=domain, velocity=phi.flow.Noise(channels=domain.rank) * 2, viscosity=0.2)

	def step_sim(self, state, forces):
		controlled_state = state.copied_with(velocity=state.velocity.data + forces.reshape(state.velocity.data.shape) * self.delta_time)
		return self.physics.step(controlled_state, self.delta_time)

	def __init__(self, epis_len=32, dt=0.5, vel_scale=1.0, use_time=False,
			name='v0', act_type=util.ActionType.DISCRETE_2, 
			act_points=default_act_points, goal_type=util.GoalType.ZERO, 
			rew_type=util.RewardType.ABSOLUTE, rew_force_factor=1, synchronized=False):
		act_params = util.get_all_act_params(act_points)	# Important for multi-dimensional cases
		act_dim = 1 if synchronized else np.sum(act_params)
		
		self.step_idx = 0
		self.epis_idx = 0
		self.epis_len = epis_len
		self.delta_time = dt
		self.vel_scale = vel_scale
		self.exp_name = name
		self.shape = act_points.shape
		self.physics = phi.flow.Burgers()
		self.action_space = util.get_action_space(act_type, act_dim)
		self.observation_space = util.get_observation_space(act_params.shape, goal_type, len(self.shape), use_time)
		self.vis_extractor = lambda s: np.squeeze(np.real(s.velocity.data), axis=0)
		self.force_gen = util.get_force_gen(act_type, act_params, self.get_random_state().velocity.data.shape, synchronized)
		self.goal_gen = util.get_goal_gen(self.force_gen, self.step_sim, 
			self.vis_extractor, self.get_random_state,
			act_type, goal_type, act_params.shape, act_dim, epis_len)
		self.obs_gen = util.get_obs_gen(goal_type, use_time, epis_len)
		self.rew_gen = util.get_rew_gen(rew_type, rew_force_factor, self.epis_len)
		self.cont_state = None
		self.pass_state = None
		self.init_state = None
		self.goal_obs = None
		self.lviz = None
		self.fviz = None

	def reset(self):
		self.cont_state = self.get_random_state()
		self.pass_state = self.cont_state.copied_with()
		self.init_state = self.cont_state.copied_with()
		self.goal_obs = self.goal_gen(self.init_state.copied_with())
		self.step_idx = 0
		
		return self.obs_gen(self.vis_extractor(self.cont_state), self.goal_obs, self.step_idx)

	def step(self, action):
		tac = time.time()
		self.step_idx += 1
		v_old = self.vis_extractor(self.cont_state)

		forces = self.force_gen(action)

		tic = time.time()
		self.cont_state = self.step_sim(self.cont_state, forces)
		self.pass_state = self.physics.step(self.pass_state, self.delta_time)
		toc = time.time()

		v_new = self.vis_extractor(self.cont_state)

		mse_old = np.sum((self.goal_obs - v_old) ** 2)
		mse_new = np.sum((self.goal_obs - v_new) ** 2)

		obs = self.obs_gen(self.vis_extractor(self.cont_state), self.goal_obs, self.step_idx)
		done = self.step_idx == self.epis_len
		reward = self.rew_gen(mse_old, mse_new, forces, done)

		if done:
			self.epis_idx += 1

		tec = time.time()
		#print(tec - tac - (toc - tic))

		return obs, reward, done, {}

	def render(self, mode='l'):
		fields, labels = self.get_fields_and_labels()

		ndim = len(self.shape)
		max_value = 2
		signed = True

		if mode == 'l':
			frame_rate = 15
			if self.lviz is None:
				if ndim == 1:
					self.lviz = visualization.LivePlotter()
				elif ndim == 2:
					self.lviz = visualization.LiveRenderer()
			self.lviz.render(fields, labels, max_value, signed, frame_rate)
		elif mode == 'f':
			remove_frames = True
			category_name = 'SpinningBurger-%s' % self.exp_name
			if self.fviz is None:
				if ndim == 1:
					self.fviz = visualization.FilePlotter(category_name)
				elif ndim == 2:
					self.fviz = visualization.FileRenderer(category_name)
			self.fviz.render(fields, labels, max_value, signed, 'Velocity', 
				self.epis_idx, self.step_idx, self.epis_len, remove_frames)
		else:
			raise NotImplementedError()

	def close(self):
		pass

	def seed(self):
		pass
