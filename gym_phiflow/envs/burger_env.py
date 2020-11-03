from gym_phiflow.envs import util, phiflow_util, visualization
import phi.tf.flow as phiflow
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
					np.real(self.prec_state.velocity.data).reshape(-1),
					np.real(self.pass_state.velocity.data).reshape(-1),
					np.real(self.init_state.velocity.data).reshape(-1),
					self.goal_obs.reshape(-1)]

			labels = ['Controlled Simulation',
					'Ground Truth Simulation',
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
		domain = phiflow.Domain(self.shape, box=phiflow.box[0:1])
		return phiflow.BurgersVelocity(domain=domain, velocity=phiflow_util.GaussianClash(1), viscosity=0.003)

	def step_sim(self, state, forces):
		controlled_state = state.copied_with(velocity=state.velocity.data + forces.reshape(state.velocity.data.shape) * self.delta_time)
		return self.physics.step(controlled_state, self.delta_time)

	def __init__(self, epis_len=32, dt=0.03, vel_scale=1.0,
			name='v0', act_type=util.ActionType.DISCRETE_2, loss_fn=util.l2_loss,
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
		self.physics = phiflow.Burgers()
		self.action_recorder = util.get_action_recorder(goal_type)
		self.action_space = util.get_action_space(act_type, act_dim)
		self.observation_space = util.get_observation_space(act_params.shape, goal_type, len(self.shape))
		self.vis_extractor = lambda s: np.squeeze(np.real(s.velocity.data), axis=0)
		self.force_gen = util.get_force_gen(act_type, act_params, self.get_random_state().velocity.data.shape, synchronized)
		self.goal_gen = util.get_goal_gen(self.force_gen, self.step_sim, 
			self.vis_extractor, self.get_random_state,
			act_type, goal_type, act_params.shape, act_dim, epis_len, self.action_recorder)
		self.obs_gen = util.get_obs_gen(goal_type, epis_len)
		self.rew_gen = util.get_rew_gen(rew_type, rew_force_factor, self.epis_len, loss_fn)
		self.cont_state = None
		self.pass_state = None
		self.init_state = None
		self.prec_state = None
		self.goal_obs = None
		self.lviz = None
		self.fviz = None
		self.force_collector = None
		self.test_mode = False

	def reset(self):
		if self.action_recorder is not None:
			self.action_recorder.reset()

		self.cont_state = self.get_random_state()
		self.goal_obs = self.goal_gen(self.cont_state.copied_with())
		self.step_idx = 0
		
		if self.test_mode:
			self.init_state = self.cont_state.copied_with()
			self.pass_state = self.cont_state.copied_with()
			self.prec_state = self.cont_state.copied_with()
		
			print('Average forces: %f' % self.force_collector.get_forces())

		return self.obs_gen(self.vis_extractor(self.cont_state), self.goal_obs, self.step_idx)

	def step(self, action):
		self.step_idx += 1
		v_old = self.vis_extractor(self.cont_state)

		forces = self.force_gen(action).copy()

		self.cont_state = self.step_sim(self.cont_state, forces)
		
		# Simulate the precomputed and uncontrolled states in test environments
		if self.test_mode:
			if self.action_recorder is not None:
				f_prec = self.force_gen(self.action_recorder.replay()).copy()
				self.prec_state = self.step_sim(self.prec_state, f_prec)
		
			self.pass_state = self.physics.step(self.pass_state, self.delta_time)
			
			self.force_collector.add_forces(forces)
		
		v_new = self.vis_extractor(self.cont_state)

		err_old = self.goal_obs - v_old
		err_new = self.goal_obs - v_new

		obs = self.obs_gen(self.vis_extractor(self.cont_state), self.goal_obs, self.step_idx)
		done = self.step_idx == self.epis_len
		reward = self.rew_gen(err_old, err_new, forces, done)

		if done:
			self.epis_idx += 1

		return obs, reward, done, {}

	def render(self, mode='l'):
		if not self.test_mode:
			self.test_mode = True
			self.force_collector = util.ForceCollector()
			self.init_state = self.cont_state.copied_with()
			self.pass_state = self.cont_state.copied_with()
			self.prec_state = self.cont_state.copied_with()

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
