from gym_phiflow.envs import util, visualization
import phi.flow
import numpy as np
import gym
import time


default_act_points = util.act_points((16,), 0)

	#[-1.5153133  ,-1.1824083 , -0.51659834,  0.4821166 ,  1.8137366 ,  2.4914613, 2.5152907  , 1.8852252 ,  0.60126436, -0.2511903 , -0.67213887, -0.66158134,-0.21951762 , 0.10902914,  0.32405895,  0.4255718 ,  0.41356775,  0.5633231, 0.8748379  , 1.3481121 ,  1.9831457 ,  2.2128086 ,  2.037101  ,  1.4560227, 0.46957383 ,-0.1301138 , -0.3430401 , -0.16920513,  0.39139116,  0.8118383, 1.0921365  , 1.2322856 ]
	#[ 1.216977  ,  0.9180328 ,  0.32014453, -0.5766879 , -1.7724645 ,   -2.5500746 , -2.909518  , -2.8507948 , -2.3739052 , -1.8905934 ,   -1.4008597 , -0.904704  , -0.40212622,  0.01870531,  0.35779056,    0.61512953,  0.7907223 ,  0.79981065,  0.64239466,  0.31847432,   -0.1719504 , -0.59456354, -0.94936514, -1.2363552 , -1.4555337 ,   -1.3725449 , -0.98738897, -0.3000657 ,  0.6894248 ,  1.4315426 ,    1.9262879 ,  2.1736605 ]
	#[ 0.28132215,  0.01652982, -0.51305485, -1.3074318 , -2.3666012 ,-2.8866472 , -2.86757   , -2.3093696 , -1.2120457 , -0.3898618 , 0.15718208,  0.429086  ,  0.42584994,  0.3588556 ,  0.22810292, 0.03359195, -0.22467732, -0.30138472, -0.1965302 ,  0.08988624, 0.55786455,  0.83477604,  0.9206207 ,  0.81539845,  0.51910937, 0.40304565,  0.46720728,  0.7115943 ,  1.1362067 ,  1.454666  , 1.6669722 ,  1.7731253 ]
	#[ 0.23698127, 0.20962676, 0.15491773, 0.07285421, -0.03656384, -0.17333639,-0.33746344, -0.528945,   -0.7477811,  -0.90284604, -0.99413985, -1.0216625,-0.9854141,  -0.88539445, -0.72160375, -0.49404186, -0.20270886,  0.03720472,0.22569886,  0.36277357,  0.44842884,  0.48266467,  0.46548107,  0.39687803, 0.2768556,   0.17183591,  0.08181907,  0.00680503, -0.0532062,  -0.09821463,-0.12822025, -0.14322305]


	#[ 	1.8177022  , 1.8131468   ,1.804036    ,1.7903697  , 1.7721481  , 1.7493712,1.7220387  , 1.690151    ,1.6537077   ,1.589743   , 1.4982569  , 1.3792493,1.2327203  , 1.0586697   ,0.85709757  ,0.6280041  , 0.3713891  , 0.15878117,-0.00981972, -0.13441356 ,-0.21500033 ,-0.25158006, -0.24415275, -0.19271839,-0.09727697, -0.01376573 , 0.05781533 , 0.11746622,  0.16518693,  0.20097746,0.22483781 , 0.23676799]

prebaked_velocity = np.array(
	[	-0.26236698, -0.1487501 ,  0.07848366,  0.4193343 ,  0.87380177,  1.1327413,1.1961529  , 1.0640365  , 0.7363921  , 0.413065   , 0.09405523 ,-0.2206372,-0.5310123 , -0.7084713 , -0.7530141 , -0.6646408 , -0.44335133, -0.24482939,-0.06907493,  0.08391204,  0.2141315 ,  0.22179411,  0.10689985, -0.13055128,-0.49055928, -0.59194547, -0.43470988, -0.01885249,  0.6556267 ,  1.161486,1.4987257  , 1.6673455 ]
	).reshape(1, 32, 1)


class BurgerEnv(gym.Env):
	# Live, File
	metadata = {'render.modes': ['l', 'f']}

	def get_fields_and_labels(self):
		ndim = len(self.shape)

		if ndim == 1:
			fields = [np.real(self.cont_state.velocity).reshape(-1),
					#np.real(self.prec_state.velocity).reshape(-1),
					#np.real(self.pass_state.velocity).reshape(-1),
					np.real(self.init_state.velocity).reshape(-1),
					self.goal_obs.reshape(-1)
					]

			labels = ['Controlled Simulation',
					#'Ground Truth Simulation',
					#'Uncontrolled Simulation',
					'Initial Density Field',
					'Goal Density Field']
		elif ndim == 2:
			fields = [self.cont_state.velocity,
					self.init_state.velocity,
					self.goal_obs.reshape([1] + list(self.goal_obs.shape))]
			labels = ['Controlled Simulation',
					'Initial Density Field',
					'Goal Density Field']
		else:
			raise NotImplementedError()
		
		return fields, labels

	def get_random_state(self):
		#global prebaked_velocity
		#return phi.flow.Burger(phi.flow.Domain(self.shape), prebaked_velocity, viscosity=0.2)
		return phi.flow.Burger(phi.flow.Domain(self.shape), phi.flow.math.randn(levels=[0,0,self.vel_scale]), viscosity=0.2)

	def step_sim(self, state, forces):
		tic = time.time()
		controlled_state = state.copied_with(velocity=state.velocity + forces.reshape(state.velocity.shape) * self.delta_time)
		s = self.physics.step(controlled_state, self.delta_time)
		toc = time.time()
		#print(toc - tic)
		return s

	def __init__(self, epis_len=32, dt=0.5, vel_scale=1.0, use_time=False,
			name='v0', act_type=util.ActionType.DISCRETE_2, use_l1_loss=False,
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
		self.physics = phi.flow.BurgerPhysics()
		self.action_recorder = util.ActionRecorder()
		self.action_space = util.get_action_space(act_type, act_dim)
		self.observation_space = util.get_observation_space(act_params.shape, goal_type, len(self.shape), use_time)
		self.vis_extractor = lambda s: np.squeeze(np.real(s.velocity), axis=0)
		self.force_gen = util.get_force_gen(act_type, act_params, self.get_random_state().velocity.shape, synchronized)
		self.goal_gen = util.get_goal_gen(self.force_gen, self.step_sim, 
			self.vis_extractor, self.get_random_state,
			act_type, goal_type, act_params.shape, act_dim, epis_len, self.action_recorder)
		self.obs_gen = util.get_obs_gen(goal_type, use_time, epis_len)
		self.rew_gen = util.get_rew_gen(rew_type, rew_force_factor, self.epis_len, use_l1_loss)
		self.cont_state = None
		self.pass_state = None
		self.init_state = None
		self.prec_state = None
		self.goal_obs = None
		self.lviz = None
		self.fviz = None
		self.force_collector = None
		self.ref_force_collector = None

	def reset(self):
		if self.action_recorder is None:
			self.action_recorder = util.ActionRecorder()
		self.action_recorder.reset()

		self.cont_state = self.get_random_state()
		self.pass_state = self.cont_state.copied_with()
		self.init_state = self.cont_state.copied_with()
		self.prec_state = self.cont_state.copied_with()
		self.goal_obs = self.goal_gen(self.init_state.copied_with())
		self.step_idx = 0
		
		if self.force_collector is not None:
			print('Average forces: %f' % self.force_collector.get_forces())

		if self.ref_force_collector is not None:
			print('Average forces reference: %f' % self.ref_force_collector.get_forces())

		return self.obs_gen(self.vis_extractor(self.cont_state), self.goal_obs, self.step_idx)

	def step(self, action):
		tic = time.time()
		self.step_idx += 1
		v_old = self.vis_extractor(self.cont_state)

		forces = self.force_gen(action).copy()

		self.cont_state = self.step_sim(self.cont_state, forces)

		f_prec = self.force_gen(self.action_recorder.replay()).copy()

		self.prec_state = self.step_sim(self.prec_state, f_prec)
		self.pass_state = self.physics.step(self.pass_state, self.delta_time)

		v_new = self.vis_extractor(self.cont_state)

		err_old = self.goal_obs - v_old
		err_new = self.goal_obs - v_new

		obs = self.obs_gen(self.vis_extractor(self.cont_state), self.goal_obs, self.step_idx)
		done = self.step_idx == self.epis_len
		reward = self.rew_gen(err_old, err_new, forces, done)

		if self.force_collector is not None:
			self.force_collector.add_forces(forces)

		if self.ref_force_collector is not None:
			self.ref_force_collector.add_forces(f_prec)

		if done:
			self.epis_idx += 1

		toc = time.time()
		#print(toc - tic)

		return obs, reward, done, {}

	def render(self, mode='l'):
		if self.force_collector is None:
			self.force_collector = util.ForceCollector()

		if self.ref_force_collector is None:
			self.ref_force_collector = util.ForceCollector()
		
		fields, labels = self.get_fields_and_labels()

		ndim = len(self.shape)
		max_value = 4
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
