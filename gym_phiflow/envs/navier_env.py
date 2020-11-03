from gym_phiflow.envs import util, visualization, shape_field
import phi.tf.flow as phiflow
import gym
import phi.tf.tf_cuda_pressuresolver
import numpy as np
import time
import tensorflow as tf


default_act_points = util.act_points((16,), 0)


def pad_to_staggered_size(field):
	return np.pad(field, [(0,int(v!=1)) for v in field.shape])


def stack_fields(state):
	field_stack = np.append(state.velocity.staggered_tensor(), pad_to_staggered_size(state.density.data), axis=-1)
	return np.squeeze(field_stack, axis=0)


def with_channel(shape):
	return tuple(list(shape) + [1])


def get_vis_extractor(all_visible):
	if all_visible:
		return lambda s: stack_fields(s)
	else:
		return lambda s: np.squeeze(pad_to_staggered_size(s.density.data), axis=0)
		#return lambda s: np.squeeze(s.density.data, axis=0)


def pad_to_shape(field, shape):
	return np.pad(field, [(0, gs-fs) for fs, gs in zip(field.shape, shape)])


class NavierEnv(gym.Env):
	# Live, File
	metadata = {'render.modes': ['l', 'f']}

	def get_fields_and_labels(self):
		ndim = len(self.den_shape)

		if ndim == 1:
			fields = [self.cont_state.density.data.reshape(-1),
					self.pass_state.density.data.reshape(-1),
					self.init_state.density.data.reshape(-1),
					self.goal_obs.reshape(-1)]

			labels = ['Controlled Simulation',
					'Uncontrolled Simulation',
					'Initial Density Field',
					'Goal Density Field']
		elif ndim == 2:
			fields = [self.cont_state.density.data,
					self.init_state.density.data,
					self.goal_obs.reshape([1] + list(self.goal_obs.shape))]
			labels = ['Controlled Simulation',
					'Initial Density Field',
					'Goal Density Field']
		else:
			raise NotImplementedError()
		
		return fields, labels

	def get_state_with(self, value):
		return phiflow.Fluid(phiflow.Domain(self.den_shape, boundaries=phiflow.CLOSED), density=value, buoyancy_factor=0.0)

	def get_random_state(self):
		return phiflow.Fluid(phiflow.Domain(self.den_shape, boundaries=phiflow.CLOSED), density=phiflow.Noise(), buoyancy_factor=0.0)

	def get_init_field_gen(self, init_field_gen):
		if init_field_gen:
			return lambda: self.get_state_with(shape_field.to_density_field(init_field_gen(), self.den_scale))
		else:
			return self.get_random_state

	def combine_to_obs(self, state, goal):
		reshaped_state_obs = self.vis_extractor(state)
		reshaped_goal_obs = pad_to_shape(goal, list(reshaped_state_obs.shape[:-1]) + [1])
		return self.obs_gen(reshaped_state_obs, reshaped_goal_obs, self.step_idx)

	def step_sim(self, state, forces):
		staggered_forces = phiflow.field.StaggeredGrid(forces.reshape(state.velocity.staggered_tensor().shape))
		controlled_state = state.copied_with(velocity=state.velocity + staggered_forces * self.delta_time)

		return self.tf_physics_step(controlled_state)
		#return self.physics.step(controlled_state, self.delta_time)

	def __init__(self, epis_len=32, dt=0.5, den_scale=1.0, 
			name='v0', act_type=util.ActionType.DISCRETE_2, act_points=default_act_points, 
			goal_type=util.GoalType.ZERO, rew_type=util.RewardType.ABSOLUTE, rew_force_factor=1, loss_fn=util.l2_loss,
			synchronized=False, init_field_gen=None, goal_field_gen=None, all_visible=False, sdf_rew=False):
		act_points = np.squeeze(act_points)
		# Multi-dimensional fields have parameters for each of these directions at each point; act_points does not reflect that
		act_params = util.get_all_act_params(act_points)
		
		act_dim = act_points.ndim if synchronized else np.sum(act_params)

		self.step_idx = 0
		self.epis_idx = 0
		self.epis_len = epis_len
		self.delta_time = dt
		self.den_scale = den_scale
		self.exp_name = name
		self.physics = phiflow.IncompressibleFlow(pressure_solver=phiflow.GeometricCG())#(pressure_solver=phi.tf.tf_cuda_pressuresolver.CUDASolver())
		#self.physics = phiflow.IncompressibleFlow()

		# Density field is one smaller in every dimension than the velocity field
		self.den_shape = tuple(d-1 for d in act_points.shape)
		# If both fields are visible, pad the density field and stack it onto the staggered velocity field
		vis_shape = util.increase_channels(act_params.shape, 1) if all_visible else with_channel(act_points.shape)
		# Goal field has the shape of the density field with one channel dimension
		goal_vis_shape = with_channel(self.den_shape)
		# Forces array has the same shape as the velocity field
		forces_shape = self.get_random_state().velocity.staggered_tensor().shape

		self.action_recorder = util.get_action_recorder(goal_type)

		self.action_space = util.get_action_space(act_type, act_dim)
		self.observation_space = util.get_observation_space(vis_shape, goal_type, 1)

		self.force_gen = util.get_force_gen(act_type, act_params, forces_shape, synchronized)
		
		self.shape_mode = goal_field_gen is not None
		self.sdf = None
		self.sdf_rew = sdf_rew
		self.init_gen = self.get_init_field_gen(init_field_gen)
		self.goal_gen = util.get_goal_gen(self.force_gen, self.step_sim,
			lambda s: s.density.data.reshape(goal_vis_shape), self.get_random_state, act_type, goal_type, 
			goal_vis_shape, act_dim, epis_len, self.action_recorder, goal_field_gen)
		self.vis_extractor = get_vis_extractor(all_visible)
		self.obs_gen = util.get_obs_gen(goal_type, epis_len)
		self.rew_gen = util.get_rew_gen(rew_type, rew_force_factor, epis_len, loss_fn)
		self.cont_state = None	# Controlled state
		self.pass_state = None	# Passive state
		self.init_state = None	# Initial state
		self.prec_state = None	# Ground truth
		self.goal_obs = None
		self.lviz = None
		self.fviz = None
		self.force_collector = None
		self.test_mode = False

		sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)))
		sim_in_ph = self.init_gen().copied_with(density=phiflow.placeholder, velocity=phiflow.placeholder)
		sim_out_ph = self.physics.step(sim_in_ph, self.delta_time)
		sess.run(tf.global_variables_initializer())
		sess.graph.finalize()
		get_feed_dict = lambda s: {k:v for (k, v) in zip(phiflow.struct.flatten(sim_in_ph), phiflow.struct.flatten(s))}
		self.tf_physics_step = lambda s: phiflow.struct.unflatten(sess.run(phiflow.struct.flatten(sim_out_ph), feed_dict=get_feed_dict(s)), s)

	def reset(self):
		if self.action_recorder is not None:
			self.action_recorder.reset()

		self.cont_state = self.init_gen()
		self.goal_obs = self.goal_gen(self.cont_state.copied_with())

		if self.test_mode:
			self.init_state = self.cont_state.copied_with()
			self.pass_state = self.cont_state.copied_with()
			self.prec_state = self.cont_state.copied_with()
			
			print('Average forces: %f' % self.force_collector.get_forces())

		if self.shape_mode:
			self.sdf = self.goal_obs
			self.goal_obs = shape_field.to_density_field(self.goal_obs, self.den_scale)
			
		self.step_idx = 0

		return self.combine_to_obs(self.cont_state, self.goal_obs)

	def step(self, action):
		self.step_idx += 1
		
		old_obs = np.squeeze(self.cont_state.density.data, axis=0)

		forces = self.force_gen(action).copy()

		self.cont_state = self.step_sim(self.cont_state, forces)
		
		if self.test_mode:
			if self.action_recorder is not None:
				f_prec = self.force_gen(self.action_recorder.replay()).copy()
				self.prec_state = self.step_sim(self.prec_state, f_prec)

			self.pass_state = self.tf_physics_step(self.pass_state)
			#self.pass_state = self.physics.step(self.pass_state, self.delta_time)
			
			self.force_collector.add_forces(forces)

		new_obs = np.squeeze(self.cont_state.density.data, axis=0)

		if self.sdf_rew:
			err_old = old_obs * self.sdf
			err_new = new_obs * self.sdf
		else:
			err_old = self.goal_obs - old_obs
			err_new = self.goal_obs - new_obs

		obs = self.combine_to_obs(self.cont_state, self.goal_obs)
		done = self.step_idx == self.epis_len
		reward = self.rew_gen(err_old, err_new, forces, done)

		print(reward)

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

		ndim = len(self.den_shape)
		max_value = 0.25
		signed = False

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
			category_name = 'SpinningNavier-%s' % self.exp_name
			if self.fviz is None:
				if ndim == 1:
					self.fviz = visualization.FilePlotter(category_name)
				elif ndim == 2:
					self.fviz = visualization.FileRenderer(category_name)
			self.fviz.render(fields, labels, max_value, signed, 'Density', 
				self.epis_idx, self.step_idx, self.epis_len, remove_frames)
		else:
			raise NotImplementedError()

	def close(self):
		pass

	def seed(self):
		pass
