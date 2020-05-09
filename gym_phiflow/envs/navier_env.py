from gym_phiflow.envs import util, visualization, shape_field
import phi.flow
import gym
import numpy as np
import time


default_act_points = util.act_points((16,), 0)


def pad_to_staggered_size(field):
	return np.pad(field, [(0,int(v!=1)) for v in field.shape])


def stack_fields(state):
	field_stack = np.append(state.velocity.staggered, pad_to_staggered_size(state.density), axis=-1)
	return np.squeeze(field_stack, axis=0)


def with_channel(shape):
	return tuple(list(shape) + [1])


def get_vis_extractor(all_visible):
	if all_visible:
		return lambda s: stack_fields(s)
	else:
		return lambda s: np.squeeze(s.density, axis=0)


def pad_to_shape(field, shape):
	return np.pad(field, [(0, gs-fs) for fs, gs in zip(field.shape, shape)])


class NavierEnv(gym.Env):
	# Live, File
	metadata = {'render.modes': ['l', 'f']}

	def get_fields_and_labels(self):
		ndim = len(self.den_shape)

		if ndim == 1:
			fields = [self.cont_state.density.reshape(-1),
					self.pass_state.density.reshape(-1),
					self.init_state.density.reshape(-1),
					self.goal_obs.reshape(-1)]

			labels = ['Controlled Simulation',
					'Uncontrolled Simulation',
					'Initial Density Field',
					'Goal Density Field']
		elif ndim == 2:
			fields = [self.cont_state.density,
					self.init_state.density,
					self.goal_obs.reshape([1] + list(self.goal_obs.shape))]
			labels = ['Controlled Simulation',
					'Initial Density Field',
					'Goal Density Field']
		else:
			raise NotImplementedError()
		
		return fields, labels

	def get_state_with(self, value):
		return phi.flow.Smoke(phi.flow.Domain(self.den_shape, phi.flow.SLIPPERY), density=value, buoyancy_factor=0.0, conserve_density=True)

	def get_random_state(self):
		return phi.flow.Smoke(phi.flow.Domain(self.den_shape, phi.flow.SLIPPERY), density=phi.flow.math.randn(levels=[self.den_scale]), buoyancy_factor=0.0, conserve_density=True)

	def get_init_field_gen(self, init_field_gen):
		if init_field_gen:
			return lambda: self.get_state_with(shape_field.to_density_field(init_field_gen(), 10))
		else:
			return self.get_random_state

	def combine_to_obs(self, state, goal):
		reshaped_state_obs = self.vis_extractor(state)
		reshaped_goal_obs = pad_to_shape(goal, list(reshaped_state_obs.shape[:-1]) + [1])
		return self.obs_gen(reshaped_state_obs, reshaped_goal_obs, self.step_idx)

	def step_sim(self, state, forces):
		staggered_forces = phi.flow.math.StaggeredGrid(forces.reshape(state.velocity.staggered.shape))
		controlled_state = state.copied_with(velocity=state.velocity + staggered_forces * self.delta_time)
		return self.physics.step(controlled_state, self.delta_time)

	def __init__(self, epis_len=32, dt=0.5, den_scale=1.0, use_time=False, 
			name='v0', act_type=util.ActionType.DISCRETE_2, act_points=default_act_points, 
			goal_type=util.GoalType.ZERO, rew_type=util.RewardType.ABSOLUTE, rew_force_factor=1, 
			synchronized=False, init_field_gen=None, goal_field_gen=None, all_visible=False, sdf_rew=False):
		# Multi-dimensional fields have parameters for each of these directions at each point; act_points does not reflect that
		act_params = util.get_all_act_params(act_points)
		
		act_dim = len(act_points.shape) if synchronized else np.sum(act_params)

		self.step_idx = 0
		self.epis_idx = 0
		self.epis_len = epis_len
		self.delta_time = dt
		self.den_scale = den_scale
		self.exp_name = name
		self.physics = phi.flow.SmokePhysics(make_input_divfree=True, make_output_divfree=True)

		# Density field is one smaller in every dimension than the velocity field
		self.den_shape = tuple(d-1 for d in act_points.shape)
		# If both fields are visible, pad the density field and stack it onto the staggered velocity field
		vis_shape = util.increase_channels(act_params.shape, 1) if all_visible else with_channel(self.den_shape)
		# Goal field has the shape of the density field with one channel dimension
		goal_vis_shape = with_channel(self.den_shape)
		# Forces array has the same shape as the velocity field
		forces_shape = self.get_random_state().velocity.staggered.shape

		self.action_space = util.get_action_space(act_type, act_dim)
		self.observation_space = util.get_observation_space(vis_shape, goal_type, 1, use_time)
		self.force_gen = util.get_force_gen(act_type, act_params, forces_shape, synchronized)
		
		self.shape_mode = goal_field_gen is not None
		self.sdf = None
		self.sdf_rew = sdf_rew
		self.init_gen = self.get_init_field_gen(init_field_gen)
		self.goal_gen = util.get_goal_gen(self.force_gen, self.step_sim,
			lambda s: s.density.reshape(goal_vis_shape), self.get_random_state, act_type, goal_type, 
			goal_vis_shape, act_dim, epis_len, goal_field_gen)

		self.vis_extractor = get_vis_extractor(all_visible)
		self.obs_gen = util.get_obs_gen(goal_type, use_time, epis_len)
		self.rew_gen = util.get_rew_gen(rew_type, rew_force_factor, epis_len, False)
		self.cont_state = None	# Controlled state
		self.pass_state = None	# Passive state
		self.init_state = None	# Initial state
		self.goal_obs = None
		self.lviz = None
		self.fviz = None

	def reset(self):
		self.init_state = self.init_gen()
		self.cont_state = self.init_state.copied_with()
		self.pass_state = self.init_state.copied_with()
		self.goal_obs = self.goal_gen(self.init_state.copied_with())

		if self.shape_mode:
			self.sdf = self.goal_obs
			self.goal_obs = shape_field.to_density_field(self.goal_obs, 10)

		self.step_idx = 0
		
		return self.combine_to_obs(self.cont_state, self.goal_obs)

	def step(self, action):
		tic = time.time()
		self.step_idx += 1
		
		old_obs = np.squeeze(self.cont_state.density, axis=0)

		forces = self.force_gen(action)

		self.cont_state = self.step_sim(self.cont_state, forces)
		self.pass_state = self.physics.step(self.pass_state, self.delta_time)

		new_obs = np.squeeze(self.cont_state.density, axis=0)

		if self.sdf_rew:
			err_old = old_obs * self.sdf
			err_new = new_obs * self.sdf
		else:
			err_old = self.goal_obs - old_obs
			err_new = self.goal_obs - new_obs

		obs = self.combine_to_obs(self.cont_state, self.goal_obs)
		done = self.step_idx == self.epis_len
		reward = self.rew_gen(err_old, err_new, forces, done)

		if done:
			self.epis_idx += 1

		toc = time.time()
		#print(toc - tic)

		return obs, reward, done, {}

	def render(self, mode='l'):
		fields, labels = self.get_fields_and_labels()

		ndim = len(self.den_shape)
		max_value = 1
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
