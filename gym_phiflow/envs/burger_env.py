import gym
import os, subprocess, time, signal
from gym import error, spaces, utils
from gym.utils import seeding
from phi.flow import *
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# Simplest Burger control environment:
#   - Actor has the possibility to add a force to one value of the velocity field
#   - Goal is to make all values of the field approach zero
#   - 1D and 2D fields supported
#
# 2 Methods of visualization: 
#   1) Realtime rendering of the field with a modified version of the renderer
#       from the gym classic control environments module
#   2) Rendering to both image and data files as provided by standard phiflow
#
class BurgerEnv(gym.Env):
	metadata = {'render.modes': ['human', 'file']}

	# =========================== POSSIBLE EXPERIMENT VARIATIONS ===============================
	def discrete_2_space(self):
		return gym.spaces.Discrete(2)

	def discrete_3_space(self):
		return gym.spaces.Discrete(3)

	def discrete_3_3_space(self):
		return gym.spaces.Discrete(27)

	def discrete_5_5_space(self):
		return gym.spaces.Discrete(25)

	def continuous_8_space(self):
		return gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)

	def continuous_complete_field_space(self):
		return gym.spaces.Box(-np.inf, np.inf, shape=(np.prod(list(self.size)),), dtype=np.float32)

	def continuous_double_size_space(self):
		return gym.spaces.Box(-np.inf, np.inf, shape=(2 * np.prod(list(self.size)),), dtype=np.float32)

	def zero_goal_obs(self):
		return np.zeros_like(self.state.velocity).reshape(-1)

	def random_goal_obs(self):
		return Burger(Domain(self.size), math.randn(levels=[0, 0, self.value_velocity_scale]), viscosity=0.2).velocity.reshape(-1)

	def reachable_goal_obs(self):
		goal_obs = self.state.copied_with(velocity=self.state.velocity)

		force_gen = lambda: 0

		if self.action_space == self.discrete_2_space():
			force_gen = lambda: np.random.randint(2)
		elif self.action_space == self.discrete_3_space():
			force_gen = lambda: np.random.randint(3)
		elif self.action_space == self.discrete_3_3_space():
			force_gen = lambda: np.random.randint(27)
		elif self.action_space == self.continuous_8_space():
			force_gen = lambda: np.random.randint(low=-1, high=2, size=(8,))
		elif self.action_space == self.continuous_complete_field_space():
			# Applies random integer forces to complete field
			force_gen = lambda: np.random.randint(3, size=self.state.velocity.size)
		else:
			raise NotImplementedError()

		for _ in range(self.ep_len):
			forces = self.create_forces(force_gen())
			goal_obs = self.step_sim(goal_obs, forces)

		return goal_obs.velocity.reshape(-1)

	def mse_new_reward(self, mse_old, mse_new, forces):
		return -mse_new

	def relative_reward(self, mse_old, mse_new, forces):
		return mse_old - mse_new

	def mse_new_and_forces_reward(self, mse_old, mse_new, forces):
		force_strength = np.sum(forces ** 2)
		return -(mse_new + force_strength)

	def current_field_obs(self):
		return np.real(self.state.velocity.reshape(-1))

	def current_and_goal_obs(self):
		return np.concatenate((self.current_field_obs(), self.goal_obs), axis=0)

	# =========================== CHANGEABLE PROPERTIES ========================================
	@property
	def action_space(self):
		return self.discrete_2_space()

	@property
	def observation_space(self):
		return self.continuous_complete_field_space()

	@property
	def size(self):
		return (32,)

	@property
	def value_velocity_scale(self):
		return 1.0

	@property
	def ep_len(self):
		return 32

	@property
	def delta_time(self):
		return 0.5
	
	def create_goal(self):
		return self.zero_goal_obs()

	def calc_reward(self, mse_old, mse_new, forces):
		return self.mse_new_reward(mse_old, mse_new, forces)

	def get_obs(self):
		if self.observation_space == self.continuous_complete_field_space():
			return self.current_field_obs()
		elif self.observation_space == self.continuous_double_size_space():
			return self.current_and_goal_obs()
		else:
			raise NotImplementedError()

	# =========================== INTERNAL HELPER METHODS ======================================
	def insert_n(self, action, forces):
		start_point = int(0.5 * (forces.shape[-2] - action.shape[0]))
		forces[0, start_point:start_point + action.shape[0], 0] = action

	def forces_2_discrete(self, action, forces):
		action = action * 2 - 1
		forces[0, 0, 0] = action

	def forces_3_discrete(self, action, forces):
		action = action - 1
		forces[0, 0, 0] = action

	def forces_3_3_discrete(self, action, forces):
		action = np.array(list(np.base_repr(action, 3).rjust(3, '0')), dtype=np.float32) - 1
		self.insert_n(action, forces)

	def forces_8_continuous(self, action, forces):
		self.insert_n(action, forces)

	def forces_complete_continuous(self, action, forces):
		forces = action.reshape(forces.shape)

	def create_forces(self, action):
		action = np.array(action)
		forces = np.zeros(self.state.velocity.shape, dtype=np.float32)

		if self.action_space == self.discrete_2_space():
			action = action * 2 - 1
			forces[0, 0, 0] = action
		elif self.action_space == self.discrete_3_space():
			action = action - 1
			forces[0, 0, 0] = action
		elif self.action_space == self.discrete_3_3_space():
			assert forces.shape[-2] >= 3
			action = np.array(list(np.base_repr(action, 3).rjust(3, '0')), dtype=np.float32) - 1
			start_point = int(0.5 * (forces.shape[-2] - 3))
			forces[0, start_point:start_point+3, 0] = action
		elif self.action_space == self.continuous_8_space():
			assert forces.shape[-2] >= 8
			start_point = int(0.5 * (forces.shape[-2] - 3))
			forces[0, start_point:start_point+8, 0] = action
		elif self.action_space == self.continuous_complete_field_space():
			forces = action.reshape(forces.shape)
		else:
			raise NotImplementedError()

		return forces

	def step_sim(self, state, forces):
		assert state.velocity.shape == forces.shape, 'Forces Array has wrong shape!'

		controlled_state = state.copied_with(velocity=state.velocity + forces * self.delta_time)

		return self.physics.step(controlled_state, self.delta_time)

	def create_scene(self):
		self.scene = Scene.create(os.path.expanduser('~/phi/data/'), 'BurgerGym', count=1, mkdir=True)
		self.image_dir = self.scene.subpath('images')

	def field_to_rgb(self, max_value=2):
		obs = self.state.velocity
		
		assert obs.shape[-1] < 3, "3D visualization not (yet) supported"

		height, width = self.state.velocity.shape[-3:-1]

		# Visualization should display field vector length
		obs = np.linalg.norm(obs, axis=-1).reshape(height, width)

		# Get the color values
		r = np.clip((obs + max_value) / (2.0 * max_value), 0.0, 1.0)
		b = 1.0 - r

		r = np.rint(255 * r**2).astype(np.uint8)
		g = np.zeros_like(r)
		b = np.rint(255 * b**2).astype(np.uint8)

		# Convert into single color array
		return np.transpose([r, g, b], (1, 2, 0))

	def get_figure(self):
		x = np.arange(self.size[0])

		self.fig = plt.figure()
		plt.ylim(-2.0, 2.0)
		init_plot, = plt.plot(x, self.init_state.velocity.reshape(-1), label='Initial Field')
		goal_plot, = plt.plot(x, self.goal_obs, label='Goal Field')
		csim_plot, = plt.plot(x, self.state.velocity.reshape(-1), label='Controlled Simulation')
		rsim_plot, = plt.plot(x, self.ref_state.velocity.reshape(-1), label='Uncontrolled Simulation')
		plt.legend(loc='upper right')

		return init_plot, goal_plot, csim_plot, rsim_plot

	def show_field_render(self):
		frame_rate = 1 / 15
		tic = time.time()

		if self.viewer is None:
			from gym_phiflow.envs import rendering
			self.viewer = rendering.SimpleImageViewer()
			
		rgb = self.field_to_rgb(max_value=2)

		self.viewer.imshow(rgb, 500, 500)
		toc = time.time()
		sleep_time = frame_rate - (toc - tic)
		if sleep_time > 0:
			time.sleep(sleep_time)
	
	def plot_to_file(self):
		if self.fig is None:			
			self.episode_index = 0
			self.create_scene()

		files = []

		# Record Images
		os.path.isdir(self.image_dir) or os.makedirs(self.image_dir)

		self.get_figure()

		path = os.path.join(self.image_dir, '%s_batch%04d_%04d.png' % ('Velocity%04i' % self.episode_index, 0, self.step_index))
		plt.savefig(path)
		plt.close()

		files += [path]

		# Record Data
		files += phi.data.fluidformat.write_sim_frame(self.scene.path, self.state.velocity, 'Velocity', self.step_index)

		if files:
			print('Frame written to %s' % files)

	def show_plot(self):
		if self.fig is None:
			plt.ion()
			self.plots = self.get_figure()

		if self.step_index == 0:
			self.plots[0].set_ydata(self.init_state.velocity.reshape(-1))
			self.plots[1].set_ydata(self.goal_obs)

		self.plots[2].set_ydata(self.state.velocity.reshape(-1))
		self.plots[3].set_ydata(self.ref_state.velocity.reshape(-1))

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()

	# =========================== CORE METHODS =================================================
	def __init__(self):
		self.step_index = 0
		self.episode_index = 0
		self.physics = BurgerPhysics()
		self.state = Burger(Domain(self.size), math.randn(levels=[0, 0, self.value_velocity_scale]), viscosity=0.2)
		self.ref_state = self.state.copied_with(velocity=self.state.velocity)
		self.init_state = self.state.copied_with(velocity=self.state.velocity)
		self.goal_obs = self.create_goal()
		self.viewer = None
		self.fig = None
		self.plots = None
		self.scene = None       # Scene is only created if simulation data is saved
		self.image_dir = None

	def step(self, action):
		self.step_index += 1

		v_old = self.state.velocity.reshape(-1)

		forces = self.create_forces(action)

		self.state = self.step_sim(self.state, forces)
		self.ref_state = self.physics.step(self.ref_state, self.delta_time)

		v_new = self.state.velocity.reshape(-1)

		# Calculate reward as the decrease in the main squared error
		mse_old = np.sum((self.goal_obs - v_old) ** 2)
		mse_new = np.sum((self.goal_obs - v_new) ** 2)
		
		obs = self.get_obs()

		reward = self.calc_reward(mse_old, mse_new, forces)

		done = self.step_index == self.ep_len

		if done:
			self.episode_index += 1

		return obs, reward, done, {}

	def reset(self):
		self.state = Burger(Domain(self.size), math.randn(levels=[0, 0, self.value_velocity_scale]), viscosity=0.2)
		self.ref_state = self.state.copied_with(velocity=self.state.velocity)
		self.init_state = self.state.copied_with(velocity=self.state.velocity)
		self.goal_obs = self.create_goal()
		self.step_index = 0

		return self.get_obs()

	def render(self, mode='live_plot'):
		if mode == 'human':
			self.show_field_render()
		elif mode == 'file' and len(self.size) == 1:
			self.plot_to_file()
		elif mode == 'live_plot' and len(self.size) == 1:
			self.show_plot()
		else:
			raise NotImplementedError()

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None

	def seed(self):
		pass
