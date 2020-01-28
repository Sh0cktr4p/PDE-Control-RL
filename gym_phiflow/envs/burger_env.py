import gym
import os, subprocess, time, signal
from gym import error, spaces, utils
from gym.utils import seeding
from phi.flow import *

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

	@property
	def action_space(self):
		return gym.spaces.Discrete(2)

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
		return np.zeros_like(self.state.velocity).reshape(-1)

	def create_forces(self, action):
		if len(action.shape) != 1:
			action = np.array([action])

		assert len(action.shape) == 1, "Action should be a vector"
		assert action.shape[0] == self.state.velocity.shape[-1], "Action has wrong number of dimensions, exp: %i, got: %i" % (self.state.velocity.shape[3], action.shape[0])
		
		action = action * 2 - 1
	
		forces = np.zeros(self.state.velocity.shape, dtype=np.float32)
		forces[0, 0, 0] = action

		return forces

	def calc_reward(self, mse_old, mse_new, forces):
		return -mse_new

	def __init__(self):
		self.step_index = 0
		self.episode_index = 0
		self.state = Burger(Domain(self.size), math.randn(levels=[0, 0, self.value_velocity_scale]), viscosity=0.2)
		self.physics = BurgerPhysics()
		self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(np.prod(list(self.size)),), dtype=np.float32)
		self.goal = self.create_goal()
		self.viewer = None
		self.figures = None
		self.scene = None       # Scene is only created when simulation data is saved
		self.image_dir = None

	def step(self, action):
		v_old = self.state.velocity.reshape(-1)

		forces = self.create_forces(action)

		# Apply action
		self.state = self.state.copied_with(velocity=self.state.velocity + forces * self.delta_time)
		# Perform pde step
		self.state = self.physics.step(self.state, self.delta_time)
		self.step_index += 1

		v_new = self.state.velocity.reshape(-1)

		# Calculate reward as the decrease in the main squared error
		mse_old = np.sum((self.goal - v_old) ** 2)
		mse_new = np.sum((self.goal - v_new) ** 2)
		
		reward = self.calc_reward(mse_old, mse_new, forces)

		done = self.step_index == self.ep_len

		if done:
			self.episode_index += 1

		return self.state.velocity.astype(np.float32).reshape(-1), reward, done, {}

	def reset(self):
		self.state = Burger(Domain(self.size), math.randn(levels=[0, 0, self.value_velocity_scale]), viscosity=0.2)
		self.goal = self.create_goal()
		self.step_index = 0

		return self.state.velocity.astype(np.float32).reshape(-1)

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

	def render(self, mode='file'):
		if mode == 'human':
			tic = time.time()
			frame_rate = 1 / 15

			if self.viewer is None:
				from gym_phiflow.envs import rendering
				self.viewer = rendering.SimpleImageViewer()
			
			rgb = self.field_to_rgb(max_value=2)

			self.viewer.imshow(rgb, 500, 500)
			toc = time.time()
			sleep_time = frame_rate - (toc - tic)
			if sleep_time > 0:
				time.sleep(sleep_time)

		if mode == 'file':
			if self.figures is None:
				self.episode_index = 0
				self.create_scene()
				self.figures = PlotlyFigureBuilder(batches=None)        

			files = []

			# Record Images
			os.path.isdir(self.image_dir) or os.makedirs(self.image_dir)
			files += self.figures.save_figures(self.image_dir, 'Velocity%04i' % self.episode_index, self.step_index, self.state.velocity)

			# Record Data
			files += phi.data.fluidformat.write_sim_frame(self.scene.path, self.state.velocity, 'Velocity', self.step_index)

			if files:
				print('Frame written to %s' % files)
			self.current_action = None

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None

	def seed(self):
		pass
