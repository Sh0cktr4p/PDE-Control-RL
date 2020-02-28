import time
import gym
from phi.flow import *

class NavierEnv(gym.Env):
	metadata = {'render.modes': ['human', 'file', 'live_plot']}

# =========================== EXPERIMENT VARIATIONS =================
	def forces_2_discrete(self, action, forces):
		action = action * 2 - 1
		forces[0, 0, 0] = action
		return forces

	def get_force_generator(self):
		act = self.action_space
		if act == self.discrete_2_space():
			return self.forces_2_discrete
		else: 
			raise NotImplementedError()

	def create_forces(self, action):
		action = np.array(action)
		forces = np.zeros(self.state.velocity.staggered.shape, dtype=np.float32)

		return self.force_generator(action, forces)

	def step_sim(self, state, forces):
		assert state.velocity.staggered.shape == forces.shape, 'Forces array has wrong size'

		controlled_state = state.copied_with(velocity=state.velocity + phi.flow.math.StaggeredGrid(forces))

		return self.physics.step(controlled_state, self.delta_time)

	def density_to_rgb(self, max_value=1):
		obs = self.state.density
		
		assert obs.shape[-1] < 3, "3D visualization not (yet) supported"

		height, width = self.state.density.shape[-3:-1]

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

	def show_field_render(self):
		frame_rate = 1 / 15
		tic = time.time()

		if self.viewer is None:
			from gym_phiflow.envs import rendering
			self.viewer = rendering.SimpleImageViewer()

		rgb = self.density_to_rgb(max_value=1)

		self.viewer.imshow(rgb, 500, 500)
		toc = time.time()

		sleep_time = frame_rate - (toc - tic)

		if sleep_time > 0:
			time.sleep(sleep_time)

	def plot_to_file(self):
		pass

	def show_plot(self):
		pass

# =========================== CORE METHODS ==========================
	def __init__(self, size=(16,), ep_len=32, dt=0.5, 
			act=gym.spaces.Discrete(2)):
		self.size = size
		self.episode_length = ep_len
		self.delta_time = dt
		self.action_space = act
		self.step_index = 0
		self.episode_index = 0
		self.physics = SmokePhysics()
		self.observation_space = self.get_obs_space()
		self.force_generator = self.get_force_generator()
		self.state = None
		self.ref_state = None
		self.init_state = None
		self.goal_obs = None
		self.viewer = None
		self.fig = None
		self.plots = None
		self.scene = None
		self.image_dir = None

	def step(self, action):
		forces = self.create_forces(action)

		old_obs = self.state.density.reshape(-1)

		forces = self.create_forces(action)

		self.state = self.step_sim(self.state, forces)
		self.ref_state = self.physics.step(self.ref_state, self.delta_time)

		new_obs = self.state.density.reshape(-1)

		mse_old = np.sum((self.goal_obs - old_obs) ** 2)
		mse_new = np.sum((self.goal_obs - new_obs) ** 2)

		obs = self.get_obs()

		reward = self.calc_reward(mse_old, mse_new, forces)

		done = self.step_index == self.episode_length

		if done:
			self.episode_index += 1

		return obs, reward, done, {}


	def reset(self):
		self.state = Smoke(Domain(self.state))
		self.ref_state = self.state.copied_with()
		self.init_state = self.state.copied_with()
		self.goal_obs = self.create_goal()
		self.step_index = 0

		return self.get_obs()

	def render(self, mode='human'):
		if mode == 'human':
			self.show_field_render()
		elif mode == 'file':
			self.plot_to_file()
		elif mode == 'live_plot':
			self.show_plot()
		else:
			raise NotImplementedError()

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None

	def seed(self):
		pass

env = NavierEnv()

env.reset()
for _ in range(200):
	env.step(0)
	env.render()