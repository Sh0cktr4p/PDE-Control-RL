import gym
from enum import Enum
import numpy as np
from gym_phiflow.envs import burgers_util

class ActionType(Enum):
	CONTINUOUS = 0			# Forces can be any floating point value
	UNMODIFIED = 1			# No forces are applied
	DISCRETE_2 = 2			# Forces can be +1 or -1
	DISCRETE_3 = 3			# Forces can be +1, 0 or -1


class GoalType(Enum):
	ZERO = 0				# Goal field is zero
	RANDOM = 1				# Goal field is random other initial state
	REACHABLE = 2			# Goal field is precomputed by applying random actions at each step
	PREDEFINED = 3			# Custom goal creation routine
	CONSTANT_FORCE = 4		# Goal field is precomputed by applying the same random action at each field value and time step
	GAUSS_FORCE = 5			# Goal field is precomputed by applying a set of gaussian distributed field values at each time step


class RewardType(Enum):
	ABSOLUTE = 0			# Consider only absolute deviation from goal field
	RELATIVE = 1			# Consider the change in deviation from goal field
	ABS_FORC = 2			# Consider absolute deviation from goal field and the amount of forces created
	FIN_APPR = 3			# Consider forces at every timestep but the approximation only in the end
	FIN_NOFC = 4


# Assembles forces array
# indices:		indices of action parameters, locations where actions are applied
# forces:		forces array matching the dimensions of the field onto which they should be applied
# actions:		array of actions to take at each index
#
# returns:		finished forces array
def create_forces(indices, forces, actions):
	if actions.size == forces.shape[-1] and actions.size != indices[0].size:
		actions = np.repeat([actions], indices[0].size // actions.size, axis=0)
	forces[indices] = actions.reshape(-1)
	return forces


# Decodes n-ary discrete actions
# action: 		the action to decode
# point count:	number of controllable field points, number of digits of decoded action value
# action count:	number of discrete actions possible per action point, base of decoded action value
#
# returns:	 	array of (point_count) values ranging from zero to (action_count-1)
def decode_discrete_action(action, point_count, action_count):
	return action // action_count ** np.arange(point_count) % action_count


# Converts encoded actions to forces
# act_type:		enum value describing the space of possible actions
# act_params:	bool array defining which parameters in the field are controlled
# forces_shape:	tuple describing the shape of the field onto which the forces are added
# synchronized:	flag determining whether action parameters are controlled by just one action for all
#
# returns:		lambda function creating a forces array to a given action
def get_force_gen(act_type, act_params, forces_shape, synchronized):
	act_params = act_params.reshape(forces_shape)

	indices = np.where(act_params)
	forces = np.zeros(forces_shape)

	act_size = act_params.shape[-1] if synchronized else np.sum(act_params)

	if act_type == ActionType.CONTINUOUS:
		return lambda a: create_forces(indices, forces, a)
	elif act_type == ActionType.UNMODIFIED:
		return lambda a: np.zeros(forces_shape)
	elif act_type == ActionType.DISCRETE_2:
		return lambda a: create_forces(indices, forces, decode_discrete_action(a, act_size, 2) * 2 - 1)
	elif act_type == ActionType.DISCRETE_3:
		return lambda a: create_forces(indices, forces, decode_discrete_action(a, act_size, 3) - 1)
	else:
		raise NotImplementedError()


# Determines the output space of the network
# Discrete action spaces -> Classifier for n-ary encoded actions
# Continuous action spaces -> Regression, each output neuron describes one action
# act_type: 	enum value describing the space of possible actions
# act_dim:		number of action parameters
#
# returns:		gym space describing network output
def get_action_space(act_type, act_dim):
	if act_type == ActionType.CONTINUOUS:
		return gym.spaces.Box(-np.inf, np.inf, shape=(act_dim,), dtype=np.float32)
	elif act_type == ActionType.DISCRETE_2:
		return gym.spaces.Discrete(2 ** act_dim)
	elif act_type == ActionType.DISCRETE_3:
		return gym.spaces.Discrete(3 ** act_dim)
	else:
		raise NotImplementedError()


# Determines the input space of the network
# vis_shape:		shape of the observable part of the current state
# goal_type:		enum value indicating if the goal state should be appended to the observation space
# goal_channels:	determines how many channels should be added if the goal is included in the observation space
# use_time:			whether the network should be presented with the index of the current time step
#
# returns:		gym space describing network input
def get_observation_space(vis_shape, goal_type, goal_channels, use_time):
	if goal_type == GoalType.ZERO:
		obs_shape = vis_shape
	elif goal_type == GoalType.RANDOM or goal_type == GoalType.REACHABLE \
			or goal_type == GoalType.PREDEFINED or goal_type == GoalType.CONSTANT_FORCE \
			or goal_type == GoalType.GAUSS_FORCE:
		obs_shape = increase_channels(vis_shape, goal_channels)
	else:
		raise NotImplementedError()

	if use_time:
		obs_shape = increase_channels(obs_shape, 1)

	return gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)


# Precomputes a trajectory with actions provided by a generator function
# action_gen:		custom action generator function
# force_gen: 		output function of a get_force_gen() call, transforms actions to forces array
# step_fn:			function provided by environment to apply forces and generate new states by a physics object
# vis_extractor:	function provided by environment extracting the observable part from a given state
# act_rec:			action recorder, used to retrace the ground truth
# epis_len:			length of a trajectory
# state:			initial state for simulation
# act_preselected:	flag to show if the same action should be applied at all time steps
#
# returns:			visible part of a goal state from a precomputed trajectory
def run_trajectory(action_gen, force_gen, step_fn, vis_extractor, act_rec, epis_len, state, act_preselected):
	if act_preselected:
		action = action_gen()
		action_gen = lambda: action
	
	for _ in range(epis_len):
		current_action = action_gen()
		act_rec.record(current_action)
		forces = force_gen(current_action)
		state = step_fn(state, forces)

	return vis_extractor(state)
	

# Yields a function for generating random actions mimicking network outputs
# act_type:			enum value describing the space of possible actions
# act_dim:			number of action parameters
# enf_disc:			flag to decide whether to always output discrete numbers regardless of act_type
#
# returns:			random action generator lambda function
def get_act_gen(act_type, act_dim, enf_disc=False):
	if act_type == ActionType.CONTINUOUS:
		if enf_disc:
			return lambda: np.random.randint(low=-1, high=2, size=act_dim)
		else:
			return lambda: np.repeat(np.random.normal(0, 0.25), act_dim)
	elif act_type == ActionType.UNMODIFIED:
		return lambda: np.zeros(shape=(act_dim,))
	elif act_type == ActionType.DISCRETE_2:
		return lambda: np.random.randint(2 ** act_dim)
	elif act_type == ActionType.DISCRETE_3:
		return lambda: np.random.randint(3 ** act_dim)
	else:
		raise NotImplementedError()


def gaussian(dim, loc, amp, sig):
	res = amp * np.exp(-0.5 * (np.arange(dim) / dim - loc) ** 2 / sig ** 2)
	#print(np.sum(res))
	return res #amp * np.exp(-0.5 * (np.arange(dim) / dim - loc) ** 2 / sig ** 2)


def gaussian_clash(dim, l_loc, l_amp, l_sig, r_loc, r_amp, r_sig):
	return gaussian(dim, l_loc, l_amp, l_sig) + gaussian(dim, r_loc, r_amp, r_sig)


def get_gauss_act_gen(act_dim):
	return lambda: gaussian(act_dim, np.random.uniform(0.4, 0.6), np.random.uniform(-0.05, 0.05) * act_dim, np.random.uniform(0.1, 0.4))


def get_gauss_clash_gen(dim):
	return lambda: gaussian_clash(dim, np.random.uniform(0.2, 0.4), np.random.uniform(0, 3), np.random.uniform(0.05, 0.15), np.random.uniform(0.6, 0.8), np.random.uniform(-3, 0), np.random.uniform(0.05, 0.15))


# Returns a function to generate a goal field for a given initial input state
# force_gen:		output function of a get_force_gen() call, transforms actions to forces array
# step_fn:			function provided by environment to apply forces and generate new states by a physics object
# vis_extractor:	function provided by environment extracting the observable part from a goal state
# rand_state_gen:	function provided by environment to create custom random states
# act_type:			enum value describing the space of possible actions
# goal_type:		enum value distinguishing the kinds of goal states
# vis_shape:		shape of the observable part of the goal state
# act_dim:			number of action parameters
# epis_len:			length of a trajectory
# act_rec:			action recorder, used to retrace the ground truth
# goal_field_gen:	custom goal field generation routine, only has effect with corresponding goal type
#
# returns:			lambda goal generator function
def get_goal_gen(force_gen, step_fn, vis_extractor, rand_state_gen, act_type, 
		goal_type, vis_shape, act_dim, epis_len, act_rec, goal_field_gen=None):
	if goal_type == GoalType.ZERO:
		return lambda s: np.zeros(shape=vis_shape, dtype=np.float32)
	elif goal_type == GoalType.RANDOM:
		return lambda s: vis_extractor(rand_state_gen())
	elif goal_type == GoalType.REACHABLE:
		action_gen = get_act_gen(act_type, act_dim, enf_disc=True)
		return lambda s: run_trajectory(action_gen, force_gen, step_fn, vis_extractor, act_rec, epis_len, s, False)
	elif goal_type == GoalType.PREDEFINED:
		assert goal_field_gen is not None
		return lambda s: np.squeeze(goal_field_gen(), axis=0)
	elif goal_type == GoalType.CONSTANT_FORCE:
		action_gen = get_act_gen(act_type, act_dim, enf_disc=False)
		return lambda s: run_trajectory(action_gen, force_gen, step_fn, vis_extractor, act_rec, epis_len, s, True)
	elif goal_type == GoalType.GAUSS_FORCE:
		action_gen = get_gauss_act_gen(act_dim)
		return lambda s: run_trajectory(action_gen, force_gen, step_fn, vis_extractor, act_rec, epis_len, s, True)
	else:
		raise NotImplementedError()


# Creates a action points bool array to specify the points where to apply forces
# size:		tuple describing the size of the controlled field
# indices:	tuple of lists specifying the controllable parameters
#
# returns:	bool array describing field size and where actions can be applied
def act_points(size, indices):
	act = np.zeros(size, dtype=np.bool)
	act[indices] = True
	return act


# Generator returning functions to asseble the observation space
# goal_type:	enum value indicating if the goal state should be appended to the observation space
# epis_len:		length of a trajectory, used to normalize time values when supplied
# use_time:		whether the network should be presented with the current time
#
# returns:		lambda network input assembly function
def get_obs_gen(goal_type, epis_len, use_time):
	if goal_type == GoalType.ZERO:
		if use_time:
			return lambda v, g, t: np.concatenate((v, np.full(v.shape, t / epis_len)), axis=-1)
		else:
			return lambda v, g, t: v
	elif goal_type == GoalType.RANDOM or goal_type == GoalType.REACHABLE \
			or goal_type == GoalType.PREDEFINED or goal_type == GoalType.CONSTANT_FORCE \
			or goal_type == GoalType.GAUSS_FORCE:
		if use_time:
			return lambda v, g, t: np.concatenate((v, g, np.full(v.shape, t / epis_len)), axis=-1)
		else:
			return lambda v, g, t: np.concatenate((v, g), axis=-1)
	else:
		raise NotImplementedError()


def l1_loss(field):
	return np.sum(np.abs(field))


def l2_loss(field):
	return np.sum(field**2)

# Returns reward functions corresponding to specifications
# rew_type:		enum value distinguishing the types of reward functions
# force_factor:	value to further control the significance of forces amount for reward values
# loss_fn:		loss function to use
#
# returns: 		lambda reward function
def get_rew_gen(rew_type, force_factor, epis_len, loss_fn):
	if rew_type == RewardType.ABSOLUTE:
		return lambda o, n, f, e: -loss_fn(n)
	elif rew_type == RewardType.RELATIVE:
		return lambda o, n, f, e: loss_fn(o) - loss_fn(n)
	elif rew_type == RewardType.ABS_FORC:
		return lambda o, n, f, e: -loss_fn(f) * force_factor - loss_fn(n) 
	elif rew_type == RewardType.FIN_APPR:
		return lambda o, n, f, e: -loss_fn(f) * force_factor - (epis_len * loss_fn(n) if e else 0)
	elif rew_type == RewardType.FIN_NOFC:
		return lambda o, n, f, e: -epis_len * loss_fn(n) if e else 0
	else:
		raise NotImplementedError()


# Expands action points array to encorporate action parameters in every available direction
# If a field value is controllable, it should be controllable in every available axis
# points: 		bool array describing at which points in the field forces can be applied
# 
# returns: 		bool array describing at which points in the field forces can be applied in which directions
def get_all_act_params(points):
	points = np.squeeze(points)
	return np.stack([points for _ in range(points.ndim)], points.ndim)


def increase_channels(shape, add_channels):
	return tuple(list(shape[:-1]) + [shape[-1] + add_channels])


class ForceCollector:
	def __init__(self, epis_len):
		self.total_forces = 0
		self.num_steps = 0
		self.epis_len = epis_len
		self.force_hist = []
		self.curr_ep_forces = 0

	def add_forces(self, forces):
		forces = np.sum(np.abs(forces))
		self.total_forces += forces
		self.curr_ep_forces += forces
		self.num_steps += 1

	def get_forces(self):
		return self.total_forces * self.epis_len / self.num_steps

	def get_force_hist(self):
		self.force_hist.append(self.curr_ep_forces)
		self.curr_ep_forces = 0
		return self.force_hist


class ActionRecorder:
	def __init__(self):
		self.actions = None
		self.step_idx = 0

	def reset(self):
		self.actions = []
		self.step_idx = 0

	def record(self, action):
		self.actions.append(action)

	def replay(self):
		action = self.actions[self.step_idx]
		self.step_idx += 1
		return action

	def overall_amt(self):
		return np.sum(np.abs(self.actions))


def get_action_recorder(goal_type):
	if goal_type == GoalType.ZERO or goal_type == GoalType.RANDOM or goal_type == GoalType.PREDEFINED:
		return None
	elif goal_type == GoalType.REACHABLE or goal_type == GoalType.CONSTANT_FORCE or goal_type == GoalType.GAUSS_FORCE:
		return ActionRecorder()
	else:
		raise NotImplementedError()