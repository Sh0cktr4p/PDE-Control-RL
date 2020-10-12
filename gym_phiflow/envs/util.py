import gym
from enum import Enum
import numpy as np
import phi.flow


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


class RewardType(Enum):
	ABSOLUTE = 0			# Consider only absolute deviation from goal field
	RELATIVE = 1			# Consider the change in deviation from goal field
	ABS_FORC = 2			# Consider absolute deviation from goal field and the amount of forces created
	FIN_APPR = 3			# Consider forces at every timestep but the approximation only in the end
	FIN_NOFC = 4			# Consider approximation in the end and no forces


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
def decode_action(action, point_count, action_count):
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
		return lambda a: create_forces(indices, forces, decode_action(a, act_size, 2) * 2 - 1)
	elif act_type == ActionType.DISCRETE_3:
		return lambda a: create_forces(indices, forces, decode_action(a, act_size, 3) - 1)
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
# use_time:			flag indicating if the network should get information about the current position in time
#
# returns:		gym space describing network input
def get_observation_space(vis_shape, goal_type, goal_channels, use_time):
	if goal_type == GoalType.ZERO:
		obs_shape = vis_shape
	elif goal_type == GoalType.RANDOM or goal_type == GoalType.REACHABLE \
		or goal_type == GoalType.PREDEFINED or goal_type == GoalType.CONSTANT_FORCE:
		obs_shape = increase_channels(vis_shape, goal_channels)
	else:
		raise NotImplementedError()

	if use_time:
		raise NotImplementedError()

	return gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)


# Precomputes a trajectory with actions provided by a generator function
# action_gen:		custom action generator function
# force_gen: 		output function of a get_force_gen() call, transforms actions to forces array
# step_fn:			function provided by environment to apply forces and generate new states by a physics object
# vis_extractor:	function provided by environment extracting the observable part from a given state
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
		act = action_gen()
		act_rec.record(act)
		forces = force_gen(act)
		state = step_fn(state, forces)

	return vis_extractor(state)


#[1, 1, 2, 2, 1, 0, 1, 1, 0, 0, 1, 1, 2, 2, 0, 2, 2, 1, 0, 1, 1, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2]
#[1, 1, 1, 2, 0, 0, 1, 2, 0, 1, 0, 2, 0, 1, 2, 1, 1, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 2, 2, 2, 1]
#[3, 44, 23, 54, 41, 57, 71, 31, 57, 5, 51, 40, 43, 54, 60, 50, 61, 20, 28, 57, 66, 12, 60, 69, 4, 66, 47, 18, 43, 76, 55, 29]
#[36, 36, 2, 51, 21, 49, 38, 46, 8, 1, 23, 33, 72, 37, 51, 45, 2, 42, 16, 49, 16, 28, 52, 28, 8, 45, 78, 10, 10, 63, 67, 67]
#[1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
prebaked_actions = [1, 1, 1, 2, 0, 2, 1, 2, 1, 1, 1, 1, 2, 0, 1, 0, 1, 2, 0, 1, 1, 0, 2, 0, 1, 1, 0, 2, 1, 0, 0, 0]
action_index = 0

def pick_action(cont):
	global action_index
	global prebaked_actions
	action = prebaked_actions[action_index]
	action_index = (action_index + 1) % 32
	if cont:
		action = decode_action(action, 1, 3) - 1
	return action

# Yields a function for generating random actions mimicking network outputs
# act_type:			enum value describing the space of possible actions
# act_dim:			number of action parameters
# enf_disc:			flag to decide whether to always output discrete numbers regardless of act_type
#
# returns:			random action generator lambda function
def get_act_gen(act_type, act_dim, enf_disc=False):
	if act_type == ActionType.CONTINUOUS:
		if enf_disc:
			#return lambda: np.array(pick_action(True))
			return lambda: np.random.randint(low=-1, high=2, size=act_dim)
		else:
			#return lambda: np.repeat(-0.087, act_dim)
			return lambda: np.repeat(np.random.normal(0, 0.1), act_dim)
	elif act_type == ActionType.UNMODIFIED:
		return lambda: np.zeros(shape=(act_dim,))
	elif act_type == ActionType.DISCRETE_2:
		return lambda: np.random.randint(2 ** act_dim)
	elif act_type == ActionType.DISCRETE_3:
		#return lambda: pick_action(False)
		return lambda: np.random.randint(3 ** act_dim)
	else:
		raise NotImplementedError()


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
# act_rec:			action recorder to reconstruct the goal creation process
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
# use_time:		flag indicating if the network should get information about the current position in time
# epis_len:		length of a trajectory, used to normalize time values when supplied
#
# returns:		lambda network input assembly function
def get_obs_gen(goal_type, use_time, epis_len):
	if use_time:
		raise NotImplementedError()

	if goal_type == GoalType.ZERO:
		return lambda v, g, t: v
	elif goal_type == GoalType.RANDOM or goal_type == GoalType.REACHABLE \
			or goal_type == GoalType.PREDEFINED or goal_type == GoalType.CONSTANT_FORCE:
		return lambda v, g, t: np.append(v, g, axis=-1)
	else:
		raise NotImplementedError()


# Returns the l1 or l2 loss of a field
# field:		the field containing the difference values
# use_l1_loss:	flag to determine whether l1 or l2 loss should be employed
def apply_loss(field, use_l1_loss):
	if use_l1_loss:
		return np.sum(np.abs(field))
	else:
		return np.sum(field ** 2)


# Returns reward functions corresponding to specifications
# rew_type:		enum value distinguishing the types of reward functions
# force_factor:	value to further control the significance of forces amount for reward values
# use_l1_loss:	flag determining whether l1 or l2 loss should be used
#
# returns: 		lambda reward function
def get_rew_gen(rew_type, force_factor, epis_len, use_l1_loss):
	if rew_type == RewardType.ABSOLUTE:
		return lambda o, n, f, e: -apply_loss(n, use_l1_loss)
	elif rew_type == RewardType.RELATIVE:
		return lambda o, n, f, e: apply_loss(o, use_l1_loss) - apply_loss(n, use_l1_loss)
	elif rew_type == RewardType.ABS_FORC:
		return lambda o, n, f, e: -apply_loss(n, use_l1_loss) - apply_loss(f, use_l1_loss) * force_factor
	elif rew_type == RewardType.FIN_APPR:
		return lambda o, n, f, e: -apply_loss(f, use_l1_loss) * force_factor - (epis_len * apply_loss(n, use_l1_loss) if e else 0)
	elif rew_type == RewardType.FIN_NOFC:
		return lambda o, n, f, e: -epis_len * apply_loss(n, use_l1_loss) if e else 0
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
	def __init__(self):
		self.total_forces = 0
		self.num_steps = 0

	def add_forces(self, forces):
		self.total_forces += np.sum(np.abs(forces))
		self.num_steps += 1

	def get_forces(self):
		return self.total_forces * 32.0 / self.num_steps


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


def get_action_recorder(goal_type):
	if goal_type == GoalType.ZERO or goal_type == GoalType.RANDOM or goal_type == GoalType.PREDEFINED:
		return None
	elif goal_type == GoalType.REACHABLE or goal_type == GoalType.CONSTANT_FORCE:
		return ActionRecorder()
	else:
		raise NotImplementedError()