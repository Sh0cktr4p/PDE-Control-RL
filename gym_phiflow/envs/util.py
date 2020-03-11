import gym
from enum import Enum
import numpy as np
import phi.flow


class ActionType(Enum):
	CONTINUOUS = 0
	UNMODIFIED = 1
	DISCRETE_2 = 2
	DISCRETE_3 = 3


class GoalType(Enum):
	ZERO = 0
	RANDOM = 1
	REACHABLE = 2
	PREDEFINED = 3
	CONSTANT_FORCE = 4


class RewardType(Enum):
	ABSOLUTE = 0
	RELATIVE = 1
	ABS_FORC = 2


def create_forces(indices, forces, actions):
	forces[indices] = actions.reshape(-1)
	return forces

def decode_action(action, point_count, action_count):
	return action // action_count ** np.arange(point_count) % action_count

def get_force_gen(act_type, act_points, forces_shape):
	act_points = act_points.reshape(forces_shape)

	indices = np.where(act_points)
	forces = np.zeros(forces_shape)

	if act_type == ActionType.CONTINUOUS:
		return lambda a: create_forces(indices, forces, a)
	elif act_type == ActionType.UNMODIFIED:
		return lambda a: np.zeros(forces_shape)
	elif act_type == ActionType.DISCRETE_2:
		return lambda a: create_forces(indices, forces, decode_action(a, np.sum(act_points), 2) * 2 - 1)
	elif act_type == ActionType.DISCRETE_3:
		return lambda a: create_forces(indices, forces, decode_action(a, np.sum(act_points), 3) - 1)
	else:
		raise NotImplementedError()


def get_action_space(act_type, act_dim):
	if act_type == ActionType.CONTINUOUS:
		return gym.spaces.Box(-np.inf, np.inf, shape=(act_dim,), dtype=np.float32)
	elif act_type == ActionType.DISCRETE_2:
		return gym.spaces.Discrete(2 ** act_dim)
	elif act_type == ActionType.DISCRETE_3:
		return gym.spaces.Discrete(3 ** act_dim)
	else:
		raise NotImplementedError()


def get_observation_space(field_size, goal_type, use_time):
	obs_size = 0

	if goal_type == GoalType.ZERO:
		obs_size = field_size
	elif goal_type == GoalType.RANDOM or goal_type == GoalType.REACHABLE \
		or goal_type == GoalType.PREDEFINED or goal_type == GoalType.CONSTANT_FORCE:
		obs_size = field_size * 2
	else:
		raise NotImplementedError()

	if use_time:
		obs_size += 1

	return gym.spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)


def run_trajectory(force_gen, action_gen, step_fn, vis_extractor, ep_len, state):
	for _ in range(ep_len):
		forces = force_gen(action_gen())
		state = step_fn(state, forces)

	return vis_extractor(state)


def get_act_gen(act_type, act_dim, enf_disc=False):
	if act_type == ActionType.CONTINUOUS:
		if enf_disc:
			return lambda: np.random.randint(low=-1, high=2, size=act_dim)
		else:
			return lambda: np.repeat(np.random.normal(0, 0.5), act_dim)
	elif act_type == ActionType.UNMODIFIED:
		return lambda: np.zeros(shape=(act_dim,))
	elif act_type == ActionType.DISCRETE_2:
		return lambda: np.random.randint(2 ** act_dim)
	elif act_type == ActionType.DISCRETE_3:
		return lambda: np.random.randint(3 ** act_dim)
	else:
		raise NotImplementedError()


def get_goal_gen(force_gen, step_fn, vis_extractor, rand_state_gen, act_type, 
		goal_type, vis_size, act_dim, ep_len, goal_field_gen=None):
	if goal_type == GoalType.ZERO:
		return lambda s: np.zeros(shape=(vis_size,), dtype=np.float32)
	elif goal_type == GoalType.RANDOM:
		return lambda s: vis_extractor(rand_state_gen())
	elif goal_type == GoalType.REACHABLE:
		action_gen = get_act_gen(act_type, act_dim, enf_disc=True)
		return lambda s: run_trajectory(force_gen, action_gen, step_fn, vis_extractor, ep_len, s)
	elif goal_type == GoalType.PREDEFINED:
		assert goal_field_gen is not None
		return lambda s: goal_field_gen()
	elif goal_type == GoalType.CONSTANT_FORCE:
		action_gen = get_act_gen(act_type, act_dim, enf_disc=False)
		return lambda s: run_trajectory(force_gen, lambda: action_gen(), step_fn, vis_extractor, ep_len, s)
	else:
		raise NotImplementedError()


def act_points(size, indices):
	act = np.zeros(size, dtype=np.bool)
	act[indices] = True
	return act


def get_obs_gen(goal_type, use_time, epis_len):
	if goal_type == GoalType.ZERO:
		if use_time:
			return lambda v, g, t: np.append(v, t / epis_len)
		else:
			return lambda v, g, t: v
	elif goal_type == GoalType.RANDOM or goal_type == GoalType.REACHABLE \
			or goal_type == GoalType.PREDEFINED or goal_type == GoalType.CONSTANT_FORCE:
		if use_time:
			return lambda v, g, t: np.append(np.concatenate((v, g), axis=0), t)
		else:
			return lambda v, g, t: np.concatenate((v, g), axis=0)
	else:
		raise NotImplementedError()


def get_rew_gen(rew_type, force_factor):
	if rew_type == RewardType.ABSOLUTE:
		return lambda o, n, f: -n
	elif rew_type == RewardType.RELATIVE:
		return lambda o, n, f: o - n
	elif rew_type == RewardType.ABS_FORC:
		return lambda o, n, f: -(n + np.sum(f ** 2) * force_factor)
	else:
		raise NotImplementedError()


def get_all_act_params(points):
	points = np.squeeze(points)
	return np.squeeze(np.stack([points for _ in range(points.ndim)], points.ndim))
