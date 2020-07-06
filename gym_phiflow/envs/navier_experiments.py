from gym_phiflow.envs.navier_env import NavierEnv
from gym_phiflow.envs import util, shape_field
import numpy as np

twenty400 = util.act_points(size=(20,20), indices=tuple(zip(*[(i, 0) for i in range(20)])))
#complete256 = np.ones(shape=(65, 65), dtype=np.bool)
#complete256 = util.act_points(size=(65,65), indices=tuple(zip(*[(i, 0) for i in range(65)])))
block256 = util.act_points(size=(16,16), indices=tuple(zip(*[(i,j) for i in range(5,11) for j in range(5,11)])))
complete256 = util.act_points(size=(16, 16), indices=tuple(zip(*[(i,j) for i in range(16) for j in range(16)])))
block1089 = util.act_points(size=(17,17), indices=tuple(zip(*[(i, j) for i in range(6,11) for j in range(6,11)])))

class NavierEnvTwo(NavierEnv):
	def __init__(self):
		super().__init__(name='v00', goal_type=util.GoalType.RANDOM, all_visible=True)

class NavierEnvContTwenty2DReachable(NavierEnv):
	def __init__(self):
		super().__init__(name='v12', act_type=util.ActionType.CONTINUOUS,
			act_points=twenty400, goal_type=util.GoalType.REACHABLE,
			rew_type=util.RewardType.ABS_FORC)

class NavierEnvContComplete2DShapes(NavierEnv):
	def __init__(self):
		super().__init__(name='v14', act_type=util.ActionType.CONTINUOUS,
			act_points=block256, goal_type=util.GoalType.PREDEFINED,
			rew_type=util.RewardType.ABSOLUTE, synchronized=True,
			init_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1), 
			goal_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1))

class NavierEnvContComplete2DShapesObs(NavierEnv):
	def __init__(self):
		super().__init__(name='v16', act_type=util.ActionType.CONTINUOUS,
			act_points=block256, goal_type=util.GoalType.PREDEFINED,
			rew_type=util.RewardType.ABSOLUTE, synchronized=True,
			init_field_gen=lambda: shape_field.get_random_sdf_field((15,15)).reshape(1,15,15,1),
			goal_field_gen=lambda: shape_field.get_random_sdf_field((15,15)).reshape(1,15,15,1),
			all_visible=True)

class NavierEnvContCompleteConstant2DShapesObs(NavierEnv):
	def __init__(self):
		super().__init__(name='v300', act_type=util.ActionType.CONTINUOUS,
			act_points=block256, goal_type=util.GoalType.CONSTANT_FORCE,
			rew_type=util.RewardType.ABSOLUTE, synchronized=True,
			init_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1),
			all_visible=True)

class NavierEnvContCompleteConstant2DShapesObsFin(NavierEnv):
	def __init__(self):
		super().__init__(name='v301', act_type=util.ActionType.CONTINUOUS,
			act_points=block256, goal_type=util.GoalType.CONSTANT_FORCE,
			rew_type=util.RewardType.FIN_NOFC, synchronized=True,
			init_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1),
			all_visible=True)

class NavierEnvContComplete2DShapesObsSDF(NavierEnv):
	def __init__(self):
		super().__init__(name='v302', act_type=util.ActionType.CONTINUOUS,
			act_points=block256, goal_type=util.GoalType.PREDEFINED,
			rew_type=util.RewardType.ABSOLUTE, synchronized=True,
			init_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1),
			goal_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1),
			all_visible=True, sdf_rew=True)

class NavierEnvContComplete2DShapesObsFinSDF(NavierEnv):
	def __init__(self):
		super().__init__(name='v303', act_type=util.ActionType.CONTINUOUS,
			act_points=block256, goal_type=util.GoalType.PREDEFINED,
			rew_type=util.RewardType.FIN_NOFC, synchronized=True,
			init_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1),
			goal_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1),
			all_visible=True, sdf_rew=True)

class NavierEnvEverything2DShapesObsFinSDF(NavierEnv):
	def __init__(self):
		super().__init__(name='v304', act_type=util.ActionType.CONTINUOUS,
			act_points=complete256, goal_type=util.GoalType.PREDEFINED,
			rew_type=util.RewardType.FIN_NOFC,
			init_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1),
			goal_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1),
			all_visible=True, sdf_rew=True)

class NavierEnvEverything2DShapesObsSDF(NavierEnv):
	def __init__(self):
		super().__init__(name='v305', act_type=util.ActionType.CONTINUOUS,
			act_points=complete256, goal_type=util.GoalType.PREDEFINED,
			rew_type=util.RewardType.ABSOLUTE,
			init_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1),
			goal_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1),
			all_visible=True, sdf_rew=True)

class NavierEnvEverything2DShapesObs(NavierEnv):
	def __init__(self):
		super().__init__(name='v306', act_type=util.ActionType.CONTINUOUS,
			act_points=complete256, goal_type=util.GoalType.PREDEFINED,
			rew_type=util.RewardType.ABSOLUTE,
			init_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1),
			goal_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1),
			all_visible=True)

class NavierEnvEverything2DShapes(NavierEnv):
	def __init__(self):
		super().__init__(name='v307', act_type=util.ActionType.CONTINUOUS,
			act_points=complete256, goal_type=util.GoalType.PREDEFINED,
			rew_type=util.RewardType.ABSOLUTE,
			init_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1),
			goal_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1))