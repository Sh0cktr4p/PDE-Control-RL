from gym_phiflow.envs.navier_env import NavierEnv
from gym_phiflow.envs import util, shape_field
import numpy as np

twenty400 = util.act_points(size=(20,20), indices=tuple(zip(*[(i, 0) for i in range(20)])))
#complete256 = np.ones(shape=(65, 65), dtype=np.bool)
#complete256 = util.act_points(size=(65,65), indices=tuple(zip(*[(i, 0) for i in range(65)])))
block256 = util.act_points(size=(16,16), indices=tuple(zip(*[(i,j) for i in range(5,11) for j in range(5,11)])))
complete64 = util.act_points(size=(8, 8), indices=tuple(zip(*[(i,j) for i in range(8) for j in range(8)])))
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

class NavierEnvEverything2DShapesObsFinSmoothed(NavierEnv):
	def __init__(self):
		super().__init__(name='v308', act_type=util.ActionType.CONTINUOUS,
			act_points=complete256, goal_type=util.GoalType.PREDEFINED,
			rew_type=util.RewardType.FIN_APPR, rew_force_factor=0.02,
			init_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1),
			goal_field_gen=lambda: shape_field.get_random_sdf_field((15, 15)).reshape(1, 15, 15, 1),
			all_visible=True)

class NavierEnvEverything2DShapesObsSqSDFSmoothed(NavierEnv):
	def __init__(self):
		super().__init__(name='v309', act_type=util.ActionType.CONTINUOUS, 
			act_points=complete256, goal_type=util.GoalType.PREDEFINED,
			rew_type=util.RewardType.ABS_FORC, rew_force_factor=0.02, loss_fn=util.l1_loss,
			init_field_gen=lambda: shape_field.get_random_sdf_field((15, 15), squared=True).reshape(1, 15, 15, 1),
			goal_field_gen=lambda: shape_field.get_random_sdf_field((15, 15), squared=True).reshape(1, 15, 15, 1),
			all_visible=True, sdf_rew=True)

class NavierEnvEverything2DShapesObsSqSDFFinSmoothed(NavierEnv):
	def __init__(self):
		super().__init__(name='v310', act_type=util.ActionType.CONTINUOUS, 
			act_points=complete256, goal_type=util.GoalType.PREDEFINED,
			rew_type=util.RewardType.FIN_APPR, rew_force_factor=0.1, loss_fn=util.l1_loss,
			init_field_gen=lambda: shape_field.get_random_sdf_field((15, 15), squared=True).reshape(1, 15, 15, 1),
			goal_field_gen=lambda: shape_field.get_random_sdf_field((15, 15), squared=True).reshape(1, 15, 15, 1),
			all_visible=True, sdf_rew=True)

class NavierEnvEverything2DShapesObsSqSmoothedSimple(NavierEnv):
	def __init__(self):
		super().__init__(name='v311', act_type=util.ActionType.CONTINUOUS,
			act_points=complete64, goal_type=util.GoalType.PREDEFINED,
			rew_type=util.RewardType.ABS_FORC, rew_force_factor=0.02,
			init_field_gen=lambda: shape_field.get_random_sdf_field((7, 7), shape_field.shapes_basic, True).reshape(1, 7, 7, 1),
			goal_field_gen=lambda: shape_field.get_random_sdf_field((7, 7), shape_field.shapes_basic, True).reshape(1, 7, 7, 1),
			all_visible=True)

class NavierEnvEverything2DShapesObsSqSmoothedCheap(NavierEnv):
	def __init__(self):
		super().__init__(name='v312', act_type=util.ActionType.CONTINUOUS,
			act_points=complete64, goal_type=util.GoalType.PREDEFINED,# 1 1 -> 5 5
			rew_type=util.RewardType.ABS_FORC, rew_force_factor=0.0,
			init_field_gen=lambda: shape_field.Rect(2, 2).get_sdf_field((7, 7), np.array([1, 1])).reshape(1, 7, 7, 1),
			goal_field_gen=lambda: shape_field.Rect(2, 2).get_sdf_field((7, 7), np.array([5, 5])).reshape(1, 7, 7, 1),
			all_visible=True)

class NavierEnvEverything2DShapesObsSDFSmoothed(NavierEnv):
	def __init__(self):
		super().__init__(name='v313', act_type=util.ActionType.CONTINUOUS, 
			act_points=complete256, goal_type=util.GoalType.PREDEFINED,
			rew_type=util.RewardType.ABS_FORC, rew_force_factor=0.01, loss_fn=util.l1_loss,
			init_field_gen=lambda: shape_field.get_random_sdf_field((15, 15), squared=False).reshape(1, 15, 15, 1),
			goal_field_gen=lambda: shape_field.get_random_sdf_field((15, 15), squared=False).reshape(1, 15, 15, 1),
			all_visible=True, sdf_rew=True)

class NavierEnvEverything2DShapesObsBalanced(NavierEnv):
	def __init__(self):
		super().__init__(name='v314', act_type=util.ActionType.CONTINUOUS, 
			act_points=complete256, goal_type=util.GoalType.PREDEFINED,
			rew_type=util.RewardType.ABSOLUTE, 
			init_field_gen=lambda: shape_field.get_random_sdf_field((15, 15), shape_field.shapes_basic_big, squared=False).reshape(1, 15, 15, 1),
			goal_field_gen=lambda: shape_field.get_random_sdf_field((15, 15), shape_field.shapes_basic_big, squared=False).reshape(1, 15, 15, 1),
			all_visible=True, rew_balancing=True)