from gym_phiflow.envs.navier_env import NavierEnv
from gym_phiflow.envs import util, shape_field
import numpy as np

twenty400 = util.act_points(size=(20,20), indices=tuple(zip(*[(i, 0) for i in range(20)])))
#complete256 = np.ones(shape=(65, 65), dtype=np.bool)
#complete256 = util.act_points(size=(65,65), indices=tuple(zip(*[(i, 0) for i in range(65)])))
block1089 = util.act_points(size=(17,17), indices=tuple(zip(*[(i, j) for i in range(6,11) for j in range(6,11)])))


class NavierEnvTwo(NavierEnv):
	def __init__(self):
		super().__init__(name='v00')

class NavierEnvContTwenty2DReachable(NavierEnv):
	def __init__(self):
		super().__init__(name='v12', act_type=util.ActionType.CONTINUOUS,
			act_points=twenty400, goal_type=util.GoalType.REACHABLE,
			rew_type=util.RewardType.ABS_FORC)

class NavierEnvContComplete2DShapes(NavierEnv):
	def __init__(self):
		super().__init__(name='v14', act_type=util.ActionType.CONTINUOUS,
			act_points=block1089, goal_type=util.GoalType.PREDEFINED,
			rew_type=util.RewardType.ABSOLUTE, synchronized=True,
			init_field_gen=lambda: shape_field.get_random_field((16, 16)).reshape(1, 16, 16, 1), 
			goal_field_gen=lambda: shape_field.get_random_field((16, 16)).reshape(1, 16, 16, 1))

class NavierEnvContComplete2DShapesObs(NavierEnv):
	def __init__(self):
		super().__init__(name='v16', act_type=util.ActionType.CONTINUOUS,
			act_points=block1089, goal_type=util.GoalType.PREDEFINED,
			rew_type=util.RewardType.ABSOLUTE, synchronized=True,
			init_field_gen=lambda: shape_field.get_random_field((16, 16)).reshape(1, 16, 16, 1),
			goal_field_gen=lambda: shape_field.get_random_field((16, 16)).reshape(1, 16, 16, 1),
			all_visible=True)