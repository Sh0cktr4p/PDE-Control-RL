from gym_phiflow.envs.navier_env import NavierEnv
from gym_phiflow.envs import util, shape_field
import numpy as np

twenty400 = util.act_points(size=(20,20), indices=tuple(zip(*[(i, 0) for i in range(20)])))
#complete256 = np.ones(shape=(65, 65), dtype=np.bool)
complete256 = util.act_points(size=(65,65), indices=tuple(zip(*[(i, 0) for i in range(65)])))

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
			act_points=complete256, goal_type=util.GoalType.PREDEFINED,
			rew_type=util.RewardType.ABS_FORC, 
			init_field_gen=lambda: shape_field.get_random_field((64, 64)).reshape(1, 64, 64, 1), 
			goal_field_gen=lambda: shape_field.get_random_field((64, 64)).reshape(1, 64, 64, 1))