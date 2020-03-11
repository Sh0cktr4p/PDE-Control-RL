from gym_phiflow.envs.navier_env import NavierEnv
from gym_phiflow.envs import util
import numpy as np

twenty400 = util.act_points(size=(20,20), indices=tuple(zip(*[(i, 0) for i in range(20)])))

class NavierEnvTwo(NavierEnv):
	def __init__(self):
		super().__init__(name='v00')

class NavierEnvContTwenty2DReachable(NavierEnv):
	def __init__(self):
		super().__init__(name='v12', act_type=util.ActionType.CONTINUOUS,
			act_points=twenty400, goal_type=util.GoalType.REACHABLE,
			rew_type=util.RewardType.ABS_FORC)