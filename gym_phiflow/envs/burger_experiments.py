from gym_phiflow.envs.burger_env import BurgerEnv
from gym_phiflow.envs import util
import numpy as np

complete16 = np.ones(shape=(16,), dtype=np.bool)

three16 = np.array([True, True, True, False, False, False, False, False,
					False, False, False, False, False, False, False, False])

eight16 = np.array([False, False, False, False, True, True, True, True,
					True, True, True, True, False, False, False, False])



class BurgerEnvTwo(BurgerEnv):
	def __init__(self):
		super().__init__(name='v00')

class BurgerEnvThree(BurgerEnv):
	def __init__(self):
		super().__init__(name='v01', act_type=util.ActionType.DISCRETE_3)

class BurgerEnvContComplete(BurgerEnv):
	def __init__(self):
		super().__init__(name='v02', act_type=util.ActionType.CONTINUOUS, 
			act_points=complete16, rew_type=util.RewardType.ABS_FORC)

class BurgerEnvTwoRel(BurgerEnv):
	def __init__(self):
		super().__init__(name='v03', rew_type=util.RewardType.RELATIVE)

class BurgerEnvThreeRandom(BurgerEnv):
	def __init__(self):
		super().__init__(name='v04', act_type=util.ActionType.DISCRETE_3,
			goal_type=util.GoalType.RANDOM)

class BurgerEnvThreeReachable(BurgerEnv):
	def __init__(self):
		super().__init__(name='v05', act_type=util.ActionType.DISCRETE_3,
			goal_type=util.GoalType.REACHABLE)

class BurgerEnvContCompleteRandom(BurgerEnv):
	def __init__(self):
		super().__init__(name='v06', act_type=util.ActionType.CONTINUOUS,
			act_points=complete16, goal_type=util.GoalType.RANDOM,
			rew_type=util.RewardType.ABS_FORC)

class BurgerEnvThreeThreeReachable(BurgerEnv):
	def __init__(self):
		super().__init__(name='v07', act_type=util.ActionType.DISCRETE_3,
			act_points=three16, goal_type=util.GoalType.REACHABLE)

class BurgerEnvContEightReachable(BurgerEnv):
	def __init__(self):
		super().__init__(name='v08', act_type=util.ActionType.CONTINUOUS,
			act_points=eight16, goal_type=util.GoalType.REACHABLE,
			rew_type=util.RewardType.ABS_FORC)

class BurgerEnvThreeThreeReachableTime(BurgerEnv):
	def __init__(self):
		super().__init__(name='v09', use_time=True, act_type=util.ActionType.DISCRETE_3,
		act_points=three16, goal_type=util.GoalType.REACHABLE)

class BurgerEnvContEightReachableTime(BurgerEnv):
	def __init__(self):
		super().__init__(name='v10', use_time=True, act_type=util.ActionType.CONTINUOUS,
			act_points=eight16, goal_type=util.GoalType.REACHABLE,
			rew_type=util.RewardType.ABS_FORC)

