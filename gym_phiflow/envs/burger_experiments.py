from gym_phiflow.envs.burger_env import BurgerEnv
from gym_phiflow.envs import util
import numpy as np

complete16 = np.ones(shape=(16,), dtype=np.bool)
complete32 = np.ones(shape=(32,), dtype=np.bool)

two16 = util.act_points(size=(16,), indices=[7,8])
three16 = util.act_points(size=(16,), indices=[7,8,9])
four16 = util.act_points(size=(16,), indices=[6,7,8,9])

eight16 = np.array([False, False, False, False, True, True, True, True,
					True, True, True, True, False, False, False, False])

sixteen64 = np.array([	[False, False, False, False, False, False, False, False],
						[False, False, False, False, False, False, False, False],
						[False, False, True, True, True, True, False, False],
						[False, False, True, True, True, True, False, False],
						[False, False, True, True, True, True, False, False],
						[False, False, True, True, True, True, False, False],
						[False, False, False, False, False, False, False, False],
						[False, False, False, False, False, False, False, False]])

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

class BurgerEnvContSixteen2DReachable(BurgerEnv):
	def __init__(self):
		super().__init__(name='v11', act_type=util.ActionType.CONTINUOUS,
			act_points=sixteen64, goal_type=util.GoalType.REACHABLE,
			rew_type=util.RewardType.ABS_FORC)

class BurgerEnvTwoReachableSync(BurgerEnv):
	def __init__(self):
		super().__init__(name='v15', act_points=complete16, 
			goal_type=util.GoalType.REACHABLE, synchronized=True)

class BurgerEnvThreeTwoReachable(BurgerEnv):
	def __init__(self):
		super().__init__(name='v100', act_type=util.ActionType.DISCRETE_3,
			act_points=two16, goal_type=util.GoalType.REACHABLE)

class BurgerEnvContTwoReachable(BurgerEnv):
	def __init__(self):
		super().__init__(name='v101', act_type=util.ActionType.CONTINUOUS,
			act_points=two16, goal_type=util.GoalType.REACHABLE,
			rew_type=util.RewardType.ABS_FORC)

class BurgerEnvThreeFourReachable(BurgerEnv):
	def __init__(self):
		super().__init__(name='v102', act_type=util.ActionType.DISCRETE_3,
			act_points=four16, goal_type=util.GoalType.REACHABLE)

class BurgerEnvContFourReachable(BurgerEnv):
	def __init__(self):
		super().__init__(name='v103', act_type=util.ActionType.CONTINUOUS,
			act_points=four16, goal_type=util.GoalType.REACHABLE,
			rew_type=util.RewardType.ABS_FORC)

class BurgerEnvContCompleteConstant(BurgerEnv):
	def __init__(self):
		super().__init__(name='v104', act_type=util.ActionType.CONTINUOUS,
			act_points=complete32, goal_type=util.GoalType.CONSTANT_FORCE,
			rew_type=util.RewardType.ABS_FORC)

class BurgerEnvContCompleteConstantSmoothed(BurgerEnv):
	def __init__(self):
		super().__init__(name='v105', act_type=util.ActionType.CONTINUOUS,
			act_points=complete32, goal_type=util.GoalType.CONSTANT_FORCE,
			rew_type=util.RewardType.ABS_FORC, rew_force_factor=2)

class BurgerEnvContCompleteConstantPow(BurgerEnv):
	def __init__(self):
		super().__init__(name='v106', act_type=util.ActionType.CONTINUOUS,
			act_points=complete32, goal_type=util.GoalType.CONSTANT_FORCE,
			rew_type=util.RewardType.ABSOLUTE)