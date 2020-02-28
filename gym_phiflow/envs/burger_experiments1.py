from gym_phiflow.envs.burger_env import BurgerEnv

class BurgerEnvTwoActions(BurgerEnv):
	@property
	def action_space(self):
		return self.discrete_2_space()

class BurgerEnvThreeActions(BurgerEnv):
	@property
	def action_space(self):
		return self.discrete_3_space()


class BurgerEnvRelativeReward(BurgerEnv):
	def calc_reward(self, mse_old, mse_new, forces):
		return self.relative_reward(mse_old, mse_new, forces)


class BurgerEnvCompleteControl(BurgerEnv):
	@property
	def action_space(self):
		return self.continuous_complete_field_space()

	def calc_reward(self, mse_old, mse_new, forces):
		return self.mse_new_and_forces_reward(mse_old, mse_new, forces)


class BurgerEnvThreeXThreeActionsReachableGoalSimple(BurgerEnv):
	@property
	def size(self):
		return (16,)

	@property
	def action_space(self):
		return self.discrete_3_3_space()

	@property
	def observation_space(self):
		return self.continuous_double_size_space()

	def create_goal(self):
		return self.reachable_goal_obs()


class BurgerEnvThreeActionsRandomGoal(BurgerEnvThreeActions):
	@property
	def observation_space(self):
		return self.continuous_double_size_space()

	def create_goal(self):
		return self.random_goal_obs()


class BurgerEnvCompleteControlRandomGoal(BurgerEnvCompleteControl):
	@property
	def observation_space(self):
		return self.continuous_double_size_space()

	def create_goal(self):
		return self.random_goal_obs()


class BurgerEnvThreeActionsReachableGoal(BurgerEnvThreeActions):
	@property
	def observation_space(self):
		return self.continuous_double_size_space()

	def create_goal(self):
		return self.reachable_goal_obs()


class BurgerEnvEightXCompleteControlReachableGoalSimple(BurgerEnv):
	@property
	def size(self):
		return (16,)

	@property
	def action_space(self):
		return self.continuous_8_space()

	@property
	def observation_space(self):
		return self.continuous_double_size_space()

	def create_goal(self):
		return self.reachable_goal_obs()

	def calc_reward(self, mse_old, mse_new, forces):
		return self.mse_new_and_forces_reward(mse_old, mse_new, forces)

class BurgerEnvCompleteControlRandomGoalSimple(BurgerEnvCompleteControlRandomGoal):
	@property
	def size(self):
		return (16,)

class BurgerEnvEightXCompleteControlReachableGoalTimeVariantSimple(BurgerEnv):
	@property
	def size(self):
		return (16,)

	@property
	def action_space(self):
		return self.continuous_8_space()

	@property
	def observation_space(self):
		return self.continuous_double_size_with_time_space()

	def create_goal(self):
		return self.reachable_goal_obs()

	def calc_reward(self, mse_old, mse_new, forces):
		return self.weighted_mse_forces_reward(mse_old, mse_new, forces)

class BurgerEnvThreeXThreeActionsRelativeRewardReachableGoalTimeVariantSimple(BurgerEnv):
	@property
	def size(self):
		return (16,)

	@property
	def action_space(self):
		return self.discrete_3_3_space()

	@property
	def observation_space(self):
		return self.continuous_double_size_with_time_space()

	def create_goal(self):
		return self.reachable_goal_obs()

	def calc_reward(self, mse_old, mse_new, forces):
		return self.relative_reward(mse_old, mse_new, forces)