from gym.envs.registration import register

register(
    id='burger-v0',
    entry_point='gym_phiflow.envs:BurgerEnvTwoActions',
)

register(
    id='burger-v1',
    entry_point='gym_phiflow.envs:BurgerEnvThreeActions',
)

register(
    id='burger-v2',
    entry_point='gym_phiflow.envs:BurgerEnvRelativeReward',
)

register(
    id='burger-v3',
    entry_point='gym_phiflow.envs:BurgerEnvCompleteControl',
)

register(
    id='burger-v4',
    entry_point='gym_phiflow.envs:BurgerEnvThreeXThreeActionsReachableGoal'
)

register(
    id='burger-v5',
    entry_point='gym_phiflow.envs:BurgerEnvThreeActionsRandomGoal'
)

register(
    id='burger-v6',
    entry_point='gym_phiflow.envs:BurgerEnvCompleteControlRandomGoal'
)

register(
    id='burger-v7',
    entry_point='gym_phiflow.envs:BurgerEnvThreeActionsReachableGoal'
)

register(
    id='burger-v8',
    entry_point='gym_phiflow.envs:BurgerEnvEightXCompleteControlRandomGoal'
)