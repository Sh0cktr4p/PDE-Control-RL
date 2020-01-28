from gym.envs.registration import register

register(
    id='burger-v0',
    entry_point='gym_phiflow.envs:BurgerEnv',
)