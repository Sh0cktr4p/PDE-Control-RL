from gym.envs.registration import register

register(
    id='burgers-v0',
    entry_point='gym_phiflow.envs:BurgersEnv',
)