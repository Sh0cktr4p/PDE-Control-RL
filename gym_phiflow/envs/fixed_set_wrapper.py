from stable_baselines3.common.vec_env import VecEnvWrapper

class FixedSetWrapper(VecEnvWrapper):
    def __init__(self, venv, data_set):
        super().__init__(self, venv)

        self.data_set = data_set

    def reset():
        