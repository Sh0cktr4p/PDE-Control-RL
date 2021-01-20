import os
from stable_baselines3.common.callbacks import BaseCallback

class TrainingMonitorCallback(BaseCallback):
    def __init__(self, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir)
