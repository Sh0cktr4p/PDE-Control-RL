from typing import Tuple

DOMAIN_SHAPE: Tuple[int, ...] = [32]     # Size and shape of the fields
VISCOSITY: float = 0.003
STEP_COUNT: int = 32                         # Trajectory length
DT: float = 0.03
DIFFUSION_SUBSTEPS: int = 1

DATA_FOLDER: str = 'forced-burgers-clash'
SCENE_COUNT: int = 1000
SIM_BATCH_SIZE: int = 100

TRAIN_RANGE: range = range(200, 1000)
VAL_RANGE: range = range(100, 200)
TEST_RANGE: range = range(0, 100)

EXP_NAME: str = 'debug'

N_ENVS: int = 10
FINAL_REWARD_FACTOR: float = STEP_COUNT
STEPS_PER_ROLLOUT: int = STEP_COUNT * N_ENVS
N_ROLLOUTS: int = 1000
TRAINING_TIMESTEPS: int = STEPS_PER_ROLLOUT * N_ROLLOUTS
N_EPOCHS: int = 10
LR: float = 1e-4
PPO_BATCH_SIZE: int = 128
