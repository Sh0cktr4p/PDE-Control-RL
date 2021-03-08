import sys; sys.path.append('../src');

from experiment import BurgersEvaluation
from config import *

if __name__ == '__main__':
    trainer = BurgersEvaluation(
        exp_name='../networks/rl-models/fixedSetTraining',
        data_path='../notebooks/forced-burgers-clash',
        data_range=range(0, 100),
        test_mode=False,
    )

    trainer.train(n_rollouts=500, save_freq=50)