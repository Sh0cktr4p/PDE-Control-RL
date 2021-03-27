import sys; sys.path.append('../src')

from experiment import BurgersTraining
from config import *

if __name__ == '__main__':
    trainer = BurgersTraining(
        exp_name='../networks/rl-models/del3', 
        domain=domain,
        viscosity=viscosity,
        step_count=step_count,
        dt=dt,
        diffusion_substeps=diffusion_substeps,
        n_envs=n_envs,
        final_reward_factor=final_reward_factor,
        steps_per_rollout=steps_per_rollout,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        test_path=test_path,
        test_range=test_range,
    )

    trainer.train(n_rollouts=1000, save_freq=50)
