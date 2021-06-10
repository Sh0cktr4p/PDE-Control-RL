import argparse
from src.experiment import BurgersTraining
from scripts.config import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='delete')

    args = parser.parse_args()

    exp_path = '../networks/rl-models/' + args.__dict__['name']

    trainer = BurgersTraining(
        path=exp_path, 
        domain=domain,
        viscosity=viscosity,
        step_cnt=step_count,
        dt=dt,
        diffusion_substeps=diffusion_substeps,
        n_envs=n_envs,
        final_reward_factor=final_reward_factor,
        steps_per_rollout=steps_per_rollout,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        data_path=data_path,
        val_range=val_range,
        test_range=test_range,
    )

    trainer.train(n_rollouts=1000, save_freq=50)