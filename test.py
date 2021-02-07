import os
import time
import numpy as np
from train import agent_backup_filename_template, burgers_fixed_set_env_name, get_env_hparams, get_paths, load, make_env


def sleep_to_keep_framerate(start_time, end_time, time_between_frames):
    time_to_sleep = time_between_frames - (end_time - start_time)
    if time_to_sleep > 0:
        time.sleep(time_to_sleep)


def test(env_name, path, use_backup, env_hparams, n_trajectories, fps, render_mode):
    experiment_path, monitor_path, agent_path, _, ppo_hparams_path = get_paths('burgers', path)
    monitor_path = monitor_path + '_test'
    time_between_frames = 1 / fps

    env = make_env(env_name, env_hparams, 32, monitor_path)
    
    if use_backup >= 0:
        agent_path = os.path.join(experiment_path, agent_backup_filename_template % use_backup)
        
    assert os.path.exists(ppo_hparams_path)
    assert os.path.exists(agent_path)

    agent = load(env, agent_path, ppo_hparams_path)
    obs = env.reset()

    trajectory_idx = 0

    while trajectory_idx < n_trajectories:
        tic = time.time()
        action, _states = agent.predict(obs, deterministic=True)
        env.render(render_mode)
        obs, reward, done, info = env.step(action)
        if done[0]:
            trajectory_idx += 1
        toc = time.time()
        if render_mode == 'live':
            sleep_to_keep_framerate(tic, toc, time_between_frames)

    env.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_name', type=str, help='environment name', default=burgers_fixed_set_env_name, choices=[burgers_fixed_set_env_name])
    parser.add_argument('--path', type=str, help='path to the model folder')
    parser.add_argument('--n_trajectories', type=int, default=10)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--render_mode', type=str, default='live', choices=['live', 'gif', 'png'])
    parser.add_argument('--use_backup', type=int, default=-1)

    parser.add_argument('--data_path', type=str, help='path to the data set to load')
    parser.add_argument('--data_range', type=range, default=range(100))

    args = parser.parse_args()
    args_dict = args.__dict__

    # Only support one environment at a time during testing
    args_dict['n_envs'] = 1

    env_name = args_dict['env_name']
    path = args_dict['path']
    use_backup = args_dict['use_backup']
    n_trajectories = args_dict['n_trajectories']
    fps = args_dict['fps']
    render_mode = args_dict['render_mode']
    env_args = get_env_hparams(args_dict)
    env_args['data_path'] = args_dict['data_path']
    env_args['data_range'] = args_dict['data_range']

    test(env_name, path, use_backup, env_args, n_trajectories, fps, render_mode)
