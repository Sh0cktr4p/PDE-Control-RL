import os
import json
import shutil
import torch
import pickle

import gym
from stable_baselines3 import PPO
from networks import RES_UNET, CNN_FUNNEL
from policy import CustomActorCriticPolicy
from function_callback import EveryNRolloutsFunctionCallback
from gym_phiflow.envs.burgers_env import BurgersEnv
from gym_phiflow.envs.vec_monitor import VecMonitor
from gym_phiflow.envs.burgers_fixed_set import BurgersFixedSetEnv


base_path = 'models'
monitor_filename = 'monitor'
agent_filename = 'agent.zip'
agent_temp_filename = 'agent_temp.zip'
agent_backup_filename_template = 'agent_backup_%02i.zip'
ppo_hparams_filename = 'hparams.pkl'

burgers_env_name = 'burgers'
burgers_fixed_set_env_name = 'burgers_fixed_env'


def get_mlp_kwargs():
    policy_kwargs = {
        'activation_fn': torch.nn.ReLU, 
        'net_arch': [70, 60, 50],
    }

    return policy_kwargs


def get_unet_kwargs():
    policy_kwargs = {
        'pi_net': RES_UNET,
        'vf_net': CNN_FUNNEL,
        'vf_latent_dim': 16,
        'pi_kwargs': {
            'sizes': [4, 8, 16, 16, 16],
        },
        'vf_kwargs': {
            'sizes': [4, 8, 16, 16, 16],
        },
    }
    return policy_kwargs


def get_paths(env_name, path):
    experiment_path = os.path.join(base_path, env_name, path)
    monitor_path = os.path.join(experiment_path, monitor_filename)
    agent_path = os.path.join(experiment_path, agent_filename)
    agent_temp_path = os.path.join(experiment_path, agent_temp_filename)
    ppo_hparams_path = os.path.join(experiment_path, ppo_hparams_filename)

    return experiment_path, monitor_path, agent_path, agent_temp_path, ppo_hparams_path


def filter_dict(d, ks):
    return {k:d[k] for k in ks}


def get_env_cls(env_name):
    if env_name == burgers_env_name:
        return BurgersEnv
    elif env_name == burgers_fixed_set_env_name:
        return BurgersFixedSetEnv
    else:
        raise NotImplementedError()


def get_next_agent_backup_index(experiment_path):
    i = 0
    while os.path.exists(os.path.join(experiment_path, agent_backup_filename_template % i)):
        i += 1

    return i


def make_env(env_name, env_hparams, n_steps, monitor_path):
    env_cls = get_env_cls(env_name)
    env = env_cls(**env_hparams)
    env = VecMonitor(env, n_steps, monitor_path)
    return env


def create(env, ppo_hparams, ppo_hparams_path):
    print('Storing hparams:')
    print(ppo_hparams)
    with open(ppo_hparams_path, 'wb') as ppo_hparams_file:
        pickle.dump(ppo_hparams, ppo_hparams_file)
    agent = PPO(env=env, verbose=1, **ppo_hparams)
    return agent


def store(agent, agent_path):
    agent.save(agent_path)


def load(env, agent_path, ppo_hparams_path):
    print("AGENT PATH: %s" % agent_path)
    with open(ppo_hparams_path, 'rb') as ppo_hparams_file:
        ppo_hparams = pickle.load(ppo_hparams_file)
    print('Hyperparameters: ')
    print(ppo_hparams)
    agent = PPO.load(agent_path, env, **ppo_hparams)
    return agent


def train(env_name, path, env_hparams, ppo_hparams, learn_hparams, rollouts_between_stores, stores_between_backups):
    assert isinstance(path, str)
    assert isinstance(env_name, str)

    experiment_path, monitor_path, agent_path, agent_temp_path, ppo_hparams_path = get_paths(env_name, path)

    n_steps = ppo_hparams['n_steps']

    next_backup_index = 0

    ppo_hparams['tensorboard_log'] = os.path.join(experiment_path, 'tensorboard/')

    if os.path.exists(experiment_path):
        assert os.path.exists(agent_path)
        assert os.path.exists(ppo_hparams_path)
        # Load an existing model
        env = make_env(env_name, env_hparams, n_steps, monitor_path)
        print('Loading existing model, ignoring new ppo hyperparameters')
        agent = load(env, agent_path, ppo_hparams_path)
        next_backup_index = get_next_agent_backup_index(experiment_path)
    else:
        print('Generating new model at %s' % experiment_path)
        os.makedirs(experiment_path)
        env = make_env(env_name, env_hparams, n_steps, monitor_path)
        agent = create(env, ppo_hparams, ppo_hparams_path)

    def store_callback_fn(repeats):
        nonlocal next_backup_index
        print('Storing model')
        store(agent, agent_temp_path)
        if os.path.exists(agent_path):
            os.remove(agent_path)
        os.rename(agent_temp_path, agent_path)
        if repeats % stores_between_backups == 0:
            print('storing backup')
            backup_path = os.path.join(experiment_path, agent_backup_filename_template % next_backup_index)
            shutil.copyfile(agent_path, backup_path)
            next_backup_index += 1

    store_callback = EveryNRolloutsFunctionCallback(rollouts_between_stores, store_callback_fn)

    agent.learn(callback=store_callback, **learn_hparams)

    env.close()


def get_env_hparams(hparams):
    env_hparam_list = [
        'n_envs',
    ]
    return filter_dict(hparams, env_hparam_list)

def get_ppo_hparams(hparams):
    ppo_hparam_list = [
        'policy',
        'learning_rate',
        'batch_size',
        'n_steps',
        'n_epochs',
        'policy_kwargs',
    ]
    return filter_dict(hparams, ppo_hparam_list)

def get_learn_hparams(hparams):
    learn_hparam_list = [
        'total_timesteps',
    ]
    return filter_dict(hparams, learn_hparam_list)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_name', type=str, help='environment name', default=burgers_env_name, choices=[burgers_env_name])
    parser.add_argument('--path', type=str, help='path to the model folder')
    parser.add_argument('--rollouts_between_stores', type=int, default=10)
    parser.add_argument('--stores_between_backups', type=int, default=10)
    # Environment arguments
    parser.add_argument('--n_envs', type=int, help='number of parallel environments', default=10)
    # PPO arguments
    #parser.add_argument('--policy', type=str, default='MlpPolicy')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_steps', type=int, help='number of steps per rollout and environment', default=320)
    parser.add_argument('--n_epochs', type=int, default=10)
    # Learning arguments
    parser.add_argument('--total_timesteps', type=int, default=320 * 10 * 1000)

    args = parser.parse_args()
    args_dict = args.__dict__
    
    args_dict['policy'] = CustomActorCriticPolicy

    args_dict['policy_kwargs'] = get_unet_kwargs()

    env_name = args_dict['env_name']
    path = args_dict['path']
    rollouts_between_stores = args_dict['rollouts_between_stores']
    stores_between_backups = args_dict['stores_between_backups']
    ppo_args = get_ppo_hparams(args_dict)
    env_args = get_env_hparams(args_dict)
    learn_args = get_learn_hparams(args_dict)

    train(env_name, path, env_args, ppo_args, learn_args, rollouts_between_stores, stores_between_backups)
