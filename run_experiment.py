import spinup
from spinup import ppo_tf1, ppo_pytorch
import tensorflow as tf
import torch
import gym
from exp_map import exp_map
import time
import datetime
import actor_critic

def run_experiment(sim_name='burger', key='00', epochs=500, save_freq=50, label=''):
	if label != '' and label[0] != '_':
		label = '_' + label
	
	name = 'gym_phiflow:%s-v%s' % (sim_name, key)
	path = 'output/%s_%s%s' % (sim_name, exp_map[key], label)

	env_fn = lambda: gym.make(name)

	ac_kwargs = dict(hidden_sizes=[20,15], activation=torch.nn.LeakyReLU)

	logger_kwargs = dict(output_dir=path, exp_name=sim_name)

	tic = time.time()
	ppo_pytorch(env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=3200, epochs=epochs, logger_kwargs=logger_kwargs, 
			pi_lr=2.8e-4, gamma=0.96, save_freq=save_freq, actor_critic=actor_critic.MLPActorCritic)
	toc = time.time()

	time_msg = 'Training time: %s' % datetime.timedelta(seconds=toc - tic)

	print(time_msg)

	with open('%s/training_stats.txt' % path, 'a') as file:
		file.write(time_msg)


run_experiment('navier', '16', 1, 100, label='cnn')
#[run_experiment('burger', '10%i' % i) for i in range(1, 5)]