import spinup
from spinup import ppo_tf1
import tensorflow as tf
import gym
from exp_map import exp_map
import time
import datetime

def run_experiment(sim_name='burger', key='00', epochs=500, save_freq=50):
	name = 'gym_phiflow:%s-v%s' % (sim_name, key)
	path = 'output/%s_%s' % (sim_name, exp_map[key])

	env_fn = lambda: gym.make(name)

	ac_kwargs = dict(hidden_sizes=[20,15], activation=tf.nn.leaky_relu)

	logger_kwargs = dict(output_dir=path, exp_name=sim_name)

	tic = time.time()
	ppo_tf1(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=3200, epochs=epochs, logger_kwargs=logger_kwargs, pi_lr=2.8e-4, gamma=0.96, save_freq=save_freq)
	toc = time.time()

	print('Training time: %s' % datetime.timedelta(seconds=toc - tic))

run_experiment('burger', '104', 1000, 100)
#[run_experiment('burger', '10%i' % i) for i in range(1, 5)]