import spinup
from spinup import ppo_tf1
import tensorflow as tf
import gym
from exp_map import exp_map
#from spinup.algos.tf1 import ppo

key = '11'
value = exp_map[key]

name = 'gym_phiflow:burger-v' + key
path = 'output/burger_' + value

env_fn = lambda: gym.make(name)

ac_kwargs = dict(hidden_sizes=[20,15], activation=tf.nn.leaky_relu)

logger_kwargs = dict(output_dir=path, exp_name='burger')

ppo_tf1(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=3200, epochs=50, logger_kwargs=logger_kwargs, pi_lr=2.8e-4, gamma=0.96)

