import spinup
from spinup import ppo
import tensorflow as tf
import gym

env_fn = lambda : gym.make('gym_phiflow:burger-v0')

ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)

logger_kwargs = dict(output_dir='output/burger_basic/', exp_name='burger')

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=50, logger_kwargs=logger_kwargs)

