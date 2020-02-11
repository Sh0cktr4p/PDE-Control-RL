import spinup
from spinup import ppo
import tensorflow as tf
import gym

exp_map = {
	'0': 'basic',
	'1': 'three_actions',
	'2': 'relative_reward',
	'3': 'complete_control',
	'4': 'three_x_three_actions_reachable_goal',
	'5': 'three_actions_random_goal',
	'6': 'complete_control_random_goal',
	'7': 'three_actions_reachable_goal',
}

key = '4'
value = exp_map[key]

name = 'gym_phiflow:burger-v' + key
path = 'output/burger_' + value

env_fn = lambda: gym.make(name)

ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)

logger_kwargs = dict(output_dir=path, exp_name='burger')

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=3200, epochs=50, logger_kwargs=logger_kwargs)

