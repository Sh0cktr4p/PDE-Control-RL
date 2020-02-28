import spinup
from spinup import ppo
import tensorflow as tf
import gym

exp_map = {
	'00': 'two',
	'01': 'three',
	'02': 'cont_complete',
	'03': 'two_rel',
	'04': 'three_random',
	'05': 'three_reachable',
	'06': 'cont_complete_random',
	'07': 'three_three_reachable',
	'08': 'cont_eight_reachable',
	'09': 'three_three_reachable_time',
	'10': 'cont_eight_reachable_time',
}

key = '00'
value = exp_map[key]

name = 'gym_phiflow:burger-v' + key
path = 'output/burger_' + value

env_fn = lambda: gym.make(name)

ac_kwargs = dict(hidden_sizes=[20,15], activation=tf.nn.leaky_relu)

logger_kwargs = dict(output_dir=path, exp_name='burger')

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=3200, epochs=500, logger_kwargs=logger_kwargs, pi_lr=2.8e-4, gamma=0.96)

