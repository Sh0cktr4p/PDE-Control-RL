import spinup
from spinup import ppo
import tensorflow as tf
import gym

exp_map = {
	'0': 'basic',
	'1': 'three_actions',
	'2': 'relative_reward',
	'3': 'complete_control',
	'4': 'three_x_three_actions_reachable_goal_simple',
	'5': 'three_actions_random_goal',
	'6': 'complete_control_random_goal',
	'7': 'three_actions_reachable_goal',
	'8': 'eight_x_complete_control_reachable_goal_simple',
	'9': 'complete_control_random_goal_simple',
	'10': 'eight_continuous_reachable_time_simple',
	'11': 'three_three_relative_reachable_time_simple'
}

key = '11'
value = exp_map[key]

name = 'gym_phiflow:burger-v' + key
path = 'output/burger_' + value

env_fn = lambda: gym.make(name)

ac_kwargs = dict(hidden_sizes=[20,15], activation=tf.nn.leaky_relu)

logger_kwargs = dict(output_dir=path, exp_name='burger')

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=3200, epochs=500, logger_kwargs=logger_kwargs, pi_lr=2.8e-4, gamma=0.96)

