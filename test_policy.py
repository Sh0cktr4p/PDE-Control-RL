from spinup.utils.test_policy import load_policy_and_env, run_policy
import gym
from exp_map import exp_map

def test_policy(key):
	name = 'gym_phiflow:burger-v' + key
	path = 'output/burger_' + exp_map[key]
	_, get_action = load_policy_and_env(path)
	env = gym.make(name)
	run_policy(env, get_action, num_episodes=10)

test_policy('11')