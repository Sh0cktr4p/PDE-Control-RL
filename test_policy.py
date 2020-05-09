from spinup.utils.test_policy import load_policy_and_env, run_policy
import gym
from exp_map import exp_map

def test_policy(sim_name='burger', key='00', itr='last', label=''):
	if label != '' and label[0] != '_':
		label = '_' + label
	name = 'gym_phiflow:%s-v%s' % (sim_name, key)
	path = 'output/%s_%s' % (sim_name, exp_map[key] + label)
	_, get_action = load_policy_and_env(path, itr)
	env = gym.make(name)
	run_policy(env, get_action, num_episodes=100)

#test_policy('burger','104', label='efn')
test_policy('burger','109', label='loong')