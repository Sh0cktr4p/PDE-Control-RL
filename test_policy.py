from spinup.utils.test_policy import load_policy_and_env, run_policy
import gym
from exp_map import exp_map

def test_policy(sim_name='burger', key='00', itr='last', label=''):
	name = 'gym_phiflow:%s-v%s' % (sim_name, key)
	path = 'output/%s/%s/%s' % (sim_name, exp_map[key], label)
	_, get_action = load_policy_and_env(path, itr)
	env = gym.make(name)
	run_policy(env, get_action, num_episodes=100)

#test_policy('burger', '108', 600, 'fixed_unet')
#test_policy('burger', '109', 750, 'fixed_old_unet')
#test_policy('burger', '110', 800, 'old_unet')
#test_policy('burger', '111', 800, 'pow')
test_policy('burger', '20', 751, 'benchmark_02')

#test_policy('navier', '306', 99, 'unet')
#test_policy('navier', '309', 200, 'fixed_old_unet')
#test_policy('navier', '309', 200, 'no_fc')
#test_policy('navier', '311', 300, 'fixed_old_unet')
#test_policy('navier', '311', 400, 'low_fc')
#test_policy('navier', '311', 500, 'yes_fc')
#test_policy('navier', '312', 300, 'no_fc')
#test_policy('navier', '312', 250, 'yes_fc')
