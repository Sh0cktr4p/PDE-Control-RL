from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gym_phiflow.envs.burgers_env import BurgersEnv
from gym_phiflow.envs.burger_env import BurgerEnv
from gym_phiflow.envs.vec_monitor import VecMonitor
from networks import ALT_UNET, CNN_FUNNEL
from sb_ac import CustomActorCriticPolicy
import gym

import torch

field_shape = (32,)

n_steps = 3200
n_envs = 10
n_rollouts = 100

env = BurgersEnv(num_envs=n_envs, field_shape=field_shape, final_reward_factor=32)
env = VecMonitor(env, n_steps, 'loggo')
#env = gym.make('gym_phiflow:burger-v20')
#env = Monitor(env)

#from stable_baselines3.common.env_util import make_atari_env

#env = Monitor(gym.make('gym_phiflow:burger-v106'), filename='lollol')

print(env.action_space.__dict__)

#network_input_shape = env.observation_space.shape
#network_output_dim = env.action_space.shape[0]
#network_sizes = [4, 8, 16, 16, 16]

'''
policy_kwargs = {
    'pi_net': ALT_UNET,
    'vf_net': CNN_FUNNEL,
    'vf_latent_dim': 16,
    'pi_kwargs': {
        'sizes': [4, 8, 16, 16, 16],
    },
    'vf_kwargs': {
        'sizes': [4, 8, 16, 16, 16],
    }
}
'''

policy_kwargs = {'activation_fn':torch.nn.ReLU, 'net_arch':[70, 60, 50]}

model = PPO('MlpPolicy', env, verbose=1)#learning_rate=2e-5, n_steps=n_steps, batch_size=64, n_epochs=10, verbose=1, policy_kwargs=policy_kwargs)
#model = PPO.load('test_run', env)

model.learn(100000)#n_steps * n_envs * n_rollouts)

#model.save('test_run')

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    print(action)
    env.render('l')
    obs, reward, done, info = env.step(action)
    if done[0]:
        env.render('l')
        obs = env.reset()

env.close()
