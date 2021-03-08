import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.ppo import PPO
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import CallbackList

from envs.vec_monitor import VecMonitor
from envs.burgers_env import BurgersEnv
from envs.burgers_fixed_set import BurgersFixedSetEnv

from function_callback import EveryNRolloutsFunctionCallback, EveryNRolloutsPlusStartFinishFunctionCallback
from policy import CustomActorCriticPolicy
from networks import RES_UNET, CNN_FUNNEL


class ExperimentFolder:
    agent_filename = 'agent'
    monitor_filename = 'monitor.csv'
    kwargs_filename = 'kwargs'
    tensorboard_filename = 'tensorboard-log'

    def __init__(self, name):
        self.store_path = name
        self.agent_path = os.path.join(self.store_path, self.agent_filename)
        self.monitor_path = os.path.join(self.store_path, self.monitor_filename)
        self.kwargs_path = os.path.join(self.store_path, self.kwargs_filename)
        self.tensorboard_path = os.path.join(self.store_path, self.tensorboard_filename)

        if not self.can_be_loaded:
            os.makedirs(self.store_path)

    @property
    def can_be_loaded(self):
        return os.path.exists(self.agent_path + '.zip')

    @staticmethod
    def exists(name):
        path = name
        return os.path.exists(path)

    def store_agent_only(self, agent):
        print('Storing agent to disk...')
        agent.save(self.agent_path)

    def store(self, agent, env_kwargs, agent_kwargs):
        print('Storing agent and hyperparameters to disk...')
        kwargs = self._group_kwargs(env_kwargs, agent_kwargs)
        agent.save(self.agent_path)
        with open(self.kwargs_path, 'wb') as kwargs_file:
            pickle.dump(kwargs, kwargs_file)

    def get(self, env_cls, env_kwargs, agent_kwargs):
        if self.can_be_loaded:
            print('Loading existing agent from %s' % (self.agent_path + '.zip'))
            return self._load(env_cls, env_kwargs, agent_kwargs)
        else:
            print('Creating new agent...')
            return self._create(env_cls, env_kwargs, agent_kwargs)

    def get_monitor_table(self):
        return pd.read_csv(self.monitor_path, skiprows=[0])

    def _create(self, env_cls, env_kwargs, agent_kwargs):
        env = self._build_env(env_cls, env_kwargs, agent_kwargs['n_steps'])
        agent = PPO(env=env, tensorboard_log=self.tensorboard_path, **agent_kwargs)
        return agent, env

    def _load(self, env_cls, env_kwargs, agent_kwargs):
        with open(self.kwargs_path, 'rb') as kwargs_file:
            kwargs = pickle.load(kwargs_file)
        kwargs['env'].update(env_kwargs)
        kwargs['agent'].update(agent_kwargs)
        env = self._build_env(env_cls, kwargs['env'], kwargs['agent']['n_steps'])
        agent = PPO.load(path=self.agent_path, env=env, tensorboard_log=self.tensorboard_path, **kwargs['agent'])
        return agent, env

    def _build_env(self, env_cls, env_kwargs, rollout_size):
        env = env_cls(**env_kwargs)
        return VecMonitor(env, rollout_size, self.monitor_path, info_keywords=('rew_unnormalized', 'forces'))

    def _group_kwargs(self, env_kwargs, agent_kwargs):
        return dict(
            env=env_kwargs,
            agent=agent_kwargs,
        )


class Experiment:
    def __init__(self, name, env_cls, env_kwargs, agent_kwargs, steps_per_rollout, num_envs, callbacks=[]):
        self.folder = ExperimentFolder(name)
        self.agent, self.env = self.folder.get(env_cls, env_kwargs, agent_kwargs)
        self.steps_per_rollout = steps_per_rollout
        self.num_envs = num_envs

        store = lambda _: self.folder.store(self.agent, env_kwargs, agent_kwargs)
        self.get_callback = lambda save_freq: CallbackList(callbacks + [EveryNRolloutsPlusStartFinishFunctionCallback(save_freq, store)])

    def train(self, n_rollouts, save_freq):
        self.agent.learn(total_timesteps=n_rollouts * self.steps_per_rollout * self.num_envs, callback=self.get_callback(save_freq))

    def plot(self):
        monitor_table = self.folder.get_monitor_table()
        avg_rew = monitor_table['rew_unnormalized']
        return plt.plot(avg_rew)

    def reset(self):
        return self.env.reset()

    def predict(self, obs, deterministic=True):
        act, _ = self.agent.predict(obs, deterministic=deterministic)
        return act

    def step_env(self, act):
        return self.env.step(act)


class BurgersTraining(Experiment):
    def __init__(
        self, 
        exp_name,
        domain,
        viscosity,
        step_count, 
        dt,
        diffusion_substeps,
        n_envs,
        final_reward_factor,
        steps_per_rollout,
        n_epochs,
        learning_rate,
        batch_size,
        test_path=None,
        test_range=range(100),
    ):   
        callbacks = []

        env_kwargs = dict(
            num_envs=n_envs,
            step_count=step_count,
            domain=domain,
            dt=dt,
            viscosity=viscosity,
            diffusion_substeps=diffusion_substeps,
            final_reward_factor=final_reward_factor,
            exp_name=exp_name,
        )

        test_env_kwargs = {k:env_kwargs[k] for k in env_kwargs if k != 'num_envs'}

        if test_path is not None:
            self.test_env = BurgersFixedSetEnv(
                data_path=test_path,
                data_range=test_range,
                test_mode=True,
                num_envs=len(test_range),
                **test_env_kwargs
            )
            callbacks.append(EveryNRolloutsFunctionCallback(1, lambda _: self._record_test_set_forces()))

        # Only add a fresh running mean to new experiments
        if not ExperimentFolder.exists(exp_name):
            env_kwargs['reward_rms'] = RunningMeanStd()

        agent_kwargs= dict(
            verbose=1,
            policy=CustomActorCriticPolicy,
            policy_kwargs=dict(
                pi_net=RES_UNET,
                vf_net=CNN_FUNNEL,
                vf_latent_dim=16,
                pi_kwargs=dict(
                    sizes=[4, 8, 16, 16, 16]
                ),
                vf_kwargs=dict(
                    sizes=[4, 8, 16, 16, 16]
                ),
            ),
            n_steps=steps_per_rollout,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        #env_kwargs['data_path'] = test_path
        #env_kwargs['data_range'] = test_range
        #env_kwargs['test_mode'] = False
        #super().__init__(exp_name, BurgersFixedSetEnv, env_kwargs, agent_kwargs, steps_per_rollout, n_envs, callbacks)
        super().__init__(exp_name, BurgersEnv, env_kwargs, agent_kwargs, steps_per_rollout, n_envs, callbacks)

    def infer_test_set_forces(self):
        self.agent.set_env(self.test_env)

        obs = self.test_env.reset()
        done = False
        forces = np.zeros((self.test_env.num_envs,), dtype=np.float32)

        i = 0

        while not done:
            i += 1
            act = self.predict(obs, False)
            obs, _, dones, infos = self.test_env.step(act)
            done = dones[0]
            forces += [infos[i]['forces'] for i in range(len(infos))]

        self.agent.set_env(self.env)

        return forces

    def infer_test_set_frames(self):
        self.agent.set_env(self.test_env)

        obs = np.array(self.test_env.reset())
        init = obs[:, :, 0]
        goal = obs[:, :, 1]
        gt_frames = self.test_env.frames
        cont_frames = [init]
        pass_frames = [init]

        pass_state = self.test_env._get_init_state()

        done = False
        infos = []

        while not done:
            act = self.predict(obs)
            obs, _, dones, infos = self.test_env.step(act)
            pass_state = self.test_env._step_sim(pass_state, ())
            pass_frames.append(pass_state.velocity.data)
            done = dones[0]
            if not done:
                cont_frames.append(np.array(obs)[:,:,0])

        cont_frames.append(goal)

        self.agent.set_env(self.env)

        return cont_frames, gt_frames, pass_frames

    def _record_test_set_forces(self):
        forces = self.infer_test_set_forces()
        force_avg = np.sum(forces) / self.test_env.num_envs

        logger.record('test set forces', force_avg)
        print('Forces on test set: %f' % force_avg)


class BurgersEvaluation(Experiment):
    def __init__(self, exp_name, data_path, data_range, test_mode=True):
        env_kwargs = dict(
            num_envs=10,
            data_path=data_path, 
            data_range=data_range,
            test_mode=test_mode,
        )
        assert ExperimentFolder.exists(exp_name)

        agent_kwargs = dict(
        )

        super().__init__(exp_name, BurgersFixedSetEnv, env_kwargs, agent_kwargs, 32, len(data_range))

        self.test_mode = test_mode

    def train(self, n_rollouts, save_freq):
        if self.test_mode:
            # Don't store anything if the agent is only to be evaluated.
            callback = None 
        else:
            # Don't store kwargs to avoid errors when continuing training afterwards with other env
            store_fn = lambda _: self.folder.store_agent_only(self.agent)
            callback = EveryNRolloutsFunctionCallback(save_freq, store_fn)

        self.agent.learn(total_timesteps=n_rollouts * self.steps_per_rollout, callback=callback)

    def infer_all_frames(self):
        obs = np.array(self.reset())
        init = obs[:, :, 0]
        goal = obs[:, :, 1]
        gt_frames = self.env.frames
        cont_frames = [init]
        pass_frames = [init]

        pass_state = self.env._get_init_state()

        done = False
        infos = []

        while not done:
            act = self.predict(obs)
            obs, _, dones, infos = self.step_env(act)
            pass_state = self.env._step_sim(pass_state, ())
            pass_frames.append(pass_state.velocity.data)
            done = dones[0]
            if not done:
                cont_frames.append(np.array(obs)[:,:,0])
            
        forces = [infos[i]['episode']['forces'] for i in range(len(infos))]

        cont_frames.append(goal)

        return cont_frames, gt_frames, pass_frames, forces
