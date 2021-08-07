import os
import pickle
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.ppo import PPO
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import CallbackList

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from one_line_output_format import OneLineOutputFormat
from envs.vec_monitor import VecMonitor
from envs.burgers_env import BurgersEnv
from envs.burgers_fixed_set import BurgersFixedSetEnv

from callbacks import CustomLoggerInjectionCallback, EveryNRolloutsFunctionCallback, EveryNRolloutsPlusStartFinishFunctionCallback, RecordInfoScalarsCallback, RecordScalarCallback
from policy import CustomActorCriticPolicy
from networks import RES_UNET, CNN_FUNNEL


class ExperimentFolder:
    agent_filename = 'agent'
    monitor_filename = 'monitor.csv'
    kwargs_filename = 'kwargs'
    tensorboard_filename = 'tensorboard-log'

    def __init__(self, path):
        self.store_path = path
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
    def exists(path):
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
        print('Tensorboard log path: %s' % self.tensorboard_path)
        if self.can_be_loaded:
            print('Loading existing agent from %s' % (self.agent_path + '.zip'))
            return self._load(env_cls, env_kwargs, agent_kwargs)
        else:
            print('Creating new agent...')
            return self._create(env_cls, env_kwargs, agent_kwargs)

    def get_monitor_table(self):
        return pd.read_csv(self.monitor_path, skiprows=[0])

    def get_tensorboard_scalar(self, scalar_name):
        path_template = os.path.join(self.tensorboard_path, 'training_phase_%i')
        # Compatibility with other naming scheme, TODO not good code, needs another revision
        if not os.path.exists(path_template % 0):
            path_template = os.path.join(self.tensorboard_path, 'PPO_%i')
        run_idx = 0
        wall_times, timesteps, scalar_values = [], [], []
        while os.path.exists(path_template % run_idx):
            event_accumulator = EventAccumulator(path_template % run_idx)
            event_accumulator.Reload()

            new_wall_times, new_timesteps, new_scalar_values = zip(*event_accumulator.Scalars(scalar_name))

            # To chain multiple runs together, the time inbetween has to be left out
            prev_run_wall_time = 0 if len(wall_times) == 0 else wall_times[-1]
            # Iterations have to be continuous even when having multiple runs
            prev_run_timesteps = 0 if len(timesteps) == 0 else timesteps[-1]

            wall_times += [prev_run_wall_time + wt - new_wall_times[0] for wt in new_wall_times]
            timesteps += [prev_run_timesteps + it - new_timesteps[0] for it in new_timesteps]
            scalar_values += new_scalar_values

            run_idx += 1

        return wall_times, timesteps, scalar_values

    def get_monitor_scalar(self, scalar_name):
        table = pd.read_csv(self.monitor_path, skiprows=[0])
        wall_times = list(table['t'])
        iterations = [i for i in range(len(wall_times))]
        scalar_values = list(table[scalar_name])

        # Make wall times of multiple runs monotonic:
        base_time = 0
        monotonic_wall_times = [wall_times[0]]
        for i in range(1, len(wall_times)):
            if wall_times[i] < wall_times[i-1]:
                base_time = monotonic_wall_times[i-1]
            monotonic_wall_times.append(wall_times[i] + base_time)
        
        return monotonic_wall_times, iterations, scalar_values



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
    def __init__(self, path, env_cls, env_kwargs, agent_kwargs, steps_per_rollout, num_envs, callbacks=[]):
        self.folder = ExperimentFolder(path)
        self.agent, self.env = self.folder.get(env_cls, env_kwargs, agent_kwargs)
        self.steps_per_rollout = steps_per_rollout
        self.num_envs = num_envs

        store = lambda _: self.folder.store(self.agent, env_kwargs, agent_kwargs)
        self.get_callback = lambda save_freq: CallbackList(callbacks + [EveryNRolloutsPlusStartFinishFunctionCallback(save_freq, store)])

    def train(self, n_rollouts, save_freq):
        self.agent.learn(total_timesteps=n_rollouts * self.steps_per_rollout * self.num_envs, callback=self.get_callback(save_freq), tb_log_name="training_phase")

    def plot(self):
        monitor_table = self.folder.get_monitor_table()
        avg_rew = monitor_table['rew_unnormalized']
        return plt.plot(avg_rew)

    def reset_env(self):
        return self.env.reset()

    def predict(self, obs, deterministic=True):
        act, _ = self.agent.predict(obs, deterministic=deterministic)
        return act

    def step_env(self, act):
        return self.env.step(act)


class BurgersTraining(Experiment):
    def __init__(
        self, 
        path,
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
        data_path=None,
        val_range=range(100, 200),
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
            exp_name=path,
        )

        evaluation_env_kwargs = {k:env_kwargs[k] for k in env_kwargs if k != 'num_envs'}

        if data_path is not None:
            self.val_env = BurgersFixedSetEnv(
                data_path=data_path,
                data_range=val_range,
                num_envs=len(val_range),
                **evaluation_env_kwargs
            )
            self.test_env = BurgersFixedSetEnv(
                data_path=data_path,
                data_range=test_range,
                num_envs=len(test_range),
                **evaluation_env_kwargs
            )

            #callbacks.append(EveryNRolloutsFunctionCallback(1, lambda _: self._record_forces(self.val_env, 'val_set_forces')))
            callbacks.append(RecordScalarCallback('val_set_forces', lambda: self._get_forces(self.val_env)))

        # Only add a fresh running mean to new experiments
        if not ExperimentFolder.exists(path):
            env_kwargs['reward_rms'] = RunningMeanStd()

        agent_kwargs= dict(
            verbose=0,
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

        callbacks.append(CustomLoggerInjectionCallback())
        callbacks.append(RecordInfoScalarsCallback('forces', 'rew_unnormalized'))

        super().__init__(path, BurgersEnv, env_kwargs, agent_kwargs, steps_per_rollout, n_envs, callbacks)

    def infer_test_set_forces(self):
        return self._infer_forces(self.test_env)

    def infer_test_set_frames(self):
        return self._infer_frames(self.test_env)

    def get_val_set_forces_data(self):
        wall_times, timesteps, forces = self.folder.get_tensorboard_scalar('val_set_forces')
        iterations = [i for i in range(len(timesteps))]
        return wall_times, iterations, forces

    def _infer_forces(self, env: BurgersFixedSetEnv):
        self.agent.set_env(env)

        obs = env.reset()
        done = False
        forces = np.zeros((env.num_envs,), dtype=np.float32)

        i = 0

        while not done:
            i += 1
            act = self.predict(obs, False)
            obs, _, dones, infos = env.step(act)
            done = dones[0]
            forces += [infos[i]['forces'] for i in range(len(infos))]

        self.agent.set_env(self.env)

        return forces

    def _infer_frames(self, env: BurgersFixedSetEnv):
        self.agent.set_env(env)

        obs = np.array(env.reset())
        init = obs[:, :, 0]
        goal = obs[:, :, 1]
        gt_frames = env.frames
        cont_frames = [init]
        pass_frames = [init]

        pass_state = env._get_init_state()

        done = False
        infos = []

        while not done:
            act = self.predict(obs)
            obs, _, dones, infos = env.step(act)
            pass_state = env._step_sim(pass_state, ())
            pass_frames.append(pass_state.velocity.data)
            done = dones[0]
            if not done:
                cont_frames.append(np.array(obs)[:,:,0])

        cont_frames.append(goal)

        self.agent.set_env(self.env)

        return cont_frames, gt_frames, pass_frames

    def _get_forces(self, env: BurgersFixedSetEnv):
        forces = self._infer_forces(env)
        force_avg = np.sum(forces) / env.num_envs

        return force_avg
        #logger.record(scalar_name, force_avg)
        #print('Forces on data set: %f' % force_avg)
