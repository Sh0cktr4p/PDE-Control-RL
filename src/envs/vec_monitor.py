from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common import logger
from typing import List, Optional, Tuple
import time
import csv
import json
import numpy as np
import os


class VecMonitor(VecEnvWrapper):
    '''
    A monitor wrapper for vectorized Gym environments, it is used to know the episode reward, length, time and other data.
    
    :param venv: the vectorized environment
    :param rollout size: number of steps per rollout
    :param filename: the location to save a log file, can be None for no log
    :param info_keywords: extra information to log, from the information return of env.step()
    '''

    EXT = 'monitor.csv'

    def __init__(
        self, 
        venv: VecEnv, 
        rollout_size: int,
        filename: Optional[str]=None,
        info_keywords: Tuple[str, ...]=(),
    ):
        super().__init__(venv)
        self.t_start = time.time()
        fieldnames = ("r", "l", "t") + info_keywords

        if filename is None:
            self.file_handler = None
            self.logger = None
        else:
            if not filename.endswith(VecMonitor.EXT):
                if os.path.isdir(filename):
                    filename = os.path.join(filename, VecMonitor.EXT)
                else:
                    filename = filename + '.' + VecMonitor.EXT
            self.file_handler = open(filename, 'at')
            self.logger = csv.DictWriter(self.file_handler, fieldnames=fieldnames)
            if os.path.getsize(filename) == 0:
                self.file_handler.write("#%s\n" % json.dumps({"t_start": self.t_start}))
                self.logger.writeheader()
                self.file_handler.flush()
        
        self.info_keywords = info_keywords
        self.curr_ep_rewards = None
        self.curr_ep_lengths = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.curr_rollout_data = {fn: [] for fn in fieldnames}
        self.curr_ep_data = {fn: np.zeros(self.num_envs, dtype=np.float32) for fn in info_keywords}
        self.total_steps = 0
        self.step_idx_in_rollout = 0
        self.rollout_size = rollout_size

    def reset(self):
        '''
        Calls the vectorized environment reset.
        :return: the first observation of the vectorized environments
        '''
        self.needs_reset = False
        obs = self.venv.reset()
        self.curr_ep_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.curr_ep_lengths = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        for key in self.curr_rollout_data:
            self.curr_rollout_data[key] = []
        for key in self.curr_ep_data:
            self.curr_ep_data[key] = np.zeros(self.num_envs, dtype=np.float32)
        self.step_idx_in_rollout = 0
        return obs

    def step_wait(self):
        if self.needs_reset:
            raise RuntimeError('Tried to step vectorized environment that needs reset!')

        obss, rews, dones, infos = self.venv.step_wait()
        
        self.curr_ep_rewards += rews
        self.curr_ep_lengths += 1

        new_infos = list(infos[:])
        for key in self.curr_ep_data:
            self.curr_ep_data[key] += [info[key] for info in infos] #[dk for dk in map(lambda d: d[key], infos)]

        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ep_rew = self.curr_ep_rewards[i]
                ep_len = self.curr_ep_lengths[i]
                ep_time = round(time.time() - self.t_start, 6)
                ep_info = {'r': ep_rew, 'l': ep_len, 't': ep_time}
                for key in self.curr_ep_data:
                    # Change in behavior: grab only the values in episode that would be overwritten
                    ep_info[key] = self.curr_ep_data[key][i]
                    self.curr_ep_data[key][i] = 0
                self.episode_rewards.append(ep_rew)
                self.episode_lengths.append(ep_len)
                self.episode_times.append(ep_time)
                self.curr_ep_rewards[i] = 0
                self.curr_ep_lengths[i] = 0
                if self.logger:
                    for key in self.curr_rollout_data:
                        self.curr_rollout_data[key].append(ep_info[key])
                info['episode'] = ep_info
                new_infos[i] = info
        self.total_steps += self.num_envs
        self.step_idx_in_rollout += 1

        if self.step_idx_in_rollout == self.rollout_size:
            if self.logger:
                # Correct the value for time (a bit ugly, I know)
                if 't' in self.curr_rollout_data:
                    self.curr_rollout_data['t'] = [time.time() - self.t_start]
                # Store the average values per rollout
                self.logger.writerow({k:safe_mean(self.curr_rollout_data[k]) for k in self.curr_rollout_data})
                self.file_handler.flush()
                for key in self.curr_rollout_data:
                    self.curr_rollout_data[key] = []
                self.step_idx_in_rollout = 0

        return obss, rews, dones, new_infos

    def close(self) -> None:
        if self.file_handler is not None:
            self.file_handler.close()
        super().close()

    def get_total_steps(self) -> int:
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        return self.episode_rewards

    def get_episode_lengths(self) -> List[int]:
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        return self.episode_times
