from gym_phiflow.envs.burgers_env import BurgersEnv
import phi.flow as phiflow
import numpy as np

class BurgersFixedSetEnv(BurgersEnv):
    def __init__(self, data_path, data_range, test_mode=True, **burgers_env_kwargs):
        assert burgers_env_kwargs['num_envs'] == 1 or burgers_env_kwargs['num_envs'] == len(data_range)
        print(burgers_env_kwargs)
        super().__init__(**burgers_env_kwargs)

        dataset = phiflow.Dataset.load(data_path, data_range)
        self.test_mode = test_mode        
        self.frames = self._get_frames_from_dataset(dataset)
        self.dataset_idx = -1

    def _get_init_state(self):
        #self.dataset_idx += 1
        state = self._get_state_of_sims_at_frame(0)
        return state

    def _get_gt_forces(self):
        frame_0 = self._get_state_of_sims_at_frame(0)
        frame_1 = self._get_state_of_sims_at_frame(1)
        forces = frame_1.velocity - self.physics.step(frame_0, self.dt).velocity
        return phiflow.FieldEffect(forces, ['velocity'])

    def _get_goal_state(self):
        return self._get_state_of_sims_at_frame(-1)

    def _step_gt(self):
        return self._get_state_of_sims_at_frame(self.step_idx)
    
    def _get_state_of_sims_at_frame(self, frame_idx):
        return phiflow.BurgersVelocity(self.domain, self.frames[frame_idx], viscosity=self.viscosity)

    def _get_frames_from_dataset(self, dataset):
        frames = np.array([list(source.get('burgers_velocity', range(self.step_count+1))) for source in dataset.sources], dtype=np.float32).squeeze(2)
        return np.swapaxes(frames, 0, 1)
