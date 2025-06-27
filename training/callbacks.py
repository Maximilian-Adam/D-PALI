from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

class TensorboardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.num_timesteps % 10000 == 0:
            infos = self.locals.get('infos', [])
            if infos and len(infos) > 0:
                # Get info from first (and only) environment
                info = infos[0]
                
                if 'cube_target_orientation' in info:
                    self.logger.record("custom/cube_target_orientation", info['cube_target_orientation'])

                if 'cube_target_quat_similarity' in info:
                    self.logger.record("custom/cube_target_quat_similarity", info['cube_target_quat_similarity'])

                if 'cube_target_distance' in info:
                    self.logger.record("custom/cube_target_distance", info['cube_target_distance'])

                if 'num_contacts' in info:
                    self.logger.record("custom/num_contacts", info['num_contacts'])

        return True
    

class VecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "vec_normalize", verbose: int = 0):
        super(VecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # The get_vec_normalize_env() method is a safe way to get the VecNormalize instance
            # It will return None if the environment is not wrapped in a VecNormalize instance
            vec_normalize_env = self.training_env
            if vec_normalize_env is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
                vec_normalize_env.save(path)
                # For compatibility with the final save, also save a version without the step count
                latest_path = os.path.join(self.save_path, "vec_normalize.pkl")
                vec_normalize_env.save(latest_path)
                if self.verbose > 0:
                    print(f"Saved VecNormalize stats to {path} and {latest_path}")
        return True