from stable_baselines3.common.callbacks import BaseCallback

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

                if 'cube_target_distance' in info:
                    self.logger.record("custom/cube_target_distance", info['cube_target_distance'])

                if 'num_contacts' in info:
                    self.logger.record("custom/num_contacts", info['num_contacts'])

        return True