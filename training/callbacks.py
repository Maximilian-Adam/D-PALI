from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        
        if infos and len(infos) > 0:
            # Get info from first (and only) environment
            info = infos[0]
            
            # Check if this info contains the values you want
            if 'cube_target_orientation' in info:
                self.logger.record("custom/cube_target_orientation", info['cube_target_orientation'])
        
            
        return True