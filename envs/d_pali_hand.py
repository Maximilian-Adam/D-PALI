from gymnasium.envs.mujoco import MuJocoEnv
from gymnasium import spaces
import numpy as np

class DPALI_Hand(MuJocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, xml="assets/d_pali_hand.xml", frame_skip=5):
        super().__init__(xml, frame_skip, render_mode="human")
        self.action_space      = spaces.Box(-1, 1, shape=(self.model.nu,), dtype=np.float32)
        obs_dim                = self.get_obs().size
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    # --------- mandatory hooks ----------
    def get_obs(self):
        qpos = self.data.qpos.flat.copy()      # joint angles
        qvel = self.data.qvel.flat.copy()      # joint velocities
        return np.concatenate([qpos, qvel])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs   = self.get_obs()
        reward= self.compute_reward()
        term  = self.check_done()
        info  = {}
        return obs, reward, term, False, info

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel * 0,
        )
        return self.get_obs()

    # --------- task-specific helpers -------
    def compute_reward(self):
        block_pos = self.data.geom("object").xpos
        target    = np.array([0.1, 0.0, 0.15])
        return -np.linalg.norm(block_pos - target)

    def check_done(self):
        return False
    

from gymnasium.envs.registration import register
register(id="d_pali_hand-v0", entry_point="d_pali_env:DPALI_HandEnv")
