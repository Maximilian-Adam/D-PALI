from pathlib import Path
import numpy as np
import mujoco       
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "mjcf"

class DPALI_Hand(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps":     100}

    def __init__(self, xml: str | None = None, frame_skip: int = 5,
                 render_mode: str | None = "human"):

        xml_path = ASSETS / (xml or "DPALI3D.xml")

        super().__init__(
            str(xml_path),        
            frame_skip,
            observation_space=None, 
            render_mode=render_mode,
        )

        # real spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,),
                                       dtype=np.float32)
        obs_dim = self._get_obs().size
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(obs_dim,),
                                            dtype=np.float32)

        # cache the geom ID of "object0" for our reward
        self._obj_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            b"object0",       
         )  # returns -1 if the geom isn’t present
        

    # ---------- helpers ----------
    def _get_obs(self) -> np.ndarray:
        obs = np.concatenate([self.data.qpos.ravel(),
                              self.data.qvel.ravel()])
        return obs.astype(np.float32)          # cast to match space

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = False
        info = {}
        return obs, reward, terminated, False, info

    def reset_model(self):
        noise = self.np_random.uniform(-0.02, 0.02, size=self.model.nq)
        self.set_state(self.init_qpos + noise, self.init_qvel * 0)
        return self._get_obs()

    # ---------- task-specific bits ----------
    def _compute_reward(self):
        if self._obj_id < 0:
            return 0.0
        # pull the world-space position of that geom
        obj_pos = self.data.geom_xpos[self._obj_id]
        target  = np.array([0.1, 0.0, 0.15], dtype=np.float32)
        return -np.linalg.norm(obj_pos - target)

    def _check_done(self):
        return False
