from pathlib import Path
import numpy as np
import mujoco       
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "mjcf"

class DPALI_Hand(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps":     100}

    def __init__(self, xml: str | None = None, 
                 frame_skip: int = 5,
                 render_mode: str | None = "human"):

        xml_path = ASSETS / (xml or "DPALI3D.xml")

        super().__init__(
            str(xml_path),        
            frame_skip,
            observation_space=None, 
            render_mode=render_mode,
            width = 1920,
            height = 1080,
        )

        # real spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,),
                                       dtype=np.float32)
        obs_dim = self._get_obs().size
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(obs_dim,),
                                            dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            obs = self.reset_model()
            self.sim.forward()  # Apply manual qpos changes
            return obs, {}


        # cache the geom ID of "object0" for our reward
        self._obj_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            b"object0",       
        )  # returns -1 if the geom isn’t present

        # cache the End Effectors geom ID for our reward
        self._end_effector_id = [
            mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,b"Hard_tip_L"),
            mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,b"Hard_tip_R"),
            mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,b"Hard_tip_U"),
        ]
        # cache the cube ID for our reward
        self._cube_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            b"cube",       
        )
        

    # ---------- helpers ----------
    def _get_obs(self) -> np.ndarray:
        obs = np.concatenate([self.data.qpos.ravel(),
                              self.data.qvel.ravel()])
        return obs.astype(np.float32)          # cast to match space

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_done()
        info = {}
        return obs, reward, terminated, False, info

    def reset_model(self):
        noise = self.np_random.uniform(-0.02, 0.02, size=self.model.nq)
        self.set_state(self.init_qpos + noise, self.init_qvel * 0)
        cube_qpos_addr = self.model.body_jntadr[self._cube_id]  # index in qpos
        cube_x = self.np_random.uniform(-0.2, 0.2)
        cube_y = self.np_random.uniform(-0.3, 0.0)
        cube_z = self.np_random.uniform(-0.2, 0.2)
        self.data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = np.array([cube_x, cube_y, cube_z])
        return self._get_obs()

    # ---------- task-specific bits ----------
    def _get_all_End_Effector_pos(self)-> np.ndarray:
        Tip_L_pos = self.data.geom_xpos[self._end_effector_id[0]]
        Tip_R_pos = self.data.geom_xpos[self._end_effector_id[1]]
        Tip_U_pos = self.data.geom_xpos[self._end_effector_id[2]]
        #print(f"End Effector position: {(Tip_L_pos, Tip_R_pos, Tip_U_pos)}")
        return np.array([Tip_L_pos, Tip_R_pos, Tip_U_pos], dtype=np.float32)
    def _get_End_Effector_pos(self, tip_name)-> np.ndarray:
        match tip_name:
            case "L":
                return self.data.geom_xpos[self._end_effector_id[0]]
            case "R":
                return self.data.geom_xpos[self._end_effector_id[1]]
            case "U":
                return self.data.geom_xpos[self._end_effector_id[2]]
            case _:
                raise ValueError(f"Unknown tip name: {tip_name}")
    def _get_cube_pos(self)-> np.ndarray:
        cube_pos = self.data.geom_xpos[self._cube_id]
        return cube_pos
    
    def _compute_reward(self):
        # Should be customized for the training task
        # here is the example of setting the reward to be the negitive distance
        # between the end effector and the target
        End_Effector_pos = self._get_End_Effector_pos('L')
        target  = self._get_cube_pos()
        diff = -np.linalg.norm(End_Effector_pos - target) 
        diff -= np.linalg.norm(End_Effector_pos - target) 
        diff -= np.linalg.norm(End_Effector_pos - target)
        return diff

    def _check_done(self):
        if self._compute_reward() > -0.001:
            return True
        return False
    

    # ---------- debug ----------
    def print_End_Effector_pos(self):
        print(f"End Effector position: {self._get_all_End_Effector_pos()}")
    def print_cube_pos(self):
        print(f"Cube position: {self._get_cube_pos()}")