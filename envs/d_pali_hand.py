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
        self.action_space = spaces.Box(-1, 1, shape=(self.model.nu,),
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


        # cache the End Effectors geom ID for our reward
        self._end_effector_id = [
            mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,b"Hard_tip_L"),
            mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,b"Hard_tip_R"),
            mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,b"Hard_tip_U"),
        ]
        # cache the cube ID for our reward
        self._cube_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            b"object",       
        )

        self._cube_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            b"object",
        )

        self._target_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            b"target",
        )
        

    # ---------- helpers ----------
    def _get_obs(self) -> np.ndarray:
        obs = np.concatenate([self.data.qpos.ravel(), #Joint Positions and velocities
                              self.data.qvel.ravel()])
        return obs.astype(np.float32)          # cast to match space

    def step(self, action):
        scaled_action = action * np.pi/3
        self.do_simulation(scaled_action, self.frame_skip)
        obs = self._get_obs()
        reward, terminated = self._compute_reward()
        info = {}
        return obs, reward, terminated, False, info

    def reset_model(self):
        noise = self.np_random.uniform(-0.02, 0.02, size=self.model.nq)
        self.set_state(self.init_qpos + noise, self.init_qvel * 0)

        cube_pos = np.array([0.0, -0.1345, 0.0]) #static cube position
        self.model.body_pos[self._cube_id] = cube_pos
        # cube_qpos_addr = self.model.body_jntadr[self._cube_id]  # index in cube pos
        # self.data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = cube_pos
        
        target_x = self.np_random.uniform(-0.04, 0.04)
        target_y = self.np_random.uniform(-0.04, 0.04)
        target_z = self.np_random.uniform(-0.14, -0.14)
        target_pos = np.array([target_x, target_y, target_z])

        self.model.body_pos[self._target_id] = target_pos
        
        return self._get_obs()

    # ---------- task-specific bits ----------
    def _get_all_End_Effector_pos(self)-> np.ndarray:
        Tip_L_pos = self.data.geom_xpos[self._end_effector_id[0]]
        Tip_R_pos = self.data.geom_xpos[self._end_effector_id[1]]
        Tip_U_pos = self.data.geom_xpos[self._end_effector_id[2]]
        #print(f"End Effector position: {(Tip_L_pos, Tip_R_pos, Tip_U_pos)}")
        return np.array([Tip_L_pos, Tip_R_pos, Tip_U_pos], dtype=np.float32)
    
    
    def _get_cube_pos(self)-> np.ndarray:
        cube_pos = self.data.xpos[self._cube_id]
        return cube_pos
    
    def _get_target_pos(self)-> np.ndarray:
        target_pos = self.data.xpos[self._target_id]
        return target_pos
    
    def _check_contacts(self):
        #Check if each end effector is in contact with the cube.
        tip_contact = [False, False, False]  # L, R, U tips
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            for tip_idx, tip_id in enumerate(self._end_effector_id):
                if (geom1_id == self._cube_geom_id and geom2_id == tip_id) or \
                (geom2_id == self._cube_geom_id and geom1_id == tip_id):
                    tip_contact[tip_idx] = True
                    break
        
        return tip_contact
    
    def _compute_reward(self):
        End_Effector_pos = self._get_all_End_Effector_pos()

        cube_pos = self._get_cube_pos()
        target_pos  = self._get_target_pos()
        dist_reward = -np.linalg.norm(cube_pos - target_pos)

        contacts = self._check_contacts()
        contact_reward = sum(0.5 for contact in contacts if contact)
        
        # Additional reward for having multiple contacts (encourages grasping)
        if sum(contacts) >= 2:
            contact_reward += 1.0
        if sum(contacts) == 3:
            contact_reward += 2.0
        
        total_reward = dist_reward + contact_reward
        
        done = False
        # Add a large bonus when cube is very close to target while being grasped
        if -dist_reward < 0.01:
            done = True
            if sum(contacts) >= 2:
                total_reward += 5.0    
        

        return total_reward, done

    

    # ---------- debug ----------
    def print_End_Effector_pos(self):
        print(f"End Effector position: {self._get_all_End_Effector_pos()}")
    def print_cube_pos(self):
        print(f"Cube position: {self._get_cube_pos()}")