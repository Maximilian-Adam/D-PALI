from pathlib import Path
import numpy as np
import mujoco       
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "mjcf"

class DPALI_Hand(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array","depth_array"],
                "render_fps": 150}

    def __init__(self, xml: str | None = None, 
                 frame_skip: int = 5,
                 render_mode: str | None = "human"):

        xml_path = ASSETS / (xml or "DPALI3D.xml")

        calculated_fps = 1 / (frame_skip * 0.002)

        self.metadata = {
            "render_modes": ["human", "rgb_array","depth_array"],
            "render_fps": calculated_fps
        }
        
        
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
        
        # Cache the cube ID for our reward
        self._cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b"object")
        self._cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, b"object")
        self._target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b"target")
        
        # Fixed target position
        self._target_pos = np.array([0.015, 0.0, -0.155])
        self._cube_initial_pos = np.array([0.015, 0, -0.15])
        
        # Episode management
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Set up action and observation spaces after initialization
        self.action_space = spaces.Box(-1, 1, shape=(self.model.nu,), dtype=np.float32)
        obs_dim = self._get_obs().size
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.last_action = None
        self.prev_action = None
        self.prev_reward = None

    # FIXED: Properly indented reset method
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0  # Reset step counter
        obs = self.reset_model()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        obs = np.concatenate([self.data.qpos.ravel(), #Joint Positions and velocities
                              self.data.qvel.ravel()])
        return obs.astype(np.float32)          # cast to match space

    def step(self, action):
        self.current_step += 1
        self.prev_action = self.last_action.copy()  
        self.last_action = action                   
        scaled_action = action #* np.pi/3
        self.do_simulation(scaled_action, self.frame_skip)
        obs = self._get_obs()
        reward, terminated = self._compute_reward()
        info = {}
        return obs, reward, terminated, False, info

    def reset_model(self):
        noise = self.np_random.uniform(-0.02, 0.02, size=self.model.nq)
        self.set_state(self.init_qpos + noise, self.init_qvel * 0)

        # Set cube to fixed initial position
        cube_jnt_addr = self.model.body_jntadr[self._cube_id]
        if cube_jnt_addr >= 0:  # Check if the body has joints
            self.data.qpos[cube_jnt_addr:cube_jnt_addr + 3] = self._cube_initial_pos
        
        target_y = self.np_random.uniform(-0.04, 0.04)
        target_pos = np.array([0.015, target_y, -0.15])

        self.model.body_pos[self._target_id] = target_pos
        
        # Forward the simulation to apply changes
        mujoco.mj_forward(self.model, self.data)

        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_action = np.zeros_like(self.last_action)
        self.prev_reward = None
        return self._get_obs()

    # ---------- task-specific bits ----------
    def _get_all_End_Effector_pos(self)-> np.ndarray:
        Tip_L_pos = self.data.geom_xpos[self._end_effector_id[0]]
        Tip_R_pos = self.data.geom_xpos[self._end_effector_id[1]]
        Tip_U_pos = self.data.geom_xpos[self._end_effector_id[2]]
        #print(f"End Effector position: {(Tip_L_pos, Tip_R_pos, Tip_U_pos)}")
        return np.array([Tip_L_pos, Tip_R_pos, Tip_U_pos], dtype=np.float32)
    
    def _get_cube_pos(self) -> np.ndarray:
        return self.data.xpos[self._cube_id].copy()
    
    def _get_cube_orientation(self) -> np.ndarray:
        return self.data.xmat[self._cube_id].reshape(3, 3)


    def _get_target_pos(self) -> np.ndarray:
        return self.data.xpos[self._target_id].copy()

    def _get_target_high(self) -> float:
        target_pos = self._get_target_pos()
        return target_pos[2]
    
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
        cube_rotmat = self._get_cube_orientation()
        alignment = np.abs(np.dot(cube_rotmat[:, 2], [0, 0, 1]))
        cube_high = cube_pos[2]
        target_pos = self._get_target_pos()
        contacts = self._check_contacts()
        
        delta_action = self.last_action - self.prev_action

        # === PHASE 1: APPROACH PHASE ===
        # Encourage gripper to approach cube
        ee_cube_distances = [np.linalg.norm(ee_pos - cube_pos) for ee_pos in end_effector_pos]
        avg_ee_cube_dist = np.mean(ee_cube_distances)
        
        # Dense approach reward (decreases as gripper gets closer)
        approach_reward = -avg_ee_cube_dist * 100.0
        
        # === PHASE 2: CONTACT PHASE ===
        num_contacts = sum(contacts)
        pose_reward = 0.0
        # Progressive contact rewards - encourage all 3 contacts
        if num_contacts == 1:
            contact_reward = 50.0  # First contact bonus
        elif num_contacts == 2:
            contact_reward = 100.0  # Two-finger grasp bonus
            pose_reward = 100 * (1 / (1 + np.exp(-20 * (alignment - 0.95))) - 0.5)
        elif num_contacts == 3:
            contact_reward = 200.0  # Perfect three-finger grasp bonus
            pose_reward = 50 * (1 / (1 + np.exp(-20 * (alignment - 0.95))) - 0.5)
        else:
            contact_reward = 0.0
        
        # === PHASE 3: MANIPULATION PHASE ===
        cube_target_dist = np.linalg.norm(cube_pos - target_pos)
        
        manipulation_reward = 1/cube_target_dist



        # Different strategies based on grasp quality
        if num_contacts >= 2:  # Good grasp - focus on moving to target
            # Bonus for maintaining grasp while moving
            grasp_maintenance_bonus = 20.0 if num_contacts == 3 else 10.0
        else:  # Poor grasp - still encourage target movement but less
            grasp_maintenance_bonus = 0.0

        cube_high_bonus = 0.0
        if cube_high > -0.15:
            # Encourage keeping the cube above a certain height
            cube_high_bonus = 1/abs(cube_high + self._get_target_high())*10

        

        # === PHASE 4: SUCCESS PHASE ===
        success_bonus = 0.0
        terminated = False
        
        # Success criteria: cube very close to target with stable grasp
        if cube_target_dist < 0.002 and num_contacts >= 2:
            success_bonus = 300.0
            terminated = True
        elif cube_target_dist < 0.005 and num_contacts >= 2:
            # Close to success bonus
            success_bonus = 500.0
        
        # === PENALTIES ===
        # Time penalty to encourage efficiency
        time_penalty = -20
        if cube_high <= -0.1:
            # Penalize if cube is too low (e.g., dropped)
            drop_penalty = -10.0
        else:
            drop_penalty = 0.0
        if num_contacts == 0:
            contact_penalty = -50.0
        else:
            contact_penalty = 0.0

        action_change_penalty = -0.05 * np.sum(np.square(delta_action))  

    
        
        # Total reward
        reward = (approach_reward + 
                  contact_reward + 
                  manipulation_reward +
                  grasp_maintenance_bonus + 
                  cube_high_bonus +
                  pose_reward +
                  success_bonus + 
                  time_penalty + 
                  drop_penalty +
                  contact_penalty +
                  action_change_penalty)
        
        # if self.prev_reward is not None:
        #     reward_diff = abs(reward - self.prev_reward)
        #     if reward_diff < 20:  
        #         #lazy_penalty = -1000  
        #         reward = 0
        
        # self.prev_reward = reward
        
        return reward, terminated
    
    def _get_info(self):
        """Return diagnostic information."""
        contacts = self._check_contacts()
        cube_pos = self._get_cube_pos()
        cube_rotmat = self._get_cube_orientation()
        alignment = np.abs(np.dot(cube_rotmat[:, 2], [0, 0, 1]))
        cube_high = cube_pos[2]
        target_pos = self._get_target_pos()

        cube_target_dist = np.linalg.norm(cube_pos - target_pos)
        manipulation_reward = 1/cube_target_dist

        
        return {
            'cube_target_distance': cube_target_dist,
            'manipulation_reward': manipulation_reward,
            'num_contacts': sum(contacts),
            'contacts': contacts,
            'cube_position': cube_pos,
            'cube_alignment': alignment,
            'cube_height': cube_high,
            'target_position': target_pos,
            'episode_step': self.current_step,
            'max_episode_steps': self.max_episode_steps
        }

    # Debug methods
    def print_End_Effector_pos(self):
        print(f"End Effector position: {self._get_all_End_Effector_pos()}")
    def print_cube_pos(self):
        print(f"Cube position: {self._get_cube_pos()}")