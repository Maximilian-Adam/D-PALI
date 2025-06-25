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
                 render_mode: str | None = "human",
                 max_episode_steps: int = 500):

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
        self._table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, b"table2")
        
        # Fixed target position
        self._target_pos = np.array([0.06, 0.0, -0.1])
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
        # Enhanced observation including task-relevant information
        joint_pos = self.data.qpos.ravel()
        joint_vel = self.data.qvel.ravel()
        
        # Get end effector positions
        ee_positions = self._get_all_End_Effector_pos().ravel()
        
        # Get cube and target positions
        cube_pos = self._get_cube_pos()
        target_pos = self._get_target_pos()

        # Get cube and target orientations
        cube_ori = self._get_cube_orientation()
        target_ori = self._get_target_orientation()

        cube_norm = np.linalg.norm(cube_ori)
        target_norm = np.linalg.norm(target_ori)
        if cube_norm < 1e-6: 
            cube_quat = np.array([1.0, 0.0, 0.0, 0.0]) 
        else:
            cube_quat = cube_ori / cube_norm
        if target_norm < 1e-6:
            target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            target_quat = target_ori / target_norm

        # Get contact information
        contacts, _ = self._check_contacts()
        contacts = np.array(contacts, dtype=np.float32)
        
        # Relative positions (cube to target, end effectors to cube)
        cube_to_target = target_pos - cube_pos
        ee_to_cube = []
        for ee_pos in self._get_all_End_Effector_pos():
            ee_to_cube.extend(cube_pos - ee_pos)

        quat_similarity = np.clip(np.abs(np.dot(cube_quat, target_quat)),0,1)

        obs = np.concatenate([
            joint_pos,
            joint_vel,
            ee_positions,
            cube_pos,
            target_pos,
            cube_to_target,
            ee_to_cube,
            contacts,
            cube_ori,
            target_ori,
            np.array([quat_similarity]).ravel()])
        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        if self.last_action is not None:
            self.prev_action = self.last_action.copy()  
        self.last_action = action                   
        scaled_action = action #* np.pi/3
        self.do_simulation(scaled_action, self.frame_skip)
        obs = self._get_obs()
        reward, terminated = self._compute_reward()
        info = self._get_info()
        return obs, reward, terminated, False, info

    def reset_model(self):
        # Add noise to initial joint positions
        noise = self.np_random.uniform(-0.02, 0.02, size=self.model.nq)
        self.set_state(self.init_qpos + noise, self.init_qvel * 0)

        # Set cube to fixed initial position
        cube_jnt_addr = self.model.body_jntadr[self._cube_id]
        if cube_jnt_addr >= 0:  # Check if the body has joints
            self.data.qpos[cube_jnt_addr:cube_jnt_addr + 3] = self._cube_initial_pos
        
        rot_axis = [0.0, 0.0, 1.0] # Z-axis rotation
        angle = np.random.uniform(0, np.pi) 
        rotation = np.zeros(4)
        mujoco.mju_axisAngle2Quat(rotation, rot_axis, angle)

        # Reset target orientation
        target_ori = rotation  # Identity quaternion
        self.model.body_quat[self._target_id] = target_ori
        self.model.body_pos[self._target_id] = self._target_pos
        
        # Forward the simulation to apply changes
        mujoco.mj_forward(self.model, self.data)
        
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
        return self.data.xquat[self._cube_id].copy()


    def _get_target_pos(self) -> np.ndarray:
        return self.data.xpos[self._target_id].copy()

    def _get_target_high(self) -> float:
        target_pos = self._get_target_pos()
        return target_pos[2]
    
    def _get_target_orientation(self) -> np.ndarray:
        return self.data.xquat[self._target_id].copy()
    
    def _check_contacts(self):
        """Check if each end effector is in contact with the cube."""
        tip_contact = [False, False, False]  # L, R, U tips
        table_contact = False
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            if (geom1_id == self._table_geom_id and geom2_id == self._cube_geom_id) or \
               (geom2_id == self._table_geom_id and geom1_id == self._cube_geom_id):
                table_contact = True
                continue
            
            for tip_idx, tip_id in enumerate(self._end_effector_id):
                if ((geom1_id == self._cube_geom_id and geom2_id == tip_id) or 
                    (geom2_id == self._cube_geom_id and geom1_id == tip_id)):
                    tip_contact[tip_idx] = True
                    break
        
        return tip_contact, table_contact
    def quat_conjugate(self,q):
        return np.array([q[0], -q[1], -q[2], -q[3]])


    def quat_mul(self,q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def quat_to_axis_angle(self,q):
        if q[0] > 1.0: 
            q = q / np.linalg.norm(q)
        angle = 2 * np.arccos(q[0])
        s = np.sqrt(1 - q[0] * q[0])
        if s < 1e-6:
            axis = np.array([1, 0, 0]) 
        else:
            axis = q[1:] / s
        return axis, angle
    
    def _compute_reward(self):
        End_Effector_pos = self._get_all_End_Effector_pos()

        cube_pos = self._get_cube_pos()
        cube_high = cube_pos[2]

        target_pos = self._get_target_pos()

        cube_ori = self._get_cube_orientation()
        target_ori = self._get_target_orientation()
        ori_delta = self.quat_mul(target_ori, self.quat_conjugate(cube_ori))
        axis, angle = self.quat_to_axis_angle(ori_delta)

        contacts, table_contacts = self._check_contacts()
        
        if self.prev_action is not None:
            delta_action = self.last_action - self.prev_action
        else:
            delta_action = np.zeros_like(self.last_action)
        # === PHASE 1: APPROACH PHASE ===
        # Encourage gripper to approach cube
        ee_cube_distances = [np.linalg.norm(ee_pos - cube_pos) for ee_pos in End_Effector_pos]
        avg_ee_cube_dist = np.mean(ee_cube_distances)
        
        # Dense approach reward (decreases as gripper gets closer)
        approach_reward = -avg_ee_cube_dist * 2000.0
        
        # === PHASE 2: CONTACT PHASE ===
        num_contacts = sum(contacts)
        pose_reward = 0.0
        if num_contacts > 1:
            pose_reward = 1000 / (1 + angle)
        # Progressive contact rewards - encourage all 3 contacts
        if num_contacts == 1:
            contact_reward = 50.0  # First contact bonus
        elif num_contacts >= 2:
            contact_reward = 100.0  # Two-finger grasp bonus
        else:
            contact_reward = 0.0
        
        # === PHASE 3: MANIPULATION PHASE ===
        cube_target_dist = np.linalg.norm(cube_pos - target_pos)
        
        manipulation_reward = 10/cube_target_dist



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
        table_contacts_penalty = -200.0 if table_contacts else 0.0
    
        
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
                  action_change_penalty+
                  table_contacts_penalty)
        
        # if self.prev_reward is not None:
        #     reward_diff = abs(reward - self.prev_reward)
        #     if reward_diff < 20:  
        #         #lazy_penalty = -1000  
        #         reward = 0
        
        # self.prev_reward = reward
        
        return reward, terminated
    
    def _get_info(self):
        """Return diagnostic information."""
        contacts, table_contact = self._check_contacts()
        cube_pos = self._get_cube_pos()

        cube_high = cube_pos[2]
        target_pos = self._get_target_pos()

        cube_target_dist = np.linalg.norm(cube_pos - target_pos)
        manipulation_reward = 1/cube_target_dist

        cube_ori = self._get_cube_orientation()
        target_ori = self._get_target_orientation()
        ori_delta = self.quat_mul(target_ori, self.quat_conjugate(cube_ori))
        axis, angle = self.quat_to_axis_angle(ori_delta)
        
        return {
            'cube_target_distance': cube_target_dist,
            'manipulation_reward': manipulation_reward,
            'num_contacts': sum(contacts),
            'contacts': contacts,
            'cube_position': cube_pos,
            'cube_height': cube_high,
            'target_position': target_pos,
            'episode_step': self.current_step,
            'max_episode_steps': self.max_episode_steps,
            'axis_angle': angle,
            'axis': axis,
        }

    # Debug methods
    def print_End_Effector_pos(self):
        print(f"End Effector position: {self._get_all_End_Effector_pos()}")
    def print_cube_pos(self):
        print(f"Cube position: {self._get_cube_pos()}")