import json
from pathlib import Path
import numpy as np
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces


ASSETS = Path(__file__).resolve().parent.parent.parent / "assets" / "mjcf"

class DPALI_Hand(MujocoEnv):
    # Default hyperparameters and reward weights
    DEFAULT_CONFIG = {
        "xml_file": "DPALI3D.xml",
        "frame_skip": 5,
        "max_episode_steps": 500,
        "render_width": 1920,
        "render_height": 1080,
        "target_pos": [0.015, 0.0, -0.15],
        "cube_initial_pos": [0.015, 0.0, -0.15],
        "target_random_range": [-0.06, 0.06],
        "reward": {
            "success_strict_dist": 0.01,
            "success_loose_dist": 0.005,
            "time_penalty": -0.001,
            "normalisation_scale": 1.0, 
        },
    }

    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": 100}

    def __init__(
        self,
        xml: str | None = None,
        frame_skip: int | None = None,
        render_mode: str | None = "human",
        max_episode_steps: int | None = None,
        config: dict | None = None,
        config_path: str | None = None,
        seed: int | None = None,
    ):
        # Load config from file if provided
        if config_path:
            with open(config_path, 'r') as f:
                ext = json.load(f)
            config = {**self.DEFAULT_CONFIG, **ext}
        else:
            config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.config = config

        # Resolve xml, frame_skip, and episode length
        xml_file = xml or config['xml_file']
        self.frame_skip = frame_skip if frame_skip is not None else config['frame_skip']
        self.max_episode_steps = max_episode_steps if max_episode_steps is not None else config['max_episode_steps']

        # Path to MJCF
        xml_path = ASSETS / xml_file

        self._seed = seed

        #timestep = self.model.opt.timestep      
        #fps      = 1 / (self.frame_skip * timestep)

        calculated_fps = 1 / (frame_skip * 0.002)

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "config": config,
            "seed": self._seed,
            "render_fps": calculated_fps
        }


        # Initialize MujocoEnv (handles internal RNG via reset)
        super().__init__(
            str(xml_path),
            self.frame_skip,
            observation_space=None,
            render_mode=render_mode,
            width=config['render_width'],
            height=config['render_height']
        )

        #timestep = self.model.opt.timestep      
        #fps      = 1 / (self.frame_skip * timestep) 

        # Cache IDs
        self._end_effector_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, b"Hard_tip_L"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, b"Hard_tip_R"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, b"Hard_tip_U"),
        ]

        self._effector_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, b"Tip_L"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, b"Tip_R"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, b"Tip_U"),
        ]

        if any(gid < 0 for gid in self._end_effector_geom_ids):
            raise ValueError("One or more end effector geoms not found in the model.")

        if any(sid < 0 for sid in self._effector_site_ids):
            raise ValueError("One or more end effector sites not found in the model.")

        self._table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, b"table2")

        if self._table_geom_id < 0:
            raise ValueError("'table2' not found in the model.")

        self._cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b"object")
        self._cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, b"object")
        self._target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b"target")

        # Convert positions to numpy arrays
        self._target_pos = np.array(config['target_pos'], dtype=np.float32)
        self._cube_initial_pos = np.array(config['cube_initial_pos'], dtype=np.float32)
        self._workspace_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b"workspace")

        # Episode counter
        self.current_step = 0

        # Define a normalized action space from -1 to 1
        num_actions = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_actions,), dtype=np.float32)
        obs_dim = self._get_obs().size
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):  
        actual_seed = seed if seed is not None else self._seed
        self._seed = actual_seed

        super().reset(seed=actual_seed, options=options)
        self.current_step = 0
        obs = self.reset_model()
        self.metadata['seed'] = self._seed

        return obs, {}

    def _get_obs(self) -> np.ndarray:
        data = self.data
        joint_pos = data.qpos.ravel()
        joint_vel = data.qvel.ravel()
        ee_positions = self._get_all_End_Effector_pos().ravel()
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
    
        ee_to_cube = np.asarray(ee_to_cube, dtype=np.float32)

        cube_to_target_ori = np.array([self.quat_angle(cube_ori, target_ori)], dtype=np.float32)

        
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
            cube_to_target_ori])
            #np.array([quat_similarity]).ravel()]),
        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        scaled_action = action * np.pi
        self.do_simulation(scaled_action, self.frame_skip)
        obs = self._get_obs()
        reward, terminated = self._compute_reward()
        truncated = self.current_step >= self.max_episode_steps
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        mujoco.mj_resetData(self.model, self.data)

        cube_jnt_addr = self.model.body_jntadr[self._cube_id]
        if cube_jnt_addr >= 0:
            self.data.qpos[cube_jnt_addr:cube_jnt_addr + 3] = self._cube_initial_pos
            self.data.qpos[cube_jnt_addr + 3:cube_jnt_addr + 7] = [1, 0, 0, 0]

        mocap_id = self._target_id  
        if mocap_id != -1 and self.model.body_mocapid[mocap_id] != -1:
            mocap_addr = self.model.body_mocapid[mocap_id]

            cube_start_pos = self._cube_initial_pos.copy()
            t_min, t_max = self.config['target_random_range']
            offset = self.np_random.uniform(t_min, t_max, size=2)
            target_pos = cube_start_pos #+ np.array([offset[0], offset[1], 0])
            self.data.mocap_pos[mocap_addr] = target_pos # Use mocap_pos

            rot_axis = [0.0, 0.0, 1.0]
            angle = self.np_random.uniform(0, 2*np.pi)
            target_ori = np.zeros(4)
            mujoco.mju_axisAngle2Quat(target_ori, rot_axis, angle)
            self.data.mocap_quat[mocap_addr] = target_ori

        mujoco.mj_forward(self.model, self.data)


        return self._get_obs()

    def _get_all_End_Effector_pos(self) -> np.ndarray:
        return np.asarray([
            self.data.site_xpos[sid].copy() for sid in self._effector_site_ids
        ], dtype=np.float32)

    def _get_cube_pos(self) -> np.ndarray:
        return self.data.xpos[self._cube_id].copy()

    def _get_target_pos(self) -> np.ndarray:
        return self.data.xpos[self._target_id].copy()
    
    def _get_cube_orientation(self) -> np.ndarray:
        return self.data.xquat[self._cube_id].copy()
    
    def _get_target_orientation(self) -> np.ndarray:
        return self.data.xquat[self._target_id].copy()
    
    def quat_angle(self, q_current: np.ndarray, q_target: np.ndarray) -> float:

        cos_half_theta = abs(float(np.dot(q_current, q_target)))
        cos_half_theta = np.clip(cos_half_theta, -1.0, 1.0)
        return 2.0 * np.arccos(cos_half_theta)

  

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
            
            for tip_idx, tip_id in enumerate(self._end_effector_geom_ids):
                if ((geom1_id == self._cube_geom_id and geom2_id == tip_id) or 
                    (geom2_id == self._cube_geom_id and geom1_id == tip_id)):
                    tip_contact[tip_idx] = True
                    break
        
        return tip_contact, table_contact
    

    def _compute_reward(self):
        cfg = self.config["reward"]

        # --------------- geometry & state ---------------------------------
        ee_pos      = self._get_all_End_Effector_pos()                # (3,3)
        cube_pos    = self._get_cube_pos()
        target_pos  = self._get_target_pos()
        cube_ori    = self._get_cube_orientation()
        target_ori  = self._get_target_orientation()
        contacts, table_contact = self._check_contacts()

        CUBE_RADIUS = 0.024
        MAX_APPROACH_DIST = 0.1

        # --------------- dense sub-rewards (range ≈ [0, 1]) --------------
        # (1) approach – bring fingertips close to the cube
        avg_ee_cube_dist = np.mean([np.linalg.norm(p - cube_pos) for p in ee_pos])
        effective_dist = np.maximum(0, avg_ee_cube_dist - CUBE_RADIUS)
        approach_reward = 1.0 - np.clip(effective_dist / (MAX_APPROACH_DIST - CUBE_RADIUS), 0.0, 1.0)

        # (2) manipulation – move the cube towards the target position
        cube_target_dist   = np.linalg.norm(cube_pos - target_pos)
        manipulation_reward = 1.0 - np.clip(cube_target_dist / 0.10, 0.0, 1.0)

        # (3) orientation – align cube orientation to the target
        ori_err = self.quat_angle(cube_ori, target_ori)
        scaled_err = np.clip(ori_err / np.pi, 0.0, 1.0)
        orientation_reward = 1.0 - (scaled_err ** 2)

        # (4) contacts – encourage a stable 3-finger grasp
        num_contacts       = sum(contacts)           
        if num_contacts == 0:
            contact_reward = -0.2
        elif num_contacts == 1:
            contact_reward = 0        
        elif num_contacts == 2:
            contact_reward = 0.2
        elif num_contacts == 3:
            contact_reward = 1

        centering_reward = 0.0
        if num_contacts == 3:
            ee_centroid = np.mean(ee_pos, axis=0)
            centering_error = np.linalg.norm(ee_centroid - cube_pos)
            centering_reward = 1.0 - np.clip(centering_error / 0.02, 0.0, 1.0)

        # weighted sum (weights sum to 1)
        shaped_reward = (
            0.1 * approach_reward
          + 0.2 * centering_reward
          + 0.2 * contact_reward
          #+ 0.3 * manipulation_reward
          + 0.5 * orientation_reward
        )

        # --------------- sparse bonuses & penalties ----------------------
        terminated    = False
        success_bonus = 0.0
        if (cube_target_dist < cfg["success_strict_dist"]
                and ori_err < 0.15
                and num_contacts == 3
                and not table_contact):
            success_bonus = 25   
            terminated    = True

        # Severe penalty for dropping or touching the table
        drop_penalty  = -1.0 if table_contact else 0.0


        penetration_penalty = 0.0
        PENALTY_SCALE = 20.0 
        for p in ee_pos:
            dist = np.linalg.norm(p - cube_pos)
            if dist < CUBE_RADIUS:
                penetration_depth = CUBE_RADIUS - dist
                penetration_penalty -= PENALTY_SCALE * penetration_depth

        # Small per-step time penalty to encourage speed
        time_penalty  = cfg["time_penalty"] 

        # --------------- aggregate & normalise ---------------------------
        total_reward = (
              shaped_reward
            + success_bonus
            + drop_penalty     
            + time_penalty
            + penetration_penalty
        )

        return total_reward, terminated

    
    def _get_info(self):
        """Return diagnostic information."""
        contacts, _ = self._check_contacts()
        cube_pos = self._get_cube_pos()
        target_pos = self._get_target_pos()
        target_ori = self._get_target_orientation()
        cube_ori = self._get_cube_orientation()
        ori_diff = self.quat_angle(cube_ori, target_ori)

        # Handle zero quaternions (initialization case)
        cube_norm = np.linalg.norm(cube_ori)
        target_norm = np.linalg.norm(target_ori)

        if cube_norm < 1e-6:  # Essentially zero
            cube_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        else:
            cube_quat = cube_ori / cube_norm

        if target_norm < 1e-6:  # Essentially zero
            target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        else:
            target_quat = target_ori / target_norm
        quat_similarity = np.clip(np.abs(np.dot(cube_quat, target_quat)), 0, 1.0)
        ori_angle = 2 * np.arccos(quat_similarity)  # Convert to angle in radians

        
        return {
            'cube_target_distance': np.linalg.norm(cube_pos - target_pos),
            'cube_target_quat_similarity': quat_similarity,
            'cube_target_orientation': ori_diff,
            'num_contacts': sum(contacts),
            'contacts': contacts,
            'cube_position': cube_pos,
            'target_position': target_pos,
            'cube_orientation': cube_ori,
            'target_orientation': target_ori,
            'episode_step': self.current_step,
            'max_episode_steps': self.max_episode_steps,
            'seed': self._seed
        }

    # Debug helpers
    def print_End_Effector_pos(self):  print(self._get_all_End_Effector_pos())
    def print_cube_pos(self):          print(self._get_cube_pos())
    def print_target_pos(self):        print(self._get_target_pos())
