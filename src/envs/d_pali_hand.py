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
        "target_pos": [0.015, 0.0, -0.155],
        "cube_initial_pos": [0.015, 0.0, -0.15],
        "target_random_y_range": [-0.02, 0.02],
        "reward": {
            "approach_mul": 5.0,
            "manipulation_mul": 100.0,
            "contact_rewards": {0: 0.0, 1: 10.0, 2: 25.0, 3: 50.0},
            "grasp_bonus_two": 10.0,
            "grasp_bonus_three": 20.0,
            "success_strict_dist": 0.002,
            "success_loose_dist": 0.005,
            "success_strict_reward": 500.0,
            "success_loose_reward": 50.0,
            "time_penalty": -0.5,
        },
    }

    # Expose render modes, fps, config, and seed in metadata
    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": None,
                "config": None,
                "seed": None}

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

        # Compute fps for rendering
        fps = 1 / (self.frame_skip * 0.002)

        self._seed = seed

        # Update metadata
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": fps,
            "config": config,
            "seed": self._seed
        }

        # Initialize MujocoEnv (handles internal RNG via reset)
        super().__init__(
            str(xml_path),
            self.frame_skip,
            observation_space=None,
            render_mode=render_mode,
            width=config['render_width'],
            height=config['render_height'],
        )

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

        self._cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b"object")
        self._cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, b"object")
        self._target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b"target")

        # Convert positions to numpy arrays
        self._target_pos = np.array(config['target_pos'], dtype=np.float32)
        self._cube_initial_pos = np.array(config['cube_initial_pos'], dtype=np.float32)

        # Episode counter
        self.current_step = 0

        # Define spaces
        self.action_space = spaces.Box(-1, 1, shape=(self.model.nu,), dtype=np.float32)
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

        # Get contact information
        contacts, _ = self._check_contacts()
        contacts = np.array(contacts, dtype=np.float32)
        
        # Relative positions (cube to target, end effectors to cube)
        cube_to_target = target_pos - cube_pos
        ee_to_cube = []
        for ee_pos in self._get_all_End_Effector_pos():
            ee_to_cube.extend(cube_pos - ee_pos)

        cube_to_target_ori = target_ori - cube_ori
        
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
            cube_to_target_ori
        ])
        
        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        scaled_action = action * np.pi / 3
        self.do_simulation(scaled_action, self.frame_skip)
        obs = self._get_obs()
        reward, terminated = self._compute_reward()
        truncated = self.current_step >= self.max_episode_steps
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        noise = self.np_random.uniform(-0.02, 0.02, size=self.model.nq)
        self.set_state(self.init_qpos + noise, self.init_qvel * 0)

        # Set cube to fixed initial position
        cube_jnt_addr = self.model.body_jntadr[self._cube_id]
        if cube_jnt_addr >= 0:  # Check if the body has joints
            self.data.qpos[cube_jnt_addr:cube_jnt_addr + 3] = self._cube_initial_pos
        
        # Reset target position
        # target_y = self.np_random.uniform(-0.04, 0.04)
        # target_pos = np.array([0.0195, -0.04, -0.14])
        # self.model.body_pos[self._target_id] = target_pos

        # Reset target orientation
        target_ori = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        self.model.body_quat[self._target_id] = target_ori
        
        # Forward the simulation to apply changes
        cube_jnt = self.model.body_jntadr[self._cube_id]
        if cube_jnt >= 0:
            self.data.qpos[cube_jnt:cube_jnt+3] = self._cube_initial_pos

        # Randomize target y
        y_min, y_max = self.config['target_random_y_range']
        rand_y = self.np_random.uniform(y_min, y_max)
        self.model.body_pos[self._target_id] = np.array([
            self._target_pos[0], rand_y, self._target_pos[2]
        ], dtype=np.float32)

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
    
    def _compute_reward(self):
        cfg = self.config['reward']
        ee_pos = self._get_all_End_Effector_pos()
        cube_pos = self._get_cube_pos()
        target_pos = self._get_target_pos()
        cube_ori = self._get_cube_orientation()
        target_ori = self._get_target_orientation()
        contacts, table_contact = self._check_contacts()
        
        # # === PHASE 1: APPROACH PHASE ===
        # # Encourage gripper to approach cube
        # ee_cube_distances = [np.linalg.norm(ee_pos - cube_pos) for ee_pos in end_effector_pos]
        # avg_ee_cube_dist = np.mean(ee_cube_distances)
        
        # # Dense approach reward (decreases as gripper gets closer)
        # approach_reward = -avg_ee_cube_dist * 5.0
        
        # # === PHASE 2: CONTACT PHASE ===
        # num_contacts = sum(contacts)
        
        # # Progressive contact rewards - encourage all 3 contacts
        # if num_contacts == 0:
        #     contact_reward = 0.0
        # elif num_contacts == 1:
        #     contact_reward = 20.0  # First contact bonus
        # elif num_contacts == 2:
        #     contact_reward = 50.0  # Two-finger grasp bonus
        # elif num_contacts == 3:
        #     contact_reward = 100.0  # Perfect three-finger grasp bonus
        
        # # === PHASE 3: MANIPULATION PHASE ===
        # cube_target_dist = np.linalg.norm(cube_pos - target_pos)
        
        # manipulation_reward = -cube_target_dist * 100.0
        
        # # === PHASE 4: SUCCESS PHASE ===
        # success_bonus = 0.0
        # terminated = False
        
        # # Success criteria: cube very close to target with stable grasp
        # if cube_target_dist < 0.002 and num_contacts >= 2:
        #     success_bonus = 500.0
        #     terminated = True
        # elif cube_target_dist < 0.005 and num_contacts >= 2:
        #     # Close to success bonus
        #     success_bonus = 50.0
        
        # # === PENALTIES ===
        # # Time penalty to encourage efficiency
        # time_penalty = -0.5
        
        # # Total reward
        # reward = (approach_reward + 
        #           contact_reward + 
        #           manipulation_reward +
        #           success_bonus + 
        #           time_penalty)

        ###OLD REWARD FUNCTION###
        # cube_target_dist = np.linalg.norm(cube_pos - target_pos)
        # if cube_target_dist < 0.002 and sum(contacts) ==3:
        #     success_bonus = 50.0
        #     terminated = True
        # else: 
        #     success_bonus = 0.0
        #     terminated = False

        # reward = -cube_target_dist - (3-sum(contacts)) * 0.05 + success_bonus

        ### Orientation Old Reward Function ###

        ## Encourage gripper to approach cube
        ee_cube_distances = [np.linalg.norm(ee_pos - cube_pos) for ee_pos in end_effector_pos]
        avg_ee_cube_dist = np.mean(ee_cube_distances)
        
        # Dense approach reward (decreases as gripper gets closer)
        approach_reward = -avg_ee_cube_dist

        cube_target_ori = np.linalg.norm(cube_ori - target_ori)
        
        terminated = False
        success_bonus = 0.0

        if (cube_target_ori < 0.1 and sum(contacts) == 3 and table_contact == False):
            success_bonus = 10
            terminated = False # Set to this to disable success based termination

        if (table_contact == True):
            table_penalty = -5.0
        else :
            table_penalty = 0.0

        reward = approach_reward - cube_target_ori - (3 - sum(contacts)) * 0.05 + success_bonus + table_penalty

        return reward, terminated
    
    def _get_info(self):
        """Return diagnostic information."""
        contacts, _ = self._check_contacts()
        cube_pos = self._get_cube_pos()
        target_pos = self._get_target_pos()
        target_ori = self._get_target_orientation()
        cube_ori = self._get_cube_orientation()
        
        return {
            'cube_target_distance': np.linalg.norm(cube_pos - target_pos),
            'cube_target_orientation': np.linalg.norm(cube_ori - target_ori),
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
