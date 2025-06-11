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
        contacts = np.array(self._check_contacts(), dtype=np.float32)

        cube_to_target = target_pos - cube_pos
        ee_to_cube = np.concatenate([cube_pos - pos for pos in self._get_all_End_Effector_pos()])

        obs = np.concatenate([joint_pos, joint_vel, ee_positions,
                              cube_pos, target_pos, cube_to_target,
                              ee_to_cube, contacts])
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

    def _check_contacts(self):
        contacts = [False]*3
        for con in self.data.contact[:self.data.ncon]:
            g1, g2 = con.geom1, con.geom2
            for idx, tip in enumerate(self._end_effector_geom_ids):
                if (g1 == self._cube_geom_id and g2 == tip) or (g2 == self._cube_geom_id and g1 == tip):
                    contacts[idx] = True
        return contacts

    def _compute_reward(self):
        cfg = self.config['reward']
        ee_pos = self._get_all_End_Effector_pos()
        cube_pos = self._get_cube_pos()
        target_pos = self._get_target_pos()
        contacts = self._check_contacts()

        # Approach
        dists = [np.linalg.norm(p - cube_pos) for p in ee_pos]
        approach = -np.mean(dists) * cfg['approach_mul']

        # Contact
        n = sum(contacts)
        contact = cfg['contact_rewards'].get(n, 0.0)

        # Manipulation
        manip = -np.linalg.norm(cube_pos - target_pos) * cfg['manipulation_mul']
        if n >= 3:
            grasp = cfg['grasp_bonus_three']
        elif n >= 2:
            grasp = cfg['grasp_bonus_two']
        else:
            grasp = 0.0

        # Success
        dist = np.linalg.norm(cube_pos - target_pos)
        if dist < cfg['success_strict_dist'] and n >= 2:
            success_r, terminate = cfg['success_strict_reward'], True
        elif dist < cfg['success_loose_dist'] and n >= 2:
            success_r, terminate = cfg['success_loose_reward'], False
        else:
            success_r, terminate = 0.0, False

        total = approach + contact + manip + grasp + success_r + cfg['time_penalty']
        return total, terminate

    def _get_info(self):
        contacts = self._check_contacts()
        return {
            'cube_target_distance': np.linalg.norm(self._get_cube_pos() - self._get_target_pos()),
            'num_contacts': sum(contacts),
            'contacts': contacts,
            'cube_position': self._get_cube_pos(),
            'target_position': self._get_target_pos(),
            'episode_step': self.current_step,
            'max_episode_steps': self.max_episode_steps,
            'seed': self._seed
        }

    # Debug helpers
    def print_End_Effector_pos(self):  print(self._get_all_End_Effector_pos())
    def print_cube_pos(self):          print(self._get_cube_pos())
    def print_target_pos(self):        print(self._get_target_pos())
