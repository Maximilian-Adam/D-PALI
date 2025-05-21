'''
Implementation of Soft Actor-Critic (SAC) with Hindsight Experience Replay (HER)
for the DPALI hand environment.

 1. Wraps the DPALI_Hand as a GoalEnv for HER.
 2. Vectorizes and normalizes the environment.
 3. Defines an SAC agent with HER replay buffer.
 4. Runs training and saves the model.
'''
import numpy as np
from typing import Dict, Any

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium_robotics.core import GoalEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback

from envs.d_pali_hand import DPALI_Hand


def make_goal_env(xml: str = None, render_mode: str = None) -> gym.Env:
    class DPALI_GoalEnv(DPALI_Hand, GoalEnv):
        def __init__(self, **kwargs):
            super().__init__(xml=xml, **kwargs)
            # Desired goal is the target position in 3D
            self.goal = np.array([0.1, 0.0, 0.15], dtype=np.float32)
            # Observation space becomes a Dict space
            obs_dim = self._get_obs().shape[0]
            self.observation_space = spaces.Dict({
                'observation': spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
                'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                'desired_goal': spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            })

        def compute_reward(self,
                           achieved_goal: np.ndarray,
                           desired_goal: np.ndarray,
                           info) -> np.ndarray:
            """
            Compute a per-sample Euclidean reward.
            If inputs are shape (3,), returns a scalar; if (N,3), returns shape (N,).
            """
            diff = achieved_goal - desired_goal
            # axis=-1 handles both (3,) -> scalar and (N,3) -> (N,)
            reward = -np.linalg.norm(diff, axis=-1)
            return reward.astype(np.float32)

        def _get_obs_dict(self) -> dict:
            obs = self._get_obs()
            # achieved goal: position of object0 fetched via Mujoco data
            geom_id = self._obj_id
            achieved = self.data.geom_xpos[geom_id].copy().astype(np.float32)
            return {'observation': obs, 'achieved_goal': achieved, 'desired_goal': self.goal}

        def reset(self, **kwargs):
            _obs, info = super().reset(**kwargs)
            return self._get_obs_dict(), info

        def step(self, action):
            obs, _, terminated, truncated, info = super().step(action)
            return self._get_obs_dict(), self.compute_reward(self.data.geom_xpos[self._obj_id],
                                                               self.goal, info), terminated, truncated, info

    try:
        register(
            id='DPALIHandGoal-v0',
            entry_point=DPALI_GoalEnv,
            max_episode_steps=500
        )
    except Exception:
        pass

    return gym.make('DPALIHandGoal-v0', render_mode=render_mode)


if __name__ == '__main__':

    vec_env = DummyVecEnv([lambda: make_goal_env(render_mode=None)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model = SAC(
        policy='MultiInputPolicy',
        env=vec_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            'n_sampled_goal': 4,
            'goal_selection_strategy': 'future',
        },
        verbose=1,
        learning_starts=4000,
        tensorboard_log='./tensorboard_dpali'
    )

    eval_env = DummyVecEnv([lambda: make_goal_env(render_mode=None)])

    eval_env = VecNormalize(
        eval_env,
       norm_obs=True,
       norm_reward=True,
       training=False,
    )

    eval_env.obs_rms = vec_env.obs_rms
    eval_env.ret_rms = vec_env.ret_rms

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./best_model',
        log_path='./eval_logs',
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
    )

    total_timesteps = 1_000_000
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model.save('models/sac_her_dpali_hand_V0.1')

    print("Training completed and model saved!")
