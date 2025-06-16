import time
import numpy as np

from stable_baselines3 import SAC
from envs.d_pali_hand_new import DPALI_Hand

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium_robotics.core import GoalEnv  

def make_goal_env(xml=None, render_mode="human"):
    class DPALI_GoalEnv(DPALI_Hand, GoalEnv):
        def __init__(self, **kwargs):
            super().__init__(xml=xml, **kwargs)
            self.goal = np.array([0.1, 0.0, 0.15], np.float32)
            obs_dim = self._get_obs().shape[0]
            self.observation_space = spaces.Dict({
                'observation': spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
                'achieved_goal': spaces.Box(-np.inf, np.inf, (3,), np.float32),
                'desired_goal': spaces.Box(-np.inf, np.inf, (3,), np.float32),
            })

        def compute_reward(self, achieved_goal, desired_goal, info):
            diff = achieved_goal - desired_goal
            return -np.linalg.norm(diff, axis=-1).astype(np.float32)

        def _get_obs_dict(self):
            obs = self._get_obs()
            achieved = self.data.geom_xpos[self._cube_id].copy().astype(np.float32)
            return {
                'observation': obs,
                'achieved_goal': achieved,
                'desired_goal': self.goal
            }

        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            return self._get_obs_dict(), info

        def step(self, action):
            obs, _, done, truncated, info = super().step(action)
            reward = self.compute_reward(
                self.data.geom_xpos[self._cube_id], self.goal, info
            )
            return self._get_obs_dict(), reward, done, truncated, info

    try:
        register(
            id="DPALIHandGoal-v0",
            entry_point=DPALI_GoalEnv,
            max_episode_steps=500
        )
    except gym.error.Error:
        pass

    return gym.make("DPALIHandGoal-v0", render_mode=render_mode)

if __name__ == "__main__":
    env = make_goal_env(render_mode="human")

    model = SAC.load("models/sac_her_dpali_hand_V0.1", env=env)

    obs, _ = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        env.render()      
        time.sleep(0.01)  

        if done or truncated:
            obs, _ = env.reset()
