import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset(seed=42)
for _ in range(500):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()

import gymnasium_robotics                     
hand = gym.make("HandManipulateBlock-v1", render_mode="human")
hand.reset()
for _ in range(1000):
    hand.step(hand.action_space.sample())
hand.close()
