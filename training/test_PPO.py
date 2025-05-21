from stable_baselines3 import PPO
import warnings
from glfw import GLFWError
import time
import gymnasium as gym, envs     # registers DPALIHand-v0
import mujoco  
import glfw
import torch
warnings.simplefilter("error", GLFWError)

env = gym.make("DPALIHand-v0", render_mode="human")
obs, _ = env.reset(seed=0)
print(env.unwrapped.print_obs_dim())


obs, _ = env.reset()
model = PPO.load("./training/checkpoints/ppo_DPALIHand-v0.1", env=env)
for _ in range(100000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    if done:
        print('Done')
        time.sleep(1/60)
        obs, _ = env.reset()


try:
    while True:
        env.render()
        time.sleep(1/60)
except GLFWError:
    pass
finally:
    env.close()