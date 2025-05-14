# scripts/test2.py
import time
import gymnasium as gym, envs     # registers DPALIHand-v0
from glfw import GLFWError       

env = gym.make("DPALIHand-v0", render_mode="human")
obs, _ = env.reset(seed=0)

for _ in range(200):
    action, _ = env.action_space.sample(), None
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()                 
    time.sleep(1/60)            
    if terminated or truncated:
        obs, _ = env.reset()

try:
    while True:
        env.render()
        time.sleep(1/60)
except GLFWError:
    pass
finally:
    env.close()
