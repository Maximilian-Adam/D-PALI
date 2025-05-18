import warnings
from glfw import GLFWError
import time
import gymnasium as gym, envs     # registers DPALIHand-v0
import mujoco  
import glfw


warnings.simplefilter("error", GLFWError)

env = gym.make("DPALIHand-v0", render_mode="human")
obs, _ = env.reset(seed=0)

for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(1/60)
    # Debugging Example
    env.unwrapped.print_End_Effector_pos()
    env.unwrapped.print_cube_pos()

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
