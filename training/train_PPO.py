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

# *********************TRAINING*********************
policy_kwargs = dict(
    net_arch=[dict(pi=[128, 128], vf=[256, 256])],
    activation_fn=torch.nn.ReLU
)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./training/logs/ppo_logs/",device="cuda", policy_kwargs=policy_kwargs)
model.learn(total_timesteps=10000000)
model.save("./training/checkpoints/ppo_DPALIHand-v0.1")

# model = PPO.load("./training/checkpoints/ppo_DPALIHand-v0.1", env=env)
# model.learn(total_timesteps=10000000)
# model.save("./training/checkpoints/ppo_DPALIHand-v0.1")
# *********************TESTING*********************
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