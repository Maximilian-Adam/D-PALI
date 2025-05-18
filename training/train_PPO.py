from stable_baselines3 import PPO
import warnings
from glfw import GLFWError
import time
import gymnasium as gym, envs     # registers DPALIHand-v0
import mujoco  
import glfw

warnings.simplefilter("error", GLFWError)

env = gym.make("DPALIHand-v0", render_mode="human")
obs, _ = env.reset(seed=0)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./training/logs/ppo_logs/",device="cpu")
model.learn(total_timesteps=100000)
model.save("./training/checkpoints/ppo_DPALIHand-v0")

obs, _ = env.reset()
model = PPO.load("./training/checkpoints/ppo_DPALIHand-v0", env=env)
for _ in range(100000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(reward)
    env.render()
    if done:
        print('Done')
        obs, _ = env.reset()


try:
    while True:
        env.render()
        time.sleep(1/60)
except GLFWError:
    pass
finally:
    env.close()