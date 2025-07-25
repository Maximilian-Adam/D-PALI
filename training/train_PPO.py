from stable_baselines3 import PPO
import warnings
from glfw import GLFWError
import time
import gymnasium as gym, envs     # registers DPALIHand-v0
import mujoco  
import glfw
import torch
warnings.simplefilter("error", GLFWError)

def setup(mode="test"):
    # Set up the environment
    _render_mode = "human" if mode == "test" else None
    env = gym.make("DPALIHand-v0", render_mode=_render_mode)
    obs, _ = env.reset()
    return env


# *********************TRAINING*********************
def training(_total_timesteps,file_path):
    env = setup("train")
    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 128,64,32], vf=[256, 256, 128, 64])],
        activation_fn=torch.nn.ReLU
    )

    obs, _ = env.reset()
    model = PPO("MlpPolicy", 
                env, 
                learning_rate=3e-3,
                verbose=1, 
                tensorboard_log="./training/logs/ppo_logs/",
                device="cuda", 
                policy_kwargs=policy_kwargs)
    print('*************Training started*************')
    model.learn(total_timesteps=_total_timesteps)
    model.save(file_path)
    env.close()
    print('*************Training finished*************')

# model = PPO.load("./training/checkpoints/ppo_DPALIHand-v0.1", env=env)
# model.learn(total_timesteps=10000000)
# model.save("./training/checkpoints/ppo_DPALIHand-v0.1")
# *********************TESTING*********************
def testing(file_path):
    env = setup("test")
    obs, _ = env.reset()
    model = PPO.load(file_path, env=env)
    i = 0
    while True:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Reward: {reward}, Done: {done}")
        env.render()
        if done or i == 1000:
            print('Done')
            time.sleep(1/2)
            obs, _ = env.reset()
            i = 0
            print('Resetting environment')
        else: i += 1
    try:
        while True:
            env.render()
            time.sleep(1/60)
    except GLFWError:
        pass
    finally:
        env.close()



if __name__ == "__main__":
    mode = "test"  # "train" or "test"
    _total_timesteps = 5000000  # Adjust this value as needed
    file_path = "./training/checkpoints/ppo_DPALIHand-v1.0"

    if mode == "train":
        training(_total_timesteps,file_path)
    elif mode == "test":
        testing(file_path)
    else:
        print("Invalid mode. Use 'train' or 'test'.")
        exit(1)

    