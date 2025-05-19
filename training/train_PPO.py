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
def training(_total_timesteps):
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
    model.save("./training/checkpoints/ppo_DPALIHand-v0")
    env.close()
    print('*************Training finished*************')

# *********************TESTING*********************
def testing():
    env = setup("test")
    obs, _ = env.reset()
    model = PPO.load("./training/checkpoints/ppo_DPALIHand-v0", env=env)
    for _ in range(100000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Reward: {reward}, Done: {done}")
        env.render()
        if done:
            print('Done')
            time.sleep(1/2)
            obs, _ = env.reset()
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
    _total_timesteps = 1000000  # Adjust this value as needed

    if mode == "train":
        training(_total_timesteps)
    elif mode == "test":
        testing()
    else:
        print("Invalid mode. Use 'train' or 'test'.")
        exit(1)

    