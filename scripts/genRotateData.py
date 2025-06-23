import time
import numpy as np
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import envs

global_folder = "Ori_V3.1"# Name of folder for saving models (Increment when training from scratch)
global_version = "v1.0"

global_save_dir = f"./training/TD3/{global_folder}/" 
global_best_model_path = os.path.join(global_save_dir, "best_model/best_model.zip")
global_stats_path = os.path.join(global_save_dir, global_version, "vec_normalize.pkl")
global_old_model_dir = f"./training/TD3/{global_folder}/best_model/best_model.zip"
global_old_stats_path = f"./training/TD3/{global_folder}/{global_version}/vec_normalize.pkl"


# --- IMPORTANT: Import the setup function from your training script ---
# This assumes rotateTest.py is in a 'scripts' folder and train_TD3.py is in the parent directory.
# Adjust the import path if your file structure is different.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def setup(mode="train", log_dir=None, max_episode_steps=500):
    """Set up the environment with optional monitoring."""
    _render_mode = "human" if mode == "test" else None
    frame_skip = 5 if mode == "test" else 10  #20 # Double frame skip for training
    env = gym.make("DPALIHand-v0", 
                   render_mode=_render_mode, 
                   max_episode_steps=max_episode_steps,
                   frame_skip=frame_skip,
                   seed=None)
    
    if log_dir and (mode == "train" or mode == "continue"):
        env = Monitor(env, log_dir)
    
    env = DummyVecEnv([lambda: env])
    

    if (mode == "train"):
        env = VecNormalize(
            env,
            training=True,           # Update statistics during training
            norm_obs=True,           # Normalize observations
            norm_reward=True,        # Normalize rewards
            clip_obs=10.0,           # Clip normalized obs to ±10
            clip_reward=20.0,        # Clip normalized rewards
            gamma=0.975,              # For reward normalization
            epsilon=1e-8
        )
    elif (mode == "continue"):
        env = VecNormalize.load(global_old_stats_path, env)
        env.training = True
        env.norm_reward = True

    elif (mode == "test"):
        # Load normalization stats for testing
        if os.path.exists(global_stats_path):
            env = VecNormalize.load(global_stats_path, env)
            env.training = False
            env.norm_reward = False 
    elif (mode == "eval_train"):
        env = VecNormalize(
            env,
            training=False,  # Don't update stats during eval
            norm_obs=True,
            norm_reward=True,  # Don't normalize rewards during eval
            clip_obs=10.0,
            gamma=0.975,
            epsilon=1e-8
        )
    elif (mode == "eval_continue"):
        stats_dir = global_stats_path
        env = VecNormalize.load(stats_dir, env)
        env.training = False 
        env.norm_reward = False

   
    
    return env



def generate_expert_data():
    """
    Creates an instance of the DPALIHand-v0 environment, runs a scripted
    policy to generate expert trajectories, and saves them to .npy files.
    """
    print("--- Starting Expert Data Generation ---")
    
    # --- Use the EXACT same setup as your training environment ---
    # We use mode "eval_train" as a good, non-training-but-normalized option
    # or create a custom one if needed. Let's use "eval_train".
    env = setup(mode="eval_train", max_episode_steps=500)

    # Reset the environment to get a starting observation
    obs = env.reset()
    
    # Parameters
    num_actuators = env.get_attr('model')[0].nu
    base_joint_indices = [2, 5, 8]
    test_duration_steps = 500
    
    # Lists to store the trajectory data
    observations_data = []
    actions_data = []
    
    print(f"Generating {test_duration_steps} steps of expert data...")

    # Main loop to control the robot
    for step in range(test_duration_steps):
        # The observation from a VecEnv has shape (1, obs_dim).
        # We append the "squeezed" observation of shape (obs_dim,) to our list.
        observations_data.append(obs.squeeze())

        target_base_action = np.sin(2 * np.pi * step / test_duration_steps)
        action = np.zeros(num_actuators)
        action[base_joint_indices] = target_base_action
        
        # We also append the clean action vector
        actions_data.append(action)
        
        # Step the environment
        obs, reward, done, info = env.step(np.array([action])) # Action must be wrapped for VecEnv
        
        # Optional: Render to see what's happening
        # env.render() 
        # time.sleep(1/60)
            
    # Clean up the environment
    env.close()

    # --- Save the collected data ---
    # Create a directory to store the data if it doesn't exist
    save_dir = "expert_data"
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert lists to NumPy arrays before saving
    observations_array = np.array(observations_data)
    actions_array = np.array(actions_data)

    print(f"Shape of saved observations: {observations_array.shape}")
    print(f"Shape of saved actions: {actions_array.shape}")
    
    np.save(os.path.join(save_dir, "expert_observations.npy"), observations_array)
    np.save(os.path.join(save_dir, "expert_actions.npy"), actions_array)
    
    print(f"\n--- Expert data generation finished. Data saved in '{save_dir}' folder. ---")


if __name__ == "__main__":
    generate_expert_data()