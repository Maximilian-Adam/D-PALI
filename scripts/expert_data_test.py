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

# Import the setup function from your training script to ensure consistency
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def replay_expert_data():
    """
    Loads the expert trajectory and replays it in the environment for visual verification.
    """
    print("--- Starting Expert Data Replay ---")
    
    # --- Load Expert Data ---
    save_dir = "expert_data"
    obs_file = os.path.join(save_dir, "expert_observations.npy")
    action_file = os.path.join(save_dir, "expert_actions.npy")
    
    if not os.path.exists(action_file):
        print(f"ERROR: Action file not found at '{action_file}'. Please generate the data first.")
        return
        
    expert_actions = np.load(action_file)
    num_steps = len(expert_actions)
    print(f"Loaded {num_steps} expert actions for replay.")

    # --- Setup Environment ---
    # Use the 'test' mode from your setup function to enable rendering
    env = setup(mode="test", max_episode_steps=num_steps + 1)
    
    # Reset the environment
    obs = env.reset()
    
    # --- Replay Loop ---
    print("Starting visual replay... Press Ctrl+C in the terminal to stop.")
    try:
        for i in range(num_steps):
            action = expert_actions[i]
            
            # Apply the expert action to the environment
            # Action must be wrapped in a list/array for the VecEnv
            obs, reward, done, info = env.step(np.array([action]))
            
            # Render the environment
            env.render()
            
            # Control replay speed
            time.sleep(1/60)
            
            if done:
                print(f"Episode ended prematurely at step {i}.")
                break
    except KeyboardInterrupt:
        print("\nReplay stopped by user.")
    finally:
        # Clean up the environment
        env.close()
        print("--- Replay Finished ---")


if __name__ == "__main__":
    replay_expert_data()