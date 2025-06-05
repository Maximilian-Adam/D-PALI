from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import warnings
from glfw import GLFWError
import time
import gymnasium as gym
import envs  # registers DPALIHand-v0
import mujoco
import glfw
import torch
import os
import numpy as np

warnings.simplefilter("error", GLFWError)

global_eval_freq = 100000
global_max_episode_steps = 500
global_save_freq = 100000
global_reward_threshold = 7500.0



def setup(mode="test", log_dir=None, max_episode_steps=500):
    """Set up the environment with optional monitoring."""
    _render_mode = "human" if mode == "test" else None
    frame_skip = 5 if mode == "test" else 20  # Double frame skip for training
    env = gym.make("DPALIHand-v0", 
                   render_mode=_render_mode, 
                   max_episode_steps=max_episode_steps,
                   frame_skip=frame_skip)
    
    if log_dir and mode == "train":
        env = Monitor(env, log_dir)
    
    obs, _ = env.reset()
    return env

def training_td3(total_timesteps, file_path, log_dir="./training/logs/", eval_freq = global_eval_freq):
    """Train using TD3 algorithm - excellent for continuous control tasks."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create training environment
    env = setup("train", log_dir + "monitor/", max_episode_steps = global_max_episode_steps)

    # Create evaluation environment (separate from training)
    eval_env = setup("train", max_episode_steps = global_max_episode_steps)

    # Add action noise for better exploration during training
    # TD3 uses noise for exploration - this is crucial for learning
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.1 * np.ones(n_actions)
    )
    
    # TD3 hyperparameters optimized for manipulation tasks
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=1e-3,           # Learning rate - can be higher for TD3
        buffer_size=1000000,          # Large replay buffer for better sample efficiency  
        learning_starts=10000,        # Start learning after collecting some experience
        batch_size=2048,               # Batch size for training
        tau=0.005,                    # Soft update coefficient for target networks
        gamma=0.99,                   # Discount factor
        train_freq=(4, "step"),       # Train after every 4 steps
        gradient_steps=4,           # Do as many gradient steps as environment steps
        action_noise=action_noise,    # Exploration noise
        policy_delay=2,               # Delay policy updates (key TD3 feature)
        target_policy_noise=0.2,      # Noise added to target policy
        target_noise_clip=0.5,        # Clip target noise
        policy_kwargs=dict(
            net_arch=[512, 512, 256],      # Network architecture [actor, critic]
            activation_fn=torch.nn.ReLU
        ),
        verbose=1,
        tensorboard_log=log_dir + "td3_tensorboard/",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Callbacks for training
    # Stop training if we achieve good performance
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold = global_reward_threshold, 
        verbose=1
    )
    
    # # Evaluate the model periodically and save the best one
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(file_path) + "/best_model/",
        log_path=log_dir + "eval_logs/",
        eval_freq=eval_freq,
        n_eval_episodes=10,           # Number of episodes for evaluation
        deterministic=True,           # Use deterministic actions for evaluation
        render=False,
        callback_on_new_best=stop_callback,
        verbose=1
    )
    
    # Save checkpoints during training
    checkpoint_callback = CheckpointCallback(
        save_freq = global_save_freq,
        save_path=os.path.dirname(file_path) + "/checkpoints/",
        name_prefix="td3_checkpoint"
    )
    
    # Combine callbacks
    callbacks = [checkpoint_callback, eval_callback]
    
    print('*************TD3 Training started*************')
    print(f"Device: {model.device}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Evaluation frequency: {eval_freq:,}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Start training
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )
    
    # Save final model
    model.save(file_path)
    
    # Clean up
    env.close()
    eval_env.close()
    
    print('*************TD3 Training finished*************')
    print(f"Model saved to: {file_path}")
    print(f"Best model saved to: {os.path.dirname(file_path)}/best_model/")

def continue_training_td3(model_path, total_timesteps, save_path, log_dir="./training/logs/"):
    """Continue training from a saved model."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up environments
    env = setup("train", log_dir + "monitor/", max_episode_steps = global_max_episode_steps)
    eval_env = setup("train", max_episode_steps = global_max_episode_steps)
    
    # Load the saved model
    model = TD3.load(model_path, env=env)
    
    # Set up callbacks
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold = global_reward_threshold, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(save_path) + "/best_model/",
        log_path=log_dir + "eval_logs/",
        eval_freq = global_eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_on_new_best=stop_callback,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq = global_save_freq,
        save_path=os.path.dirname(save_path) + "/checkpoints/",
        name_prefix="td3_checkpoint_continued"
    )
    
    print('*************Continuing TD3 Training*************')
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=False  # Don't reset timestep counter
    )
    
    model.save(save_path)
    env.close()
    eval_env.close()
    print('*************Training continuation finished*************')

def testing_td3(file_path, num_episodes=10, max_episode_steps = global_max_episode_steps):
    """Test a trained TD3 model."""
    env = setup("test", max_episode_steps=max_episode_steps)
    
    # Load the trained model
    model = TD3.load(file_path, env=env)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"Testing TD3 model: {file_path}")
    print(f"Running {num_episodes} episodes...")
    
    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            step_count = 0
            episode_info = []
            
            print(f"\n--- Episode {episode + 1} ---")
            
            while True:
                # Use deterministic actions for testing
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                step_count += 1
                episode_info.append(info)
                
                # Print progress every 50 steps
                if step_count % 50 == 0:
                    print(f"  Step {step_count}: Reward: {reward:.3f}, "
                          f"Cube-Target Dist: {info.get('cube_target_distance', 'N/A'):.4f}, "
                          f"Contacts: {info.get('num_contacts', 'N/A')}")
                
                env.render()
                time.sleep(1/60)  # Control rendering speed
                
                done = terminated or truncated
                if done:
                    break
            
            # Episode summary
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            
            if terminated:  # Task completed successfully
                success_count += 1
                print(f"  ✓ SUCCESS! Task completed in {step_count} steps")
            else:
                print(f"  ✗ Episode ended by timeout after {step_count} steps")
            
            print(f"  Episode Reward: {episode_reward:.2f}")
            
            # Show final state info
            final_info = episode_info[-1]
            print(f"  Final cube-target distance: {final_info.get('cube_target_distance', 'N/A'):.4f}")
            print(f"  Final contacts: {final_info.get('num_contacts', 'N/A')}")
            
            time.sleep(1)  # Pause between episodes
            
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    except GLFWError:
        print("\nGLFW Error occurred")
    finally:
        env.close()
        
        # Print testing summary
        if episode_rewards:
            print(f"\n{'='*50}")
            print("TESTING SUMMARY")
            print(f"{'='*50}")
            print(f"Episodes completed: {len(episode_rewards)}")
            print(f"Success rate: {success_count}/{len(episode_rewards)} ({100*success_count/len(episode_rewards):.1f}%)")
            print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            print(f"Best episode reward: {max(episode_rewards):.2f}")
            print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
            print(f"{'='*50}")

def hyperparameter_search():
    """Run a simple hyperparameter search for TD3."""
    hyperparams = [
        {"lr": 1e-3, "buffer_size": 1000000, "batch_size": 256, "learning_starts": 25000},
        {"lr": 3e-4, "buffer_size": 500000, "batch_size": 128, "learning_starts": 10000},
        {"lr": 1e-4, "buffer_size": 1000000, "batch_size": 512, "learning_starts": 50000},
    ]
    
    for i, params in enumerate(hyperparams):
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER SEARCH - Configuration {i+1}/{len(hyperparams)}")
        print(f"Parameters: {params}")
        print(f"{'='*60}")
        
        file_path = f"./training/hyperparameter_search/td3_config_{i+1}"
        training_td3(
            total_timesteps=200000,  # Shorter training for search
            file_path=file_path,
            log_dir=f"./training/hyperparameter_search/logs_config_{i+1}/"
        )

if __name__ == "__main__":
    mode = "train"  # "train", "test", "continue", or "hypersearch"
    
    # Configuration
    total_timesteps = 500000 
    file_path = "./training/checkpoints/TD3/td3_DPALIHand-v5.0"
    
    if mode == "train":
        training_td3(total_timesteps, file_path)
        
    elif mode == "test":
        testing_td3(file_path, num_episodes=5)
        
    elif mode == "continue":
        # Continue training from existing model
        existing_model = "./training/checkpoints/TD3/td3_DPALIHand-v4.0"
        new_save_path = "./training/checkpoints/TD3/td3_DPALIHand-v4.1"
        continue_training_td3(existing_model, 400000, new_save_path)
        
    elif mode == "hypersearch":
        hyperparameter_search()
        
    else:
        print("Invalid mode. Use 'train', 'test', 'continue', or 'hypersearch'.")
        exit(1)