from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from envs import DPALI_Hand
import warnings
from glfw import GLFWError
import time
import gymnasium as gym
import envs # registers DPALIHand-v0
import mujoco
import glfw
import torch
import os
import numpy as np
from typing import Callable
#from math import cos, pi
from callbacks import TensorboardCallback  # Custom callbacks for TensorBoard logging
import traceback

###Learning rate, success based?, hyperparameters, balance reward, normalise observations

warnings.simplefilter("error", GLFWError)

"""Global configuration parameters for training and evaluation."""
global_mode = "train"
global_total_timesteps = 1500000 # Total timesteps for training
global_eval_freq = 50000 # Frequency of evaluation during training (in steps)
global_max_episode_steps = 500 # Maximum steps per episode during training
global_save_freq = 500000 # Frequency of saving model checkpoints (in steps)
global_reward_threshold = 2500.0 # Reward threshold for stopping training


global_initial_lr = 3e-4 
global_final_lr = 1e-5  
global_folder = "Ori_V2.1"# Name of folder for saving models (Increment when training from scratch)
global_version = "v1.0" # Sub-version for tracking changes (increment when you use continue training)

global_save_dir = f"./training/TD3/{global_folder}/" # Directory to save models
global_best_model_path = os.path.join(global_save_dir, "best_model/best_model.zip")
global_stats_path = os.path.join(global_save_dir, global_version, "vec_normalize.pkl")
global_old_model_dir = f"./training/TD3/{global_folder}/best_model/best_model.zip"
global_old_stats_path = f"./training/TD3/{global_folder}/{global_version}/vec_normalize.pkl" # Make sure this path is correct for your old stats



def setup(mode="train", log_dir=None, max_episode_steps=500):
    """Set up the environment with optional monitoring."""
    _render_mode = "human" if mode == "test" else None
    frame_skip = 5 if mode == "test" else frame_skip = 10  #20 # Double frame skip for training
    env = gym.make("DPALIHand-v0", 
                   render_mode=_render_mode, 
                   max_episode_steps=max_episode_steps,
                   frame_skip=frame_skip,
                   seed=20)
    
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
            norm_reward=False,  # Don't normalize rewards during eval
            clip_obs=10.0,
            gamma=0.99,
            epsilon=1e-8
        )
    elif (mode == "eval_continue"):
        stats_dir = global_stats_path
        env = VecNormalize.load(stats_dir, env)
        env.training = False 
        env.norm_reward = False

   
    
    return env


def lr_schedule(initial_value: float, final_value : float) -> Callable[[float], float]:

    def func(progress_remaining: float) -> float:
        t = 1 - progress_remaining
        output = final_value + (initial_value - final_value) * 0.5 * (1 + np.cos(np.pi * t))
        return output

    return func

def training_td3(total_timesteps, save_dir, log_dir="./training/logs/", eval_freq = global_eval_freq):
    """Train using TD3 algorithm - excellent for continuous control tasks."""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create training environment
    env = setup("train", log_dir + "monitor/", max_episode_steps = global_max_episode_steps)

    # Create evaluation environment (separate from training)
    eval_env = setup("eval_train", max_episode_steps = global_max_episode_steps)

    # Add action noise for better exploration during training
    # TD3 uses noise for exploration - this is crucial for learning
    n_actions = env.action_space.shape[0]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.15 * np.ones(n_actions),    
        theta=0.15,                         
        dt=1e-2                              
    )
    
    # TD3 hyperparameters optimized for manipulation tasks

    optimized_hyperparams = {
    'learning_rate': 0.0002846710930711627,
    'batch_size': 512,
    'buffer_size': 1000000,
    'tau': 0.005,
    'gamma': 0.98,
    'policy_kwargs': {
        'net_arch': {
            'pi': [512, 512, 256],
            'qf': [1024, 64]
        }
    }
}
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=lr_schedule(global_initial_lr, global_final_lr),    # Learning rate 
        buffer_size=int(optimized_hyperparams['buffer_size']),          # Large replay buffer for better sample efficiency  
        learning_starts=10000,        # Start learning after collecting some experience
        batch_size=optimized_hyperparams['batch_size'],               # Batch size for training
        tau=optimized_hyperparams['tau'],
        gamma=optimized_hyperparams['gamma'],                   # Discount factor
        train_freq=(4, "step"),       # Train after every 4 steps
        gradient_steps=4,             # Do as many gradient steps as environment steps
        action_noise=action_noise,    # Exploration noise
        policy_delay=2,               # Delay policy updates (key TD3 feature)
        target_policy_noise=0.2,      # Noise added to target policy
        target_noise_clip=0.5,        # Clip target noise
        policy_kwargs=dict(
        #net_arch=optimized_hyperparams['policy_kwargs']['net_arch'],
        net_arch=[256, 256, 256],
        activation_fn=torch.nn.ReLU
        ),
        verbose=0,
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
        best_model_save_path=os.path.join(save_dir, "best_model/"),
        log_path=log_dir + "eval_logs/",
        eval_freq=eval_freq,
        n_eval_episodes=10,           # Number of episodes for evaluation
        deterministic=True,           # Use deterministic actions for evaluation
        render=False,
        callback_on_new_best=stop_callback,
        verbose=0
    )
    
    # Save checkpoints during training
    checkpoint_callback = CheckpointCallback(
        save_freq = global_save_freq,
        save_path=os.path.join(save_dir, "checkpoints/"),
        name_prefix="td3_checkpoint"
    )

    #custom_callback = TensorboardCallback(
    #    verbose=0
    #)
    
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
        callback=callbacks,
        tb_log_name=f"{global_folder}_{global_version}"
    )
    
    # Save final model
    model.save(os.path.join(save_dir, "final_model.zip"))
    env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    
    # Clean up
    env.close()
    eval_env.close()
    
    print('*************TD3 Training finished*************')
    print(f"Model saved to: {save_dir}")

def continue_training_td3(model_path, total_timesteps, save_dir, log_dir="./training/logs/"):
    """Continue training from a saved model."""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up environments
    env = setup("continue", log_dir + "monitor/", max_episode_steps = global_max_episode_steps)
    eval_env = setup("eval_continue", max_episode_steps = global_max_episode_steps)
    
    # Load the saved model
    model = TD3.load(model_path, env=env)

    # Gonna keep this out for now
    # new_lr = 0.0007593145723955295 / 2; # Static learning rate
    # model.learning_rate = new_lr  # Update stored value
    # for param_group in model.actor.optimizer.param_groups:
    #     param_group['lr'] = new_lr  # Update actual optimizer
    # for param_group in model.critic.optimizer.param_groups:
    #     param_group['lr'] = new_lr  # Update actual optimizer
    
    # Set up callbacks
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold = global_reward_threshold, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best_model/"),
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
        save_path=os.path.join(save_dir, "checkpoints/"),
        name_prefix="td3_checkpoint_continued"
    )

    custom_callback = TensorboardCallback(
        verbose=1
    )
    
    print('*************Continuing TD3 Training*************')
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, custom_callback],
        reset_num_timesteps=False,  # Don't reset timestep counter
        tb_log_name=global_folder + "_" + global_version  # TensorBoard log name
    )
    
    model.save(os.path.join(save_dir, "final_model.zip"))
    env.save(os.path.join(save_dir, "vec_normalize.pkl")) # Save VecNormalize stats
    env.close()
    eval_env.close()
    print('*************Training continuation finished*************')

def testing_td3(model_path, num_episodes=10, max_episode_steps = global_max_episode_steps):
    
    """Test a trained TD3 model."""
    try:
        env = setup("test", max_episode_steps=max_episode_steps)
 
        # Load the trained model
        # model = TD3.load(model_path, env=env)

        #If model is messed up use this:

        safe_lr_schedule = 0.0001 

        # Create a dictionary to override the problematic component.
        custom_objects = {
            "learning_rate": safe_lr_schedule
        }

        # Pass the custom_objects dictionary to the load function.
        model = TD3.load(
            model_path,
            env=env,
            custom_objects=custom_objects
        )
    
    except Exception as e:
        # This is the crucial part. It will catch ANY error and print it.
        print(f"\n\n--- AN ERROR OCCURRED ---")
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"ERROR MESSAGE: {e}")
        print("\n--- TRACEBACK ---")
        traceback.print_exc() # Prints the full, detailed error traceback
        print("--------------------")

    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"Testing TD3 model: {model_path}")
    print(f"Running {num_episodes} episodes...")
    
    try:
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            step_count = 0
            episode_info = []
            
            print(f"\n--- Episode {episode + 1} ---")
            
            while True:
                # Use deterministic actions for testing
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                info = info[0] # Since there's only one evironment
                reward = reward[0]
                done = done[0]
                episode_reward += reward
                step_count += 1
                episode_info.append(info)
                
                # Print progress every 50 steps
                if step_count % 50 == 0:
                    print(f"  Step {step_count}: Reward: {reward:.3f}, "
                          f"Cube-Target Orientation: {info['cube_target_orientation']:.4f}, "
                          f"Contacts: {info['num_contacts']}")
                
                env.render()
                time.sleep(1/60)  # Control rendering speed
                
                if done:
                    break
            
            # Episode summary
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            
            if done and step_count < max_episode_steps:
                success_count += 1
                print(f"  ✓ SUCCESS! Task completed in {step_count} steps")
            else:
                print(f"  ✗ Episode ended by timeout after {step_count} steps")

            print(f"  Episode Reward: {episode_reward:.2f}")
            
            # Show final state info
            final_info = episode_info[-1]
            print(f"  Final cube-target distance: {final_info.get('cube_target_distance', 'N/A'):.4f}")
            #print(f"  Final contacts: {final_info.get('num_contacts', 'N/A')}")
            #env.unwrapped.print_End_Effector_pos()

            print(f"  Final cube-target orientation: {final_info['cube_target_orientation']:.4f}")
            print(f"  Final contacts: {final_info['num_contacts']}")
            
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


if __name__ == "__main__":
    mode = global_mode  # "train", "test", "continue", or "hypersearch"
    
    # Configuration
    total_timesteps = global_total_timesteps
    
    if mode == "train":
        training_td3(global_total_timesteps, global_save_dir)
        
    elif mode == "test":
        model_to_test = global_best_model_path
        stats_to_use = global_stats_path

        # --- CRUCIAL: Verify files exist before trying to load them ---
        if not os.path.exists(model_to_test):
            print(f"\nFATAL: Model file not found at '{model_to_test}'")
            print("Please ensure that training has been run and a 'best_model.zip' file was saved in the correct directory.")
            exit(1)
        
        if not os.path.exists(stats_to_use):
            print(f"\nFATAL: Normalization stats file not found at '{stats_to_use}'")
            print("This 'vec_normalize.pkl' file is required to run the test with the correct environment normalization.")
            exit(1)

        testing_td3(model_to_test, num_episodes=5)
        
    elif mode == "continue":
        continue_training_td3(global_old_model_dir, global_total_timesteps, global_save_dir)
    elif mode == "hypersearch":
        hyperparameter_search()
        
    else:
        print("Invalid mode. Use 'train', 'test', 'continue', or 'hypersearch'.")
        exit(1)