import optuna
import numpy as np
from stable_baselines3 import TD3
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import envs


def setup_environment():
    """Setup your environment for hyperparameter search."""
    env = gym.make("DPALIHand-v0", render_mode=None, max_episode_steps=400)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    return env

def evaluate_model_performance(model, env):
    """Evaluate model performance with multiple episodes."""
    
    eval_env = gym.make("DPALIHand-v0", render_mode=None, max_episode_steps=400)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    total_reward = 0
    success_count = 0
    n_episodes = 10
    
    for episode in range(n_episodes):
        obs = eval_env.reset()
        episode_reward = 0
        success = False
        
        for step in range(400):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0]
            
            # Check for success (adjust based on your task)
            if info[0].get('cube_target_orientation', 1.0) < 0.1:
                success = True
            
            if done:
                break
        
        total_reward += episode_reward
        if success:
            success_count += 1
    
    eval_env.close()
    
    # Combined score: average reward + success bonus
    avg_reward = total_reward / n_episodes
    success_rate = success_count / n_episodes
    combined_score = avg_reward + success_rate * 100  # Bonus for success
    
    return combined_score

# QUICK START VERSION (Recommended)
def quick_gripper_search():
    """Simplified search focusing on most important parameters."""
    
    def objective(trial):
        # Focus on the big 4
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-4, log=True)
        batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])
        gamma = trial.suggest_float('gamma', 0.98, 0.999)
        net_size = trial.suggest_categorical('net_size', [256, 512, 1024])
        
        # Use good defaults for everything else
        env = setup_environment()
        
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            buffer_size=1500000,  # Good default
            tau=0.005,           # Good default
            policy_kwargs=dict(net_arch=[net_size, net_size, 256]),
            verbose=0
        )
        
        model.learn(total_timesteps=100000)  # Quick training
        score = evaluate_model_performance(model, env)
        env.close()
        
        return score
    
    study = optuna.create_study(study_name="gripper_optimization", 
                                storage="sqlite:///gripper_search.db",  # Persistent storage
                                direction='maximize',
                                load_if_exists=True)  # Resume if study already exists)
    study.optimize(objective, n_trials=25) 
    
    return study.best_params

# PARAMETER RANGES SUMMARY
def get_recommended_ranges():
    """Summary of recommended parameter ranges for gripper tasks."""
    
    return {
        # TIER 1: High Impact
        'learning_rate': (5e-5, 8e-4, 'log'),      # Most critical
        'batch_size': [256, 512, 1024],            # Stability
        'gamma': (0.975, 0.999),                   # Long-term planning
        'net_size': [512, 1024, 1536],             # Network capacity
        'n_layers': (2, 4),                        # Network depth
        
        # TIER 2: Moderate Impact  
        'buffer_size': [1000000, 1500000, 2000000], # Memory
        'train_freq': [1, 2, 4],                   # Update frequency
        'gradient_steps': [1, 2, 4],               # Training intensity
        'tau': (0.003, 0.008),                     # Target network update
        
        # TIER 3: Fine-tuning
        'target_policy_noise': (0.1, 0.25),       # Exploration
        'target_noise_clip': (0.4, 0.6),          # Noise control
        'action_noise_std': (0.05, 0.2),          # Action exploration
        'learning_starts': [5000, 10000, 25000],   # Training delay
    }

if __name__ == "__main__":
    print("Starting quick gripper search...")
    output = quick_gripper_search()
    print("Best hyperparameters found:")
    try:
        print(output)
    except Exception as e:
        print(f"Error printing output: {e}")
    for key, value in output.items():
        print(f"{key}: {value}")