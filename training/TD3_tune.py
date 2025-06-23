import gymnasium as gym
import numpy as np

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import report

from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import torch
import os

def setup(mode="test", log_dir=None, max_episode_steps=500):
    import envs  # registers DPALIHand-v0
    """Set up the environment with optional monitoring."""
    _render_mode = "human" if mode == "test" else None
    frame_skip = 5 if mode == "test" else 20
    env = gym.make("DPALIHand-v0", 
                   render_mode=_render_mode, 
                   max_episode_steps=max_episode_steps,
                   frame_skip=frame_skip)
    
    if log_dir and mode == "train":
        env = Monitor(env, log_dir)
    
    obs, _ = env.reset()
    return env

def train_td3(config):    
    max_episodes = config["max_episodes"] if "max_episodes" in config else 100000
    report_batch = config["report_batch"] if "report_batch" in config else 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = setup("train", max_episode_steps = 500)
    eval_env = setup("eval", max_episode_steps=500)

    model = TD3("MlpPolicy", env,
                learning_rate=config["learning_rate"],
                batch_size=config["batch_size"],
                buffer_size=int(config["buffer_size"]),
                tau=config["tau"],
                gamma=config["gamma"],
                policy_kwargs=config["policy_kwargs"],
                verbose=0,
                device=device)
    for i in range(max_episodes // report_batch):
        model.learn(total_timesteps=report_batch,reset_num_timesteps=False)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)
        report({"mean_reward": mean_reward, "training_iteration": (i+1)*report_batch}) 

# search_space = {
#     "learning_rate": tune.loguniform(1e-5, 1e-3),
#     "batch_size": tune.choice([256,512]),
#     "buffer_size": tune.choice([5e5, 1e6]),
#     "tau": tune.uniform(0.005, 0.02),
#     "gamma": tune.uniform(0.9, 0.99),
#     "policy_kwargs": tune.choice([
#         {"pi": [1024,512,512, 256], "qf": [1024, 64]},
#         {"pi": [1024,512,256], "qf": [1024, 64]},
#         {"pi": [512,512, 256], "qf": [1024, 64]},
#         {"pi": [512, 256], "qf": [1024, 64]},
#     ]),
#     "max_episodes": 100000,
#     "report_batch": 1000
# }

search_space = {
    "learning_rate": tune.loguniform(1e-4, 9e-4),
    "batch_size": tune.choice([512, 1024]),
    "buffer_size": tune.choice([1e6,5e6]),
    "tau": tune.choice([0.005]),
    "gamma": tune.choice([0.99]),
    "policy_kwargs": tune.choice([
        {"net_arch": {"pi": [1024,512,256], "qf": [512, 256,64]}},
        {"net_arch": {"pi": [1024,512,256], "qf": [1024, 64]}},
        {"net_arch": {"pi": [512,512, 256], "qf": [1024, 64]}},
        {"net_arch": {"pi": [512,512, 256], "qf": [512, 256,64]}},
    ]),
    "max_episodes": 100000,
    "report_batch": 1000
}


scheduler = ASHAScheduler(
    metric="mean_reward",
    mode="max",
    max_t=100000, ###############
    grace_period=2000,
    reduction_factor=2
)

tuner = tune.Tuner(
    tune.with_resources(train_td3, {"cpu": 19, "gpu": 1}),
    param_space=search_space,
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=20, ##############
    ),
    run_config=tune.RunConfig(
        verbose=1,
        storage_path = "file://" + os.path.abspath("./training/logs/TD3_tune_results/"),
        name="td3_dpali_experiment"
    ),
    
)
results = tuner.fit()
best_result = results.get_best_result(
    metric="mean_reward",
    mode="max",
    scope="last"
)

print("Best config:", best_result.config)
print("Best reward:", best_result.metrics["mean_reward"])
print("Trial path:", best_result.path)

df = results.get_dataframe()
df.to_csv("./training//logs/TD3_tune_results/td3_all_trials_2.csv", index=False)
print("All trials saved to td3_all_trials.csv")