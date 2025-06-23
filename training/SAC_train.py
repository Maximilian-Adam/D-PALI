from ray import tune
from ray.rllib.algorithms.sac import SACConfig
from gymnasium.envs.registration import register
import envs  # registers DPALIHand-v0
import os
from envs.d_pali_hand import DPALI_Hand
import gymnasium as gym
from ray.tune.registry import register_env
import torch

def env_creator(env_config):
    return DPALI_Hand(**env_config)

register_env("DPALIHand-v0", env_creator)

config = SACConfig()
config.environment(
    env="DPALIHand-v0",
    env_config={
        "render_mode": None,
        "frame_skip": 20,
    }
)
config.framework("torch")
config.env_runners(
    num_env_runners=10,
    rollout_fragment_length=200,
    explore=True,
)
config.training(
    train_batch_size=512,
    tau=0.005,
    gamma=0.99,
    replay_buffer_config={"capacity": 1_000_000},
    target_entropy="auto",
)

config.resources(
    num_cpus_per_worker=1, # Set to 0 if you don't
    num_gpus=1,  
)

config.evaluation_config = {
    "env_runners": {
        "explore": False
    }
}


tune.Tuner(
    "SAC",
    run_config=tune.RunConfig(
        stop={"training_iteration": 200},
        storage_path= "file://" + os.path.abspath("./training/logs/sac_train/"),
        name="sac_dpali_v1"
    ),
    param_space=config.to_dict()
).fit()
