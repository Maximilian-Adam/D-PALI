import gymnasium as gym
from ray.rllib.algorithms.algorithm import Algorithm
import numpy as np
import time
import os

# ======= CONFIG ========
CHECKPOINT_PATH = "/home/roy/Documents/STUDY/IC/YEAR3_PROJECT/D-PALI/training/logs/sac_train/sac_dpali_v1/SAC_DPALIHand-v0_fd4fa_00000_0_2025-06-18_04-19-05/checkpoint_000000"  # 替换为你实际的路径
NUM_EPISODES = 5
RENDER = True
SLEEP_TIME = 1 / 60
# ========================

def evaluate_policy(checkpoint_path, num_episodes):
    print(f"Loading SAC model from checkpoint: {checkpoint_path}")
    algo = Algorithm.from_checkpoint(checkpoint_path)

    env = gym.make("DPALIHand-v0", render_mode="human" if RENDER else None, frame_skip=20)

    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_info = []

        print(f"\n===== Episode {ep + 1} =====")
        while not done:
            action = algo.compute_single_action(obs, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1
            episode_info.append(info)

            if RENDER:
                env.render()
                time.sleep(SLEEP_TIME)

        final_info = episode_info[-1] if episode_info else {}
        print(f"✓ Episode reward: {total_reward:.2f} | Steps: {steps}")
        print(f"  Final cube_target_distance: {final_info.get('cube_target_distance', 'N/A'):.4f}")
        print(f"  Final num_contacts: {final_info.get('num_contacts', 'N/A')}")
        rewards.append(total_reward)

    env.close()
    print("\n===== Evaluation Summary =====")
    print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Max reward: {np.max(rewards):.2f}")
    print(f"Min reward: {np.min(rewards):.2f}")

if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    evaluate_policy(CHECKPOINT_PATH, NUM_EPISODES)
