### Engineering (Python Class)

| Issue                                     | Possible Solution                                                                              | Working on it |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------- |
| |                  |              |

---

### Physics / MJCF

| Issue                                          | Possible Solution                                                                            | Working on it |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------- |
| Fingertip positions taken from geoms           | Add lightweight `site` markers at each fingertip and read `site_xpos` for cleaner kinematics |               |
| No domain-randomisation of materials           | Randomise friction, densities, and visual textures inside `reset_model`                      |               |
| Unrealistic link masses / inertias             | Tune masses and inertias to match the real hand, improving contact stability                 |               |
| Complex closed-chain from equality constraints | Where rigid, replace with welded joints (or soft welds) for numerical robustness             |               |

---
### RL Design 

| Issue                                                                                                     | Possible Solution                                                                                                                                              | Working on it |
| --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| **Single-worker environment** – training collects data from one simulator instance, limiting throughput.  | Wrap the env with `make_vec_env` or `gymnasium.vector.AsyncVectorEnv` (e.g. 8 workers); if you can, compile the model with *mjx* for GPU-accelerated stepping. |               |
| **No observation / return normalisation** – raw values of very different scales feed the critic.          | Place the env inside `VecNormalize` (or `RunningMeanStd` custom wrapper) so both observations *and* rewards are standardised online.                           |               |
| **Fixed Gaussian action noise (σ = 0.1)** throughout training.                                            | Swap to a scheduled σ decay, or enable state-dependent exploration (`gSDE`) that TD3 in SB3 supports; improves late-stage fine control.                        |               |
| **Single static learning rate (1 × 10⁻³).**                                                               | Use a linear or cosine LR schedule (`learn_rate_schedule` callback) or search LR with Optuna/Weights & Biases sweeps.                                          |               |
| **Vanilla FIFO replay buffer.**                                                                           | Replace with `PrioritizedReplayBuffer` for better sample efficiency, or `HerReplayBuffer` if you add a sparse-reward mode.                                     |               |
| **Train frame-skip = 20 vs. test = 5** – policy faces different dynamics at deployment.                   | Keep the same frame-skip in both modes, or introduce a curriculum that gradually lowers skip during training.                                                  |               |
| **Dense-only reward; high success threshold (5000).**                                                     | Add an optional sparse-reward flag and progressively tighten the success radius to avoid over-shaping.                                                         |               |
| **Evaluation uses identical env config** – no robustness check.                                           | Create a second `eval_env` with different friction/lighting seeds or without observation normalisation to measure generalisation.                              |               |
