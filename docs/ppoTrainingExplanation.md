1. **Environment setup**

   * `DPALIHand-v0` is registered in `envs/__init__.py` and points to the `DPALI_Hand` MuJoCo class.&#x20;
   * The environment returns a flat observation of joint positions + velocities and accepts continuous actions that are scaled to ± π⁄3 before simulation.&#x20;
   * The reward combines

     * a **distance term** (cube → target),
     * **contact bonuses** for one, two, or all three fingertips touching the cube, and
     * an extra success bonus when the cube is very close to the target while grasped, which also ends the episode.&#x20;

2. **Training script (`train_PPO.py`)**

   * Creates the environment in “train” mode (no rendering) via `setup("train")`.&#x20;
   * Instantiates a **PPO** agent (`stable_baselines3`) with:

     * MLP policy, ReLU activations,
     * custom network sizes – π branch \[128, 128, 64, 32] and V branch \[256, 256, 128, 64],
     * learning-rate 3 × 10⁻³, CUDA device, and TensorBoard logging.&#x20;
   * Calls `model.learn(total_timesteps)` for the user-supplied step budget, then saves to `training/checkpoints/ppo_DPALIHand-v0`.&#x20;

3. **Testing / Demo**

   * The same script can be run in “test” mode, or you can use `test_PPO.py`. Both simply load the saved checkpoint, step for many iterations, and render the scene in real time.

