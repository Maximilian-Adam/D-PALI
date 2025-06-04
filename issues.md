Issues:
- Reset was nested in init, so it was never actually called.
- Observation space only included joint velocities and positions, no cube, end effectors or target
- Episodes only ended when they were done, never truncated due to time or issue (e.g. cube falling off)
- PPO is bad for training
- No mid training checkpoints
