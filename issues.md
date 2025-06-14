# Issues:

## 01/06/25
- Reset was nested in init, so it was never actually called.
- Observation space only included joint velocities and positions, no cube, end effectors or target
- Episodes only ended when they were done, never truncated due to time or issue (e.g. cube falling off)
- PPO is bad for training
- No mid training checkpoints

## 02/06/25
- Reward function improved, but still not moving the cube much.
    - Look into long grippers and see if they're having a negative impact
- Try putting down a fixed target first, and getting it to move there before adding in randomisation
- Need to update model

## 03/06/25
- v4.1 is currently the best, with a static target it nearly gets it right, just slightly off, and doesn't grip very well
- Need to add orientation to the observation space too

## 09/06/25
- Normalise Obsrevations 1 
- Dynamic learning rate 2
- Limit successed based rewards 3
- Balance rewards 4
- Custom Tensorboard logs
- Back link is wrong
- Update Model
- Find hyperparamters 
- Find workspace