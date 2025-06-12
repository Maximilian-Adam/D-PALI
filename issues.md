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

## 10/06/25
- Reward may need tweaking (current ori v1 best model tries to turn, but further training (v1.1) yielded no results)
- Stil need:
    - Updated Model
    - Find Hyperparamters
    - Find workspace
- Also may want to look into tweaking action noise for better exploration, and also developing a better network architecture
- Basically, go through the reccomendations Claude gave too (and/or just do the hyperparamter stuff)

## 11/06/25
- Started Hyperparameter search, got 10 trails done, aiming for a total of 50.
    - Using Bayseian optimisation for the search
- Also began looking into the workspace stuff with the new Matlab scripts given, the graphs for our values don't quite look right, so from here there are two optioons:
    - Figure out what's going wrong in the current matlab script for old gripper
    - Make our own script for our gripper (leaning towards this atm)
- These are still left to do, but likely won't be completed over the next few days:
    - Update Model (with smaller base)
    - Find Hyperparameters (doing)
    - Find Workspace (doing)
    - Tweak reward 
    - Tweak noise action
    - Claude reccomendations
