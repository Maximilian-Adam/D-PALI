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

## 12/06/25
- All to do items from yesterday still remain
- Current hyperparameters are looking good, they've got the reward down to -200 on average after 100k steps. 
    - Based on this, some reward rebalancing is likely needed
    - Mainly higher success bonus, and a higher (x10) approach reward and contact (x2). I think keeping the table penalty as is should be fine
    - Seems like rebalancing the reward is more important than the hyperparameter searching, so I'll likely need to do that first, then research the hyperparameters.

## 13/06/25 
- Again, previous items still remain. I've completed the hyperparameter search, and the results (for the negative biased reward) are better in terms of contacts (Ori v2.0), but it doesn't really try to move the cube very much towards the target orientation. 
    - Akin might be doing a hyperparameter serach with the positive biased reward, so if he's doing that, I can try experimenting with more training for this model to see if there's an improvement, and/or increasing the action noise, in order to get it to try and explore turning more. 

## 14/06/25 
- Ori v2.1 - Training continued for 500k steps, no changes
- Ori v2.2 - Total of 2.5M steps, didn't turn much but good grip
- Ori v3.0 - Turns pretty well, and has a consistent 2 finger grip without dropping (best so far)

## 15/06/25
- Ori v3.1 is pretty decent, but I think I'm still gonna try adjusting the reward function to be more responsive, and training a new model, as it's off slightly still

## 16/06/25
- Ori v4.0 has some major changes to reward function as well as environment (quaternion's difference calculated correctly), which has given some pretty good performance improvements.
- The model is now more easily able to match a rotation. The only bit where it struggles seems to be when it collides with another finger 
- Might begin some randomisation to the orientation

## 17/06/25
- Ori v4.1 is training with randomisation now, hopefully the grip will improve here too
    - Update, did not. It sort of works, but still doesn't really match it, and needs the 3rd finger gripping
- Ori v4.2 will be a fresh model with higher success reward to skew results.
    - Pretty good at rotations (even if random) but still only using 2 fingers
- Action scale was also severly limited (1/3 of what it should be), need to likely restart training to accomodate this change 
    - Should also make the position control easier
    - 3 finger grasp should also be rewarded more heavily in next model

## 18/06/25
- Ori v4.3 training as described above
- Still refuses to hold with 3 fingers, but seems to be able to match the orientation better, only thing it messes up now is rotataing it in a different axis

## 19/06/25
- Updated model for more overlapping workspace
- Running new v5.0 with new model
- May need to switch to old link for long link

## 23/05/25
- v5.0 n 5.1 were decent, not super consistent due to the model dropping the cube by default
- v5.2 is okay, it can rotate pretty well, but if it drops the cube, it gets confused
- v5.3 is gonna be a continuation of this, we shall see
    - v5.3 works a bit better, still drops it though, so I've tweaked some values in the xml, and hopefully it should be able to hold it from the start now.
- v5.4 is gonna be trained with the new xml (end effectors can clip for this though, since it's using a new sovler)
    - Ok that was a steaming pile of horse shit. Gonna stick to the old solver, because it somehow unlearnt everything from the previous itterations.

## 24/06/25
- Gonna try doing orientation in a different axis today, and then also potentailly a position run, and hope for the best
- May wanna try normalisng reward values too
    - Nvm vec normalise already does this


