# D-PALI
Repo For D-Pali Project


## How to run simulation
- Create virutal environment locally (make sure your venv name is in the gitignore)
  
  ```bash
  python -m venv venvName
  ```
- Activate your venv
    ```bash
    venvName/scripts/activate
    ```
- Install required files
  ```bash
  pip install -r requirements.txt
  ```
- Setup project
  ```bash
  python -m pip install -e .
  ```
- Run test2.py script to check everything works
  ```bash
  python scripts/test2.py
  ```
  
## TD3 Training:
### Modes
The TD3 training has 3 main modes, training, testing and continuation. They can be selected via the main call at the bottom of the script. There you can also adjust the filepath for the model, and the total timesteps for training (with the exception of continuation).

Continuation has it's own variables for old and new filepath, and extra steps to train.

### Variables
There's a few key variables at the top of the file too:
- global_eval_freq - This defines how often (in steps) the model is evaluated (for a new best model to be saved)
- global_max_episode_steps - The number of steps the enviroment can take before the model is reset
- global_save_freq - This defines how often (in steps) the model will save mid-training checkpoints
- global_reward_threshold - This is how high the average reward must be (during an eval) for the training to end early (set very high to ensure all timesteps are trained, reccomended on first runs. Or set to a reasonable value for your reward function to save early and save time)

### Performance
The script is setup to automatically detect a GPU or CPU config, however there are some variables you may want to experiment with to get the best performance out of your system:
- Batch size
- frame_skip
- train_freq and gradient_steps
- For maximum performance, you may also want to experiment with multiple enviroments learning on the same model, however that isn't covered here (yet)