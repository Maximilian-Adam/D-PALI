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
  
## PPO training
The first stage is to train the gripper to catch the cube. The reward function is set to sum of the distance from all end effectors and the cube. 
### How to run PPO training
- turn on tensorboard monitoring
```bash
pip install tensorboard
tensorboard --version
tensorboard --logdir=training/logs/ppo_logs/
```
- run training script: make sure you are in the root dir of this repo
```bash
python3 ./training/train_PPO.py
```