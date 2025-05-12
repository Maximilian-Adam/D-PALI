from stable_baselines3 import SAC
from sb3_contrib import TQC

env = gym.make("d_pali_hand-v0")
model = SAC("MultiInputPolicy", env, verbose=1)
model.learn(3_000_000)
model.save("d_pali_hand_sac")
