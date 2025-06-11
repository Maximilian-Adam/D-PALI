from gymnasium.envs.registration import register
from .d_pali_hand import DPALI_Hand

register(
    id="DPALIHand-v0",
    entry_point="envs.d_pali_hand:DPALI_Hand",
)