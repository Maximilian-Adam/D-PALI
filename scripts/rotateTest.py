import gymnasium as gym
import envs  # This line is important to register your custom environment
import time
import numpy as np

def test_base_joint_movement():
    print("--- Starting Base Joint Movement Test ---")
    
    env = gym.make("DPALIHand-v0", frame_skip = 5, render_mode="human")

    obs, info = env.reset()

    num_actuators = env.unwrapped.model.nu
    
    # The indices of the base joints in the action vector, based on
    # the order in Shared3D.xml
    base_joint_indices = [2, 5, 8] # Corresponds to L Base, R Base, U Base
    
    test_duration_steps = 500
    
    print(f"Testing for {test_duration_steps} steps...")
    print(f"Actuator count: {num_actuators}. Base joint indices: {base_joint_indices}")

    observations_data = []
    actions_data = []

    for step in range(test_duration_steps):
  
        target_base_action = np.sin(2 * np.pi * step / test_duration_steps)
        
        action = np.zeros(num_actuators)
        
        action[base_joint_indices] = target_base_action
        
        env.step(action)

        env.render()
        
        if step % 50 == 0:
            print(f"Step {step}: Commanding base joints to target angle ~ {np.round(target_base_action * np.pi, 2)} rad")

    # Clean up the environment
    env.close()
    print("\n--- Test Finished ---")

if __name__ == "__main__":
    test_base_joint_movement()