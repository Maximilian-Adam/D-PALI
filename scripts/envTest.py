import gymnasium as gym
import envs  # This line is important to register your custom environment
import time
import numpy as np

def run_environment_test():
    """
    Creates an instance of the DPALIHand-v0 environment and tests its reset logic.
    """
    print("--- Starting Environment Test ---")
    
    # Create a single instance of the environment, with rendering enabled.
    # We do not use any wrappers like VecNormalize for this simple test.
    env = gym.make("DPALIHand-v0", frame_skip = 5, render_mode="human")

    # Run 5 test episodes
    for episode in range(5):
        print(f"\n--- Episode {episode + 1} ---")
        
        # Reset the environment. This should trigger your randomization logic.
        obs, info = env.reset()
        
        # Access the underlying environment to call your helper method
        # .unwrapped is used to get past any potential wrappers
        target_pos = env.unwrapped._get_target_pos()
        target_orientation = env.unwrapped._get_target_orientation()
        cube_pos = env.unwrapped._get_cube_pos()
        cube_orientation = env.unwrapped._get_cube_orientation()
        print(f"Randomized Target Position: {np.round(target_pos, 4)}")
        print(f"Cube Position: {np.round(cube_pos, 4)}")

        # Render the environment for a short time to visually confirm
        for _ in range(100): # Render for 50 steps
            env.render()
            time.sleep(1/60) # Slow down rendering to be watchable

    # Clean up the environment
    env.close()
    print("\n--- Test Finished ---")

if __name__ == "__main__":
    run_environment_test()