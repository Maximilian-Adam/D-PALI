import gymnasium as gym
import numpy as np
import mujoco
import envs                 

SCALE = np.pi / 3         

def deg2action(deg: float) -> float:
    return float(np.clip(np.deg2rad(deg) / SCALE, -1.0, 1.0))

def get_actuator_names(env):
    raw = env.unwrapped
    names = []
    for i in range(raw.model.nu):
        nm = mujoco.mj_id2name(raw.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if isinstance(nm, bytes):
            nm = nm.decode('utf-8')
        names.append(nm)
    return names


if __name__ == "__main__":
    env = gym.make("DPALIHand-v0", render_mode="human")
    try:
        obs, _ = env.reset()
        actuator_names = get_actuator_names(env)

        print(f"Found {len(actuator_names)} actuators:")
        for i, name in enumerate(actuator_names):
            print(f"  ctrl[{i}] → “{name}”")

        poses = [
            {
                "L Front":  20,
                "L Back":  -10,
                "L Base":    0,
                "R Front":  20,
                "R Back":   -10,
                "R Base":    0,
            },
            {
                "L Front":   0,
                "L Back":    0,
                "L Base":   30,
                "R Front":   0,
                "R Back":    0,
                "R Base":   30,
            },
        ]
        max_steps_per_pose = 500

        for idx, desired_deg_by_name in enumerate(poses, start=1):
            action = np.zeros(len(actuator_names), dtype=np.float32)
            for i, name in enumerate(actuator_names):
                if name in desired_deg_by_name:
                    action[i] = deg2action(desired_deg_by_name[name])

            print(f"\n>> Applying pose {idx}/{len(poses)}: {desired_deg_by_name}")

            for step in range(max_steps_per_pose):
                obs, _, done, truncated, _ = env.step(action)
                env.render()
            else:
                print(f"   → Pose {idx} done, Moving to next…")


        print("\nAll poses applied. Exiting…")

    finally:
        env.close()
