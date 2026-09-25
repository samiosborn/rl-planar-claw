# scripts/run_env.py

import random

import config.simulation as SIM_CONFIG
from src.env import PlanarClawEnv
from src.scene import apply_camera


# Create environment
env = PlanarClawEnv(gui=True)

# Side-on camera
apply_camera()

try:
    # Reset state
    state = env.reset()

    print("Initial observation:", state)
    print("Observation shape:", state.shape)

    # Run one episode
    done = False

    for _ in range(SIM_CONFIG.MAX_EPISODE_STEPS + 1):
        # Random actions
        actions = [
            random.randrange(len(SIM_CONFIG.JOINT_ACTION_VELOCITIES))
            for _ in SIM_CONFIG.JOINTS
        ]

        # Step environment
        next_state, reward, done = env.step(actions)

        # Update state
        state = next_state

        # Check whether episode has ended
        if done:
            break

    # Failsafe
    if not done:
        raise RuntimeError("Episode exceeded maximum number of steps")

    # Print episode statistics
    print(
        f"Episode ended | "
        f"steps={env.step_count} | "
        f"done={done}"
    )

finally:
    # Close environment
    env.close()
