# scripts/run_env.py

import random

import config.simulation as CONFIG
from src.env import PlanarClawEnv


# Create environment
env = PlanarClawEnv(gui=True)

try:
    # Reset state
    state = env.reset()

    print("Initial observation:", state)
    print("Observation shape:", state.shape)

    # Run one episode
    done = False

    for _ in range(CONFIG.MAX_EPISODE_STEPS + 1):
        # Random actions
        actions = [
            random.randrange(len(CONFIG.JOINT_ACTION_VELOCITIES))
            for _ in CONFIG.JOINTS
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
        f"success_steps={env.success_steps} | "
        f"done={done}"
    )

finally:
    # Close environment
    env.close()
