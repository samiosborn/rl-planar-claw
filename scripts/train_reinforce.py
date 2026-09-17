# scripts/train_reinforce.py

import torch

import config.simulation as CONFIG
from src.algorithms.reinforce import PolicyNetwork, train_batch
from src.env import PlanarClawEnv


# Initialise environment
env = PlanarClawEnv(gui=False)

# Initialise policy
policy = PolicyNetwork()

# Initialise optimiser
optimiser = torch.optim.Adam(policy.parameters(), lr=CONFIG.LEARNING_RATE)

try:
    # Number of episodes sampled so far
    episodes_done = 0

    # Optimiser update counter
    update = 0

    # Train policy in trajectory batches until the episode budget is spent
    while episodes_done < CONFIG.NUM_TRAINING_EPISODES:
        # Handle a final, smaller batch instead of dropping episodes
        batch_size = min(
            CONFIG.REINFORCE_BATCH_SIZE,
            CONFIG.NUM_TRAINING_EPISODES - episodes_done,
        )

        diagnostics = train_batch(
            env,
            policy,
            optimiser,
            CONFIG.GAMMA,
            batch_size,
        )

        # Range of sampled episodes covered by this update
        first_episode = episodes_done
        last_episode = episodes_done + batch_size - 1

        # Print training diagnostics (one line per optimiser update)
        print(
            f"Update {update:4d} | "
            f"episodes {first_episode:4d}-{last_episode:<4d} | "
            f"mean return {diagnostics['mean_discounted_return']:8.2f} | "
            f"min {diagnostics['min_discounted_return']:8.2f} | "
            f"max {diagnostics['max_discounted_return']:8.2f} | "
            f"loss {diagnostics['mean_loss']:10.2f} | "
            f"grad {diagnostics['gradient_norm']:8.2f}"
        )

        episodes_done += batch_size
        update += 1

finally:
    env.close()
