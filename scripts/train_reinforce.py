# scripts/train_reinforce.py

from datetime import datetime

import torch

import config.simulation as CONFIG
from src.algorithms.reinforce import PolicyNetwork, train_batch
from src.env import PlanarClawEnv


# Initialise environment
env = PlanarClawEnv(gui=False)

# Initialise policy
policy = PolicyNetwork()

# Initialise optimiser
optimiser = torch.optim.Adam(
    policy.parameters(),
    lr=CONFIG.LEARNING_RATE,
)

# Create checkpoint directory
CONFIG.REINFORCE_CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Save policy checkpoint
def save_checkpoint(policy, optimiser, episodes_completed):
    # Current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Checkpoint path
    checkpoint_path = (
        CONFIG.REINFORCE_CHECKPOINT_DIR
        / f"{timestamp}_episode_{episodes_completed}.pt"
    )

    # Save training state
    torch.save(
        {
            "episode": episodes_completed,
            "timestamp": timestamp,
            "policy_state_dict": policy.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
        },
        checkpoint_path,
    )

    print(f"Saved checkpoint: {checkpoint_path}")


try:
    # Initialise training progress
    episodes_completed = 0
    update = 0

    # Save initial untrained policy
    save_checkpoint(
        policy,
        optimiser,
        episodes_completed,
    )

    # Train policy
    while episodes_completed < CONFIG.NUM_TRAINING_EPISODES:
        # Number of episodes remaining
        episodes_remaining = (
            CONFIG.NUM_TRAINING_EPISODES
            - episodes_completed
        )

        # Batch size, including possible smaller final batch
        batch_size = min(
            CONFIG.REINFORCE_BATCH_SIZE,
            episodes_remaining,
        )

        # Episode range for current batch
        episode_start = episodes_completed
        episode_end = episodes_completed + batch_size - 1

        # Train policy on batch of trajectories
        diagnostics = train_batch(
            env,
            policy,
            optimiser,
            CONFIG.GAMMA,
            batch_size,
        )

        # Update number of sampled episodes
        episodes_completed += batch_size

        # Print training diagnostics
        print(
            f"Update {update:4d} | "
            f"episodes {episode_start:4d}-{episode_end:<4d} | "
            f"mean return {diagnostics['mean_discounted_return']:8.2f} | "
            f"min {diagnostics['min_discounted_return']:8.2f} | "
            f"max {diagnostics['max_discounted_return']:8.2f} | "
            f"loss {diagnostics['mean_loss']:10.2f} | "
            f"grad {diagnostics['gradient_norm']:8.2f}"
        )

        # Save checkpoint at configured episode intervals
        if (
            episodes_completed % CONFIG.CHECKPOINT_INTERVAL_EPISODES == 0
            or episodes_completed == CONFIG.NUM_TRAINING_EPISODES
        ):
            save_checkpoint(
                policy,
                optimiser,
                episodes_completed,
            )

        # Increment optimiser update count
        update += 1

finally:
    env.close()
    