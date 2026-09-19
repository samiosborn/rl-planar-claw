# scripts/train_reinforce.py

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import torch

import config.simulation as CONFIG
from src.algorithms.reinforce import PolicyNetwork, train_batch
from src.parallel_rollout import init_worker


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


# Persistent worker pool for the whole training run
# Each worker creates its own PyBullet environment and policy exactly once (see src/parallel_rollout.py)
# No PyBullet connection is created in this process
pool = ProcessPoolExecutor(
    max_workers=CONFIG.REINFORCE_NUM_WORKERS,
    initializer=init_worker,
)

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

        # Train policy on a batch of trajectories sampled in parallel
        # seed_start = episodes_completed guarantees every episode in the run gets a unique seed
        diagnostics = train_batch(
            pool,
            policy,
            optimiser,
            CONFIG.GAMMA,
            batch_size,
            episodes_completed,
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

        # Save checkpoint on the first update that reaches each interval boundary
        # Batches are not aligned to the interval, so testing divisibility would only save at the least common multiple
        if (
            episodes_completed // CONFIG.CHECKPOINT_INTERVAL_EPISODES
            > episode_start // CONFIG.CHECKPOINT_INTERVAL_EPISODES
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
    pool.shutdown(wait=True)
