# scripts/train_reinforce.py

from concurrent.futures import ProcessPoolExecutor

import torch

import config.reinforce as REINFORCE_CONFIG
from src.algorithms.reinforce import PolicyNetwork, sample_action, train_batch
from src.checkpoint import (
    save_checkpoint,
    should_print_progress,
    should_save_checkpoint,
)
from src.parallel_rollout import init_worker


# Initialise policy
policy = PolicyNetwork()

# Initialise optimiser
optimiser = torch.optim.Adam(
    policy.parameters(),
    lr=REINFORCE_CONFIG.LEARNING_RATE,
)

# Create checkpoint directory
REINFORCE_CONFIG.CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Persistent worker pool for the whole training run
# Each worker creates its own PyBullet environment and policy exactly once (see src/parallel_rollout.py)
# No PyBullet connection is created in this process
pool = ProcessPoolExecutor(
    max_workers=REINFORCE_CONFIG.NUM_WORKERS,
    initializer=init_worker,
    initargs=(PolicyNetwork, sample_action),
)

try:
    # Initialise training progress
    # completed_updates counts optimiser updates already performed
    episodes_completed = 0
    completed_updates = 0

    # Train policy
    while episodes_completed < REINFORCE_CONFIG.NUM_TRAINING_EPISODES:
        # Save policy after exactly completed_updates optimiser updates (0 is the untrained policy)
        if should_save_checkpoint(completed_updates, REINFORCE_CONFIG.CHECKPOINT_INTERVAL_UPDATES):
            save_checkpoint(
                policy,
                optimiser,
                completed_updates,
                episodes_completed,
                REINFORCE_CONFIG.CHECKPOINT_DIR,
            )

        # Number of episodes remaining
        episodes_remaining = (
            REINFORCE_CONFIG.NUM_TRAINING_EPISODES
            - episodes_completed
        )

        # Batch size, including possible smaller final batch
        batch_size = min(
            REINFORCE_CONFIG.BATCH_SIZE,
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
            REINFORCE_CONFIG.GAMMA,
            batch_size,
            episodes_completed,
        )

        # Print training diagnostics every PRINT_INTERVAL_UPDATES updates
        # The batch was sampled from the policy after completed_updates updates, before the next one
        if should_print_progress(completed_updates, REINFORCE_CONFIG.PRINT_INTERVAL_UPDATES):
            print(
                f"Update {completed_updates:4d} | "
                f"episodes {episode_start:4d}-{episode_end:<4d} | "
                f"mean return {diagnostics['mean_discounted_return']:8.2f} | "
                f"min {diagnostics['min_discounted_return']:8.2f} | "
                f"max {diagnostics['max_discounted_return']:8.2f} | "
                f"loss {diagnostics['mean_loss']:10.2f} | "
                f"grad {diagnostics['gradient_norm']:8.2f}"
            )

        # Update counters after the optimiser step
        episodes_completed += batch_size
        completed_updates += 1

    # Save final policy after the last optimiser update
    save_checkpoint(
        policy,
        optimiser,
        completed_updates,
        episodes_completed,
        REINFORCE_CONFIG.CHECKPOINT_DIR,
    )

finally:
    pool.shutdown(wait=True)
