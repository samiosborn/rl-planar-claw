# src/checkpoint.py
#
# Training clock, progress and checkpoint scheduling for REINFORCE
# The clock is completed_updates: the number of optimiser updates already performed
# completed_updates = 0 is the initial untrained policy
# Scheduling never depends on batch size or on the number of sampled episodes

from datetime import datetime

import torch

import config.simulation as CONFIG


# Whether to print at this update count
def should_print_progress(completed_updates):
    return completed_updates % CONFIG.PRINT_INTERVAL_UPDATES == 0


# Whether to checkpoint at this update count
def should_save_checkpoint(completed_updates):
    return completed_updates % CONFIG.CHECKPOINT_INTERVAL_UPDATES == 0


# Build update-based checkpoint path
def checkpoint_path(timestamp, completed_updates):
    return CONFIG.REINFORCE_CHECKPOINT_DIR / f"{timestamp}_update_{completed_updates}.pt"


# Save policy checkpoint
def save_checkpoint(policy, optimiser, completed_updates, episodes_sampled):
    # Current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Checkpoint path
    path = checkpoint_path(timestamp, completed_updates)

    # Save training state
    torch.save(
        {
            "update": completed_updates,
            "episodes_sampled": episodes_sampled,
            "timestamp": timestamp,
            "policy_state_dict": policy.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
        },
        path,
    )

    print(f"Saved checkpoint: {path}")

    return path
