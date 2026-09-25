# src/checkpoint.py

from datetime import datetime

import torch


# Whether to print at this update count
def should_print_progress(completed_updates: int, interval: int) -> bool:
    return completed_updates % interval == 0


# Whether to checkpoint at this update count
def should_save_checkpoint(completed_updates: int, interval: int) -> bool:
    return completed_updates % interval == 0


# Build update-based checkpoint path
def checkpoint_path(directory, timestamp, completed_updates: int):
    return directory / f"{timestamp}_update_{completed_updates}.pt"


# Save REINFORCE checkpoint
def save_reinforce_checkpoint(
    policy,
    optimiser,
    completed_updates: int,
    episodes_sampled: int,
    directory,
):
    # Current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Checkpoint path
    path = checkpoint_path(
        directory,
        timestamp,
        completed_updates,
    )

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


# Save PPO checkpoint
def save_ppo_checkpoint(
    policy,
    value_network,
    policy_optimiser,
    value_optimiser,
    completed_updates: int,
    episodes_sampled: int,
    directory,
):
    # Current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Checkpoint path
    path = checkpoint_path(
        directory,
        timestamp,
        completed_updates,
    )

    # Save training state
    torch.save(
        {
            "update": completed_updates,
            "episodes_sampled": episodes_sampled,
            "timestamp": timestamp,
            "policy_state_dict": policy.state_dict(),
            "value_state_dict": value_network.state_dict(),
            "policy_optimiser_state_dict": policy_optimiser.state_dict(),
            "value_optimiser_state_dict": value_optimiser.state_dict(),
        },
        path,
    )

    print(f"Saved checkpoint: {path}")

    return path
