# scripts/plot_rollout_reinforce.py
import argparse
from pathlib import Path
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

import config.reinforce as REINFORCE_CONFIG
import config.simulation as SIM_CONFIG
from src.algorithms.reinforce import PolicyNetwork, sample_action
from src.env import PlanarClawEnv
from src.rollout import rollout_episode
from src.visualisation.trajectory import plot_trajectory_grid


UPDATE_PATTERN = re.compile(r"_update_(\d+)\.pt$")


# Extract update number from checkpoint filename
def checkpoint_update(path):
    match = UPDATE_PATTERN.search(Path(path).name)
    if match is None:
        raise ValueError(f"Checkpoint filename does not contain an update number: {path}")
    return int(match.group(1))


# Discover checkpoints in update order
def discover_checkpoints(directory=REINFORCE_CONFIG.CHECKPOINT_DIR):
    paths = list(Path(directory).glob("*.pt"))
    valid_paths = [path for path in paths if UPDATE_PATTERN.search(path.name)]
    return sorted(valid_paths, key=lambda path: (checkpoint_update(path), path.name))


# Select evenly spaced checkpoints
def select_evenly_spaced(paths, count):
    if count < 1:
        raise ValueError("--num-checkpoints must be at least 1")
    if not paths:
        raise ValueError("No REINFORCE checkpoints were found")
    if count > len(paths):
        raise ValueError(
            f"Requested {count} checkpoints, but only {len(paths)} are available"
        )
    if count == 1:
        return [paths[-1]]

    indices = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[index] for index in indices]


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot sampled trajectories from REINFORCE checkpoints."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--checkpoints", nargs="+", type=Path, help="Explicit checkpoint paths"
    )
    selection.add_argument(
        "--num-checkpoints",
        type=int,
        help="Number of evenly spaced checkpoints to discover",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=SIM_CONFIG.NUM_ROLLOUT_EPISODES,
        help="Trajectories sampled per checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SIM_CONFIG.PROJECT_ROOT / "trajectory_checkpoints.png",
        help="Saved figure path",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--alpha-by-return",
        action="store_true",
        help="Make lower-return trajectories slightly fainter",
    )
    return parser.parse_args()


# Sample trajectories from each checkpoint
def sample_checkpoint_trajectories(checkpoint_paths, episodes, seed=None):
    if episodes < 1:
        raise ValueError("--episodes must be at least 1")

    env = PlanarClawEnv(gui=False)
    results = []
    try:
        for path in checkpoint_paths:
            # Reset random state for each checkpoint
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

            # Load policy
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            policy = PolicyNetwork()
            policy.load_state_dict(checkpoint["policy_state_dict"])
            policy.eval()

            # Sample trajectories
            trajectories = [
                rollout_episode(env, policy, sample_action, REINFORCE_CONFIG.GAMMA)
                for _ in range(episodes)
            ]

            # Summarise discounted returns
            returns = np.asarray(
                [trajectory["discounted_return"] for trajectory in trajectories]
            )
            print(
                f"{path.name}: mean={returns.mean():.4f}, std={returns.std():.4f}, "
                f"min={returns.min():.4f}, max={returns.max():.4f}"
            )
            results.append(trajectories)
    finally:
        env.close()
    return results


# Build and save trajectory figure
def main():
    args = parse_args()

    # Load checkpoint paths
    checkpoint_paths = args.checkpoints
    if checkpoint_paths is None:
        checkpoint_paths = select_evenly_spaced(
            discover_checkpoints(), args.num_checkpoints
        )

    missing = [path for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint not found: {missing[0]}")

    # Sample trajectories
    trajectories = sample_checkpoint_trajectories(
        checkpoint_paths, args.episodes, args.seed
    )

    # Plot checkpoints
    titles = [f"Update {checkpoint_update(path)}" for path in checkpoint_paths]
    fig, _ = plot_trajectory_grid(
        trajectories,
        titles=titles,
        alpha_by_return=args.alpha_by_return,
    )

    # Save figure
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {args.output}")

    # Display plot
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
