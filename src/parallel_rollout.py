# src/parallel_rollout.py

import atexit
import os

import torch

from src.env import PlanarClawEnv
from src.rollout import sample_episode


# Per-worker state
_worker_env = None
_worker_policy = None
_worker_sample_action = None


# Pool initialiser
def init_worker(policy_factory, sample_action):
    global _worker_env, _worker_policy, _worker_sample_action

    # Separate worker processes
    torch.set_num_threads(1)

    # Create worker environment
    _worker_env = PlanarClawEnv(gui=False)

    # Create worker policy
    _worker_policy = policy_factory()

    # Store action sampler
    _worker_sample_action = sample_action

    # Close PyBullet connection
    atexit.register(_worker_env.close)


# Sample one complete trajectory from a frozen policy
def _sample_episode_task(policy_state_dict: dict, seed: int) -> dict:
    # Seeding per task, with a seed unique to this episode
    torch.manual_seed(seed)

    # Load the same policy
    _worker_policy.load_state_dict(policy_state_dict)

    # Sample episode
    return sample_episode(_worker_env, _worker_policy, _worker_sample_action)


# Return a worker's process id and environment identity
def _worker_identity() -> tuple[int, int]:
    return os.getpid(), id(_worker_env)


# Collect a batch of complete trajectories in parallel
def collect_trajectories_parallel(pool, policy_state_dict: dict, batch_size: int, seed_start: int) -> list[dict]:
    # Submit this batch of tasks
    futures = [
        pool.submit(_sample_episode_task, policy_state_dict, seed_start + episode_index)
        for episode_index in range(batch_size)
    ]

    # Preserve submission order
    return [future.result() for future in futures]
