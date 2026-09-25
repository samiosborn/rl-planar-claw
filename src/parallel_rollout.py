# src/parallel_rollout.py

import atexit
import os

import torch

from src.env import PlanarClawEnv
from src.rollout import sample_episode


# Per-worker state, created exactly once per worker process by init_worker - a process can only host one PlanarClawEnv at a time
_worker_env = None
_worker_policy = None
_worker_sample_action = None


# Pool initialiser: create this worker's environment and policy once
def init_worker(policy_factory, sample_action):
    global _worker_env, _worker_policy, _worker_sample_action

    # Parallelism should come from separate worker processes (not from nested per-process PyTorch thread pools)
    torch.set_num_threads(1)

    # Create worker environment
    _worker_env = PlanarClawEnv(gui=False)

    # Create worker policy
    _worker_policy = policy_factory()

    # Store action sampler
    _worker_sample_action = sample_action

    # Close this worker's PyBullet connection
    atexit.register(_worker_env.close)


# Worker task: sample one complete trajectory from a frozen policy snapshot
def _sample_episode_task(policy_state_dict: dict, seed: int) -> dict:
    # Seeding per task, with a seed unique to this episode, guarantees independent samples
    torch.manual_seed(seed)

    # Every trajectory in a batch must be sampled from the exact same frozen snapshot
    _worker_policy.load_state_dict(policy_state_dict)

    # Sample episode with the injected action sampler
    return sample_episode(_worker_env, _worker_policy, _worker_sample_action)


# Return this worker's process id and environment identity
def _worker_identity() -> tuple[int, int]:
    return os.getpid(), id(_worker_env)


# Collect batch_size complete trajectories in parallel from one frozen policy snapshot
def collect_trajectories_parallel(pool, policy_state_dict: dict, batch_size: int, seed_start: int) -> list[dict]:
    # Submit exactly batch_size tasks, each with a seed unique to this batch
    futures = [
        pool.submit(_sample_episode_task, policy_state_dict, seed_start + episode_index)
        for episode_index in range(batch_size)
    ]

    # future.result() preserves submission order
    return [future.result() for future in futures]
