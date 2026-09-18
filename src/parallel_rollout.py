# src/parallel_rollout.py
#
# Multiprocessing-specific logic for sampling trajectories in parallel
# The policy-gradient mathematics stays in src/algorithms/reinforce.py
# This module only concerns itself with worker lifecycle and dispatch

import atexit
import os

import torch

from src.algorithms.reinforce import PolicyNetwork
from src.env import PlanarClawEnv
from src.rollout import sample_episode


# Per-worker state, created exactly once per worker process by init_worker
# A process can only host one PlanarClawEnv at a time (see src/env.py)
# Each worker reuses this single environment and policy across every task it receives
_worker_env = None
_worker_policy = None


# Pool initialiser: create this worker's environment and policy once
def init_worker():
    global _worker_env, _worker_policy

    # Parallelism should come from separate worker processes
    # Not from nested per-process PyTorch thread pools
    torch.set_num_threads(1)

    _worker_env = PlanarClawEnv(gui=False)
    _worker_policy = PolicyNetwork()

    # Close this worker's PyBullet connection cleanly when the process exits
    atexit.register(_worker_env.close)


# Worker task: sample one complete trajectory from a frozen policy snapshot
# Must be module-level so it can be pickled and sent to worker processes
def _sample_episode_task(policy_state_dict: dict, seed: int) -> dict:
    # Forked worker processes inherit the parent's RNG state
    # Seeding per task, with a seed unique to this episode, guarantees independent samples
    torch.manual_seed(seed)

    # Every trajectory in a batch must be sampled from the exact same frozen snapshot
    _worker_policy.load_state_dict(policy_state_dict)

    return sample_episode(_worker_env, _worker_policy)


# Return this worker's process id and environment identity
# Used only by tests, to confirm worker processes and environments are isolated and reused
def _worker_identity() -> tuple[int, int]:
    return os.getpid(), id(_worker_env)


# Collect batch_size complete trajectories in parallel from one frozen policy snapshot
# No optimiser update may happen until every trajectory here has been collected
# This is what keeps a batch on-policy
def collect_trajectories_parallel(pool, policy_state_dict: dict, batch_size: int, seed_start: int) -> list[dict]:
    # Submit exactly batch_size tasks, each with a seed unique to this batch
    futures = [
        pool.submit(_sample_episode_task, policy_state_dict, seed_start + episode_index)
        for episode_index in range(batch_size)
    ]

    # future.result() preserves submission order
    # Worker exceptions propagate naturally
    return [future.result() for future in futures]
