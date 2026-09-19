# tests/test_parallel_rollout.py

from concurrent.futures import ProcessPoolExecutor

import pytest

import config.simulation as CONFIG
from src.algorithms.reinforce import PolicyNetwork
from src.parallel_rollout import _worker_identity, collect_trajectories_parallel, init_worker


# Freeze one explicit CPU copy of a policy's parameters, exactly as train_batch does
def _snapshot(policy):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in policy.state_dict().items()
    }


# Test that collecting a batch returns exactly N trajectories
def test_collect_trajectories_parallel_returns_n_trajectories(pool):
    policy = PolicyNetwork()
    batch_size = 5

    trajectories = collect_trajectories_parallel(pool, _snapshot(policy), batch_size, seed_start=0)

    assert len(trajectories) == batch_size


# Test that every returned trajectory has the expected (fixed) episode length
def test_collect_trajectories_parallel_trajectory_lengths(pool):
    policy = PolicyNetwork()
    batch_size = 5

    trajectories = collect_trajectories_parallel(pool, _snapshot(policy), batch_size, seed_start=10)

    for trajectory in trajectories:
        assert trajectory["episode_length"] == CONFIG.MAX_EPISODE_STEPS
        assert len(trajectory["states"]) == CONFIG.MAX_EPISODE_STEPS
        assert len(trajectory["actions"]) == CONFIG.MAX_EPISODE_STEPS
        assert len(trajectory["rewards"]) == CONFIG.MAX_EPISODE_STEPS


# Test that a batch smaller than the pool's worker count still works
def test_batch_smaller_than_worker_count(pool):
    policy = PolicyNetwork()

    # Fewer trajectories than the pool's 4 workers, so some stay idle
    trajectories = collect_trajectories_parallel(pool, _snapshot(policy), batch_size=2, seed_start=20)

    assert len(trajectories) == 2
    assert all(trajectory["episode_length"] == CONFIG.MAX_EPISODE_STEPS for trajectory in trajectories)


# Test that a final partial batch of size 1 still works
def test_final_partial_batch_of_one(pool):
    policy = PolicyNetwork()

    trajectories = collect_trajectories_parallel(pool, _snapshot(policy), batch_size=1, seed_start=30)

    assert len(trajectories) == 1
    assert trajectories[0]["episode_length"] == CONFIG.MAX_EPISODE_STEPS


# Leading steps compared in the seeding tests
# The fingers start close to the cube, so even early steps can involve contact, which is not bit-identical across process histories
# Actions are compared exactly; rewards only loosely, as contact makes the cube's motion depend slightly on earlier episodes
SEEDING_WINDOW = 10


# Test that a fresh worker given the same policy snapshot and seed samples the same actions
def test_same_snapshot_and_seed_give_same_actions_in_fresh_workers():
    policy = PolicyNetwork()
    snapshot = _snapshot(policy)

    # Fresh single-worker pools share no process history
    pool_a = ProcessPoolExecutor(max_workers=1, initializer=init_worker)
    pool_b = ProcessPoolExecutor(max_workers=1, initializer=init_worker)

    try:
        trajectory_a = collect_trajectories_parallel(pool_a, snapshot, batch_size=1, seed_start=40)[0]
        trajectory_b = collect_trajectories_parallel(pool_b, snapshot, batch_size=1, seed_start=40)[0]
    finally:
        pool_a.shutdown(wait=True)
        pool_b.shutdown(wait=True)

    assert trajectory_a["actions"][:SEEDING_WINDOW] == trajectory_b["actions"][:SEEDING_WINDOW]
    assert trajectory_a["rewards"][:SEEDING_WINDOW] == pytest.approx(trajectory_b["rewards"][:SEEDING_WINDOW], rel=2e-2)


# Test that reusing a worker leaks no RNG or policy state between tasks
def test_same_snapshot_and_seed_give_same_actions_when_a_worker_is_reused():
    policy = PolicyNetwork()
    snapshot = _snapshot(policy)

    single_worker_pool = ProcessPoolExecutor(max_workers=1, initializer=init_worker)

    try:
        first = collect_trajectories_parallel(single_worker_pool, snapshot, batch_size=1, seed_start=45)[0]

        # An unrelated episode in between advances the worker's RNG and environment
        collect_trajectories_parallel(single_worker_pool, snapshot, batch_size=1, seed_start=46)

        second = collect_trajectories_parallel(single_worker_pool, snapshot, batch_size=1, seed_start=45)[0]
    finally:
        single_worker_pool.shutdown(wait=True)

    assert first["actions"][:SEEDING_WINDOW] == second["actions"][:SEEDING_WINDOW]
    assert first["rewards"][:SEEDING_WINDOW] == pytest.approx(second["rewards"][:SEEDING_WINDOW], rel=2e-2)


# Test that different seeds do not accidentally produce identical action sequences
def test_different_seeds_produce_different_actions(pool):
    policy = PolicyNetwork()
    snapshot = _snapshot(policy)

    trajectory_a = collect_trajectories_parallel(pool, snapshot, batch_size=1, seed_start=50)[0]
    trajectory_b = collect_trajectories_parallel(pool, snapshot, batch_size=1, seed_start=51)[0]

    assert trajectory_a["actions"] != trajectory_b["actions"]


# Test that worker environments are isolated
def test_worker_environments_are_isolated(pool):
    num_tasks = 8

    # Submit every task before waiting on any result so the pool can spread them across workers
    futures = [pool.submit(_worker_identity) for _ in range(num_tasks)]
    identities = [future.result() for future in futures]

    env_id_by_pid = {}

    for pid, env_id in identities:
        if pid in env_id_by_pid:
            # The same worker process must always report the same (reused) environment
            assert env_id_by_pid[pid] == env_id
        else:
            env_id_by_pid[pid] = env_id

    # With more tasks than workers, more than one worker process must have been used
    assert len(env_id_by_pid) > 1
