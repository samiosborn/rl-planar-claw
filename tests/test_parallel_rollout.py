# tests/test_parallel_rollout.py

import config.simulation as CONFIG
from src.algorithms.reinforce import PolicyNetwork
from src.parallel_rollout import _worker_identity, collect_trajectories_parallel


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

    # The shared test pool has 4 workers
    # Request fewer trajectories than that, so some workers remain idle
    trajectories = collect_trajectories_parallel(pool, _snapshot(policy), batch_size=2, seed_start=20)

    assert len(trajectories) == 2
    assert all(trajectory["episode_length"] == CONFIG.MAX_EPISODE_STEPS for trajectory in trajectories)


# Test that a final partial batch of size 1 still works
def test_final_partial_batch_of_one(pool):
    policy = PolicyNetwork()

    trajectories = collect_trajectories_parallel(pool, _snapshot(policy), batch_size=1, seed_start=30)

    assert len(trajectories) == 1
    assert trajectories[0]["episode_length"] == CONFIG.MAX_EPISODE_STEPS


# Test that an identical policy snapshot and identical seed reproduce identical sampled actions
def test_reproducible_with_same_snapshot_and_seed(pool):
    policy = PolicyNetwork()
    snapshot = _snapshot(policy)

    trajectory_a = collect_trajectories_parallel(pool, snapshot, batch_size=1, seed_start=40)[0]
    trajectory_b = collect_trajectories_parallel(pool, snapshot, batch_size=1, seed_start=40)[0]

    assert trajectory_a["actions"] == trajectory_b["actions"]
    assert trajectory_a["rewards"] == trajectory_b["rewards"]


# Test that different seeds do not accidentally produce identical action sequences
def test_different_seeds_produce_different_actions(pool):
    policy = PolicyNetwork()
    snapshot = _snapshot(policy)

    trajectory_a = collect_trajectories_parallel(pool, snapshot, batch_size=1, seed_start=50)[0]
    trajectory_b = collect_trajectories_parallel(pool, snapshot, batch_size=1, seed_start=51)[0]

    assert trajectory_a["actions"] != trajectory_b["actions"]


# Test that worker environments are isolated
# Each worker process reuses exactly one environment across its own tasks
# Distinct workers are genuinely separate processes
def test_worker_environments_are_isolated(pool):
    num_tasks = 8

    # Submit every task before waiting on any result
    # This gives the pool a chance to use more than one of its worker processes
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
