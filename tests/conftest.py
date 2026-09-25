# tests/conftest.py

from concurrent.futures import ProcessPoolExecutor

import pytest

from src.algorithms.reinforce import PolicyNetwork, sample_action
from src.parallel_rollout import init_worker


# Module-scoped pool to avoid process start-up per test; a fixed worker count keeps it independent of config.reinforce.NUM_WORKERS
@pytest.fixture(scope="module")
def pool():
    executor = ProcessPoolExecutor(max_workers=4, initializer=init_worker, initargs=(PolicyNetwork, sample_action))

    try:
        yield executor
    finally:
        executor.shutdown(wait=True)
