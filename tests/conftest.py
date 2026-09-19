# tests/conftest.py

from concurrent.futures import ProcessPoolExecutor

import pytest

from src.parallel_rollout import init_worker


# Module-scoped pool to avoid process start-up per test; a fixed worker count keeps it independent of REINFORCE_NUM_WORKERS
@pytest.fixture(scope="module")
def pool():
    executor = ProcessPoolExecutor(max_workers=4, initializer=init_worker)

    try:
        yield executor
    finally:
        executor.shutdown(wait=True)
