# tests/conftest.py

from concurrent.futures import ProcessPoolExecutor

import pytest

from src.parallel_rollout import init_worker


# Shared worker pool, reused across the tests in a module to avoid paying process-startup cost per test
# A small fixed worker count keeps the test suite's resource usage independent of REINFORCE_NUM_WORKERS
@pytest.fixture(scope="module")
def pool():
    executor = ProcessPoolExecutor(max_workers=4, initializer=init_worker)

    try:
        yield executor
    finally:
        executor.shutdown(wait=True)
