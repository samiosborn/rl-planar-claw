# tests/test_env.py

import numpy as np
import pytest

import config.simulation as CONFIG
from src.env import PlanarClawEnv


# Test reset observation
def test_reset_observation():
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Reset environment
        observation = env.reset()

        # Check observation
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (CONFIG.OBSERVATION_DIM,)
        assert observation.dtype == np.float32
        assert np.all(np.isfinite(observation))

    finally:
        # Close environment
        env.close()


# Test step return values
def test_step():
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Reset environment
        env.reset()

        # Hold all joints
        actions = [1] * len(CONFIG.JOINTS)

        # Step environment
        next_state, reward, terminated, truncated = env.step(actions)

        # Check returned transition
        assert isinstance(next_state, np.ndarray)
        assert next_state.shape == (CONFIG.OBSERVATION_DIM,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    finally:
        # Close environment
        env.close()


# Test invalid number of actions
def test_invalid_action_count():
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Too few actions
        actions = [1] * (len(CONFIG.JOINTS) - 1)

        # Check error
        with pytest.raises(ValueError):
            env.step(actions)

    finally:
        # Close environment
        env.close()


# Test invalid action index
def test_invalid_action_index():
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Create actions with one invalid index
        actions = [1] * len(CONFIG.JOINTS)
        actions[0] = len(CONFIG.JOINT_ACTION_VELOCITIES)

        # Check error
        with pytest.raises(ValueError):
            env.step(actions)

    finally:
        # Close environment
        env.close()


# Test invalid action type
def test_invalid_action_type():
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Create actions with one invalid type
        actions = [1] * len(CONFIG.JOINTS)
        actions[0] = 1.0

        # Check error
        with pytest.raises(TypeError):
            env.step(actions)

    finally:
        # Close environment
        env.close()


# Test episode truncation
def test_episode_truncation():
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Reset environment
        env.reset()

        # Hold all joints
        actions = [1] * len(CONFIG.JOINTS)

        # Run until final allowed step
        for _ in range(CONFIG.MAX_EPISODE_STEPS):
            _, _, terminated, truncated = env.step(actions)

            # Episode should not terminate through failure
            assert not terminated

        # Check truncation
        assert truncated
        assert env.step_count == CONFIG.MAX_EPISODE_STEPS

    finally:
        # Close environment
        env.close()


# Test reset statistics
def test_reset_statistics():
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Reset environment
        env.reset()

        # Take one step
        actions = [1] * len(CONFIG.JOINTS)
        env.step(actions)

        # Reset environment
        env.reset()

        # Check statistics
        assert env.step_count == 0
        assert env.success_steps == 0

    finally:
        # Close environment
        env.close()
