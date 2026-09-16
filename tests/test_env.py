# tests/test_env.py

import math

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
        next_state, reward, done = env.step(actions)

        # Check returned transition
        assert isinstance(next_state, np.ndarray)
        assert next_state.shape == (CONFIG.OBSERVATION_DIM,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

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


# Test episode length
def test_episode_length():
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Reset environment
        env.reset()

        # Hold all joints
        actions = [1] * len(CONFIG.JOINTS)

        # Run until final allowed step
        for step in range(1, CONFIG.MAX_EPISODE_STEPS + 1):
            _, _, done = env.step(actions)

            # Episode ends only on the final allowed step
            assert done == (step == CONFIG.MAX_EPISODE_STEPS)

        # Check final step count
        assert done
        assert env.step_count == CONFIG.MAX_EPISODE_STEPS

    finally:
        # Close environment
        env.close()


# Test large angle errors do not end an episode early
@pytest.mark.parametrize("angle_error", [-math.pi, -math.pi / 2, math.pi / 2, math.pi])
def test_large_angle_error(angle_error, monkeypatch):
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Hold angle error at a controlled value
        monkeypatch.setattr(env, "get_angle_error", lambda: angle_error)
        actions = [1] * len(CONFIG.JOINTS)

        # Run a complete episode
        for step in range(1, CONFIG.MAX_EPISODE_STEPS + 1):
            _, reward, done = env.step(actions)

            # Keep the same reward and episode length
            assert reward == -abs(angle_error)
            assert done == (step == CONFIG.MAX_EPISODE_STEPS)

        assert env.success_steps == 0

    finally:
        # Close environment
        env.close()


# Test success tracking does not end an episode early
def test_success_tracking(monkeypatch):
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Hold angle error within success tolerance
        monkeypatch.setattr(env, "get_angle_error", lambda: CONFIG.SUCCESS_TOLERANCE)
        actions = [1] * len(CONFIG.JOINTS)

        # Run a complete episode
        for step in range(1, CONFIG.MAX_EPISODE_STEPS + 1):
            _, _, done = env.step(actions)

            # Track success without ending early
            assert env.success_steps == step
            assert done == (step == CONFIG.MAX_EPISODE_STEPS)

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
