# tests/test_rollout_reinforce.py

import math

import config.simulation as CONFIG

from src.algorithms.reinforce import PolicyNetwork, sample_action
from src.env import PlanarClawEnv
from src.rollout import rollout_episode


# Test trajectory structure
def test_rollout_episode():
    # Initialise
    env = PlanarClawEnv(gui=False)
    policy = PolicyNetwork()
    gamma = 0.99

    try:
        # Sample trajectory
        trajectory = rollout_episode(env, policy, sample_action, gamma)

        # Extract trajectory data
        states = trajectory["states"]
        actions = trajectory["actions"]
        rewards = trajectory["rewards"]
        angle_errors = trajectory["angle_errors"]

        # One action and reward per transition
        assert len(actions) == len(rewards)

        # State quantities include initial state
        assert len(states) == len(rewards) + 1
        assert len(angle_errors) == len(rewards) + 1

        # Episode contains exactly the allowed number of transitions
        assert len(rewards) == CONFIG.MAX_EPISODE_STEPS

        # Check trajectory fields
        assert set(trajectory) == {
            "states", "actions", "rewards", "angle_errors", "discounted_return",
        }

    finally:
        env.close()


# Test discounted episodic return
def test_rollout_discounted_return():
    # Initialise
    env = PlanarClawEnv(gui=False)
    policy = PolicyNetwork()
    gamma = 0.99

    try:
        # Sample trajectory
        trajectory = rollout_episode(env, policy, sample_action, gamma)

        # Compute expected discounted return
        expected_return = sum(
            (gamma ** t) * reward
            for t, reward in enumerate(trajectory["rewards"])
        )

        # Check returned value
        assert math.isclose(
            trajectory["discounted_return"],
            expected_return,
            rel_tol=1e-6,
        )

    finally:
        env.close()
        