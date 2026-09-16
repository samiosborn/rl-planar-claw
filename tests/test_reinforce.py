# tests/test_reinforce.py

import math

import torch

import config.simulation as CONFIG
from src.algorithms.reinforce import PolicyNetwork, sample_action, train_episode
from src.env import PlanarClawEnv


# Test policy output
def test_policy_output():
    # Create policy
    policy = PolicyNetwork()

    # Create dummy state of zeros
    state = torch.zeros(CONFIG.OBSERVATION_DIM)

    # Forward pass
    logits = policy.forward(state)

    # Check output shape
    assert logits.shape == (
        len(CONFIG.JOINTS),
        len(CONFIG.JOINT_ACTION_VELOCITIES),
    )


# Test action sampling
def test_sample_action():
    # Create policy
    policy = PolicyNetwork()

    # Create dummy state
    state = torch.zeros(CONFIG.OBSERVATION_DIM)

    # Sample action
    actions, log_prob = sample_action(policy, state)

    # Check action shape
    assert actions.shape == (len(CONFIG.JOINTS),)

    # Check log probability is a scalar
    assert log_prob.shape == torch.Size([])

    # Check actions are valid
    assert actions.dtype == torch.long
    assert torch.all(actions >= 0)
    assert torch.all(actions < len(CONFIG.JOINT_ACTION_VELOCITIES))


# Test training on a complete episode
def test_train_episode():
    # Initialise
    env = PlanarClawEnv(gui=False)

    try:
        # Use reproducible policy initialisation and action sampling
        with torch.random.fork_rng():
            torch.manual_seed(0)
            policy = PolicyNetwork()
            optimiser = torch.optim.Adam(policy.parameters(), lr=0.001)

            # Copy parameters before training
            parameters_before = [parameter.detach().clone() for parameter in policy.parameters()]

            # Train on one episode
            diagnostics = train_episode(env, policy, optimiser, CONFIG.GAMMA)

        # Check episode length and diagnostics
        assert env.step_count == CONFIG.MAX_EPISODE_STEPS
        assert diagnostics["episode_length"] == CONFIG.MAX_EPISODE_STEPS
        assert math.isfinite(diagnostics["loss"])
        assert math.isfinite(diagnostics["gradient_norm"])
        assert diagnostics["gradient_norm"] >= 0.0

        # Check that the optimiser updated at least one parameter
        assert any(
            not torch.equal(before, after)
            for before, after in zip(parameters_before, policy.parameters())
        )

    finally:
        # Close environment
        env.close()
