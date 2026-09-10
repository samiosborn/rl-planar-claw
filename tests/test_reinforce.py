# tests/test_reinforce.py

import torch

import config.simulation as CONFIG
from src.algorithms.reinforce import PolicyNetwork, sample_action


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
