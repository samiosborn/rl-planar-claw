# tests/test_ppo.py

import torch
from torch.distributions import Categorical

import config.simulation as SIM_CONFIG
from src.algorithms.ppo import (
    PolicyNetwork,
    ValueNetwork,
    compute_action_log_probs,
    compute_gae,
    compute_values,
    sample_action,
    compute_probability_ratios,
    compute_value_loss, 
    optimise_epoch, 
)


# Test critic output for a batch of states
def test_value_network_batch_output():
    value_network = ValueNetwork()

    states = torch.randn(5, SIM_CONFIG.OBSERVATION_DIM)

    values = value_network(states)

    assert values.shape == (5,)


# Test critic output for a single state
def test_value_network_single_output():
    value_network = ValueNetwork()

    state = torch.randn(SIM_CONFIG.OBSERVATION_DIM)

    value = value_network(state)

    assert value.shape == ()


# Test sampled action shape
def test_sample_action_shape():
    policy = PolicyNetwork()

    state = torch.randn(SIM_CONFIG.OBSERVATION_DIM)

    actions, log_prob = sample_action(policy, state)

    assert actions.shape == (len(SIM_CONFIG.JOINTS),)
    assert log_prob.shape == ()


# Test complete-action log-probability is the sum across joints
def test_sample_action_log_prob():
    policy = PolicyNetwork()

    state = torch.randn(SIM_CONFIG.OBSERVATION_DIM)

    actions, log_prob = sample_action(policy, state)

    logits = policy(state)
    distribution = Categorical(logits=logits)

    expected_log_prob = distribution.log_prob(actions).sum()

    assert torch.allclose(log_prob, expected_log_prob)


# Test action log-probabilities for a batch
def test_compute_action_log_probs():
    policy = PolicyNetwork()

    states = torch.randn(5, SIM_CONFIG.OBSERVATION_DIM)

    logits = policy(states)
    distribution = Categorical(logits=logits)

    actions = distribution.sample()

    log_probs = compute_action_log_probs(
        policy,
        states,
        actions,
    )

    expected_log_probs = distribution.log_prob(actions).sum(dim=-1)

    assert log_probs.shape == (5,)
    assert torch.allclose(log_probs, expected_log_probs)


# Test critic values for a batch of states
def test_compute_values():
    value_network = ValueNetwork()

    states = torch.randn(5, SIM_CONFIG.OBSERVATION_DIM)

    values = compute_values(value_network, states)

    assert values.shape == (5,)


# Test GAE with lambda zero equals TD residuals
def test_compute_gae_lambda_zero():
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 1.0])
    final_value = torch.tensor(1.5)

    advantages = compute_gae(
        rewards,
        values,
        final_value,
        gamma=0.9,
        gae_lambda=0.0,
    )

    expected = torch.tensor([
        1.0 + 0.9 * 1.0 - 0.5,
        2.0 + 0.9 * 1.5 - 1.0,
    ])

    assert torch.allclose(advantages, expected)


# Test GAE backwards recursion
def test_compute_gae_backwards():
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 1.0])
    final_value = torch.tensor(1.5)

    gamma = 0.9
    gae_lambda = 0.8

    advantages = compute_gae(
        rewards,
        values,
        final_value,
        gamma,
        gae_lambda,
    )

    delta_1 = 2.0 + 0.9 * 1.5 - 1.0
    advantage_1 = delta_1

    delta_0 = 1.0 + 0.9 * 1.0 - 0.5
    advantage_0 = delta_0 + 0.9 * 0.8 * advantage_1

    expected = torch.tensor([
        advantage_0,
        advantage_1,
    ])

    assert torch.allclose(advantages, expected)


# Test identical policies give probability ratio one
def test_compute_probability_ratios_identity():
    log_probs = torch.tensor([-0.5, -1.2, -2.0])

    ratios = compute_probability_ratios(
        log_probs,
        log_probs,
    )

    expected = torch.ones(3)

    assert torch.allclose(ratios, expected)


# Test probability ratios from log-probabilities
def test_compute_probability_ratios():
    current_log_probs = torch.log(
        torch.tensor([0.4, 0.6])
    )

    old_log_probs = torch.log(
        torch.tensor([0.2, 0.3])
    )

    ratios = compute_probability_ratios(
        current_log_probs,
        old_log_probs,
    )

    expected = torch.tensor([2.0, 2.0])

    assert torch.allclose(ratios, expected)


# Test critic value loss
def test_compute_value_loss():
    values = torch.tensor([1.0, 2.0, 3.0])
    value_targets = torch.tensor([2.0, 2.0, 5.0])

    loss = compute_value_loss(
        values,
        value_targets,
    )

    # ((1 - 2)^2 + (2 - 2)^2 + (3 - 5)^2) / 3
    expected = torch.tensor(5.0 / 3.0)

    assert torch.allclose(loss, expected)


# Test one PPO epoch updates actor and critic parameters
def test_optimise_epoch_updates_parameters():
    policy = PolicyNetwork()
    value_network = ValueNetwork()

    policy_optimiser = torch.optim.Adam(policy.parameters(), lr=1e-3)
    value_optimiser = torch.optim.Adam(value_network.parameters(), lr=1e-3)

    states = torch.randn(5, SIM_CONFIG.OBSERVATION_DIM)

    with torch.no_grad():
        logits = policy(states)
        distribution = Categorical(logits=logits)
        actions = distribution.sample()
        old_log_probs = distribution.log_prob(actions).sum(dim=-1)

    batch = {
        "states": states,
        "actions": actions,
        "old_log_probs": old_log_probs,
        "advantages": torch.ones(5),
        "value_targets": torch.ones(5),
    }

    old_policy_parameters = [
        parameter.detach().clone()
        for parameter in policy.parameters()
    ]

    old_value_parameters = [
        parameter.detach().clone()
        for parameter in value_network.parameters()
    ]

    optimise_epoch(
        policy,
        value_network,
        policy_optimiser,
        value_optimiser,
        batch,
        clip_epsilon=0.2,
    )

    assert any(
        not torch.allclose(old, new)
        for old, new in zip(old_policy_parameters, policy.parameters())
    )

    assert any(
        not torch.allclose(old, new)
        for old, new in zip(old_value_parameters, value_network.parameters())
    )

