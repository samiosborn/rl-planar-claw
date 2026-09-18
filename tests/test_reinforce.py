# tests/test_reinforce.py

import math

import torch
from torch.distributions import Categorical

import config.simulation as CONFIG
from src.algorithms.reinforce import (
    PolicyNetwork,
    compute_action_log_probs,
    compute_returns,
    compute_trajectory_loss,
    run_episode,
    sample_action,
    train_batch,
)
from src.env import PlanarClawEnv
from src.rollout import sample_episode


# Test policy output for a single state
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


# Test policy output for a batch of T states
def test_policy_output_batched():
    # Create policy
    policy = PolicyNetwork()

    # Create a batch of T dummy states
    T = 7
    states = torch.zeros(T, CONFIG.OBSERVATION_DIM)

    # Forward pass
    logits = policy.forward(states)

    # Check output shape
    assert logits.shape == (
        T,
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


# Test that the vectorised trajectory log-probabilities equal an independent reconstruction
# Reconstructed timestep-by-timestep using the same Categorical distribution
def test_compute_action_log_probs_matches_timestep_by_timestep():
    policy = PolicyNetwork()

    T = 5
    states = torch.randn(T, CONFIG.OBSERVATION_DIM)
    actions = torch.stack([
        torch.randint(0, len(CONFIG.JOINT_ACTION_VELOCITIES), (len(CONFIG.JOINTS),))
        for _ in range(T)
    ])

    # Vectorised: one forward pass for the whole trajectory
    vectorised_log_probs = compute_action_log_probs(policy, states, actions)
    assert vectorised_log_probs.shape == (T,)

    # Timestep-by-timestep reconstruction, matching the single-state formula
    # logits = policy(state)
    # log_prob = Categorical(logits=logits).log_prob(action).sum()
    for t in range(T):
        logits = policy.forward(states[t])
        distribution = Categorical(logits=logits)
        expected_log_prob = distribution.log_prob(actions[t]).sum()

        assert torch.allclose(vectorised_log_probs[t], expected_log_prob, atol=1e-6)


# Test that compute_trajectory_loss equals an independently reconstructed loss
# -sum_t G_t log pi(a_t | s_t)
def test_compute_trajectory_loss_matches_manual_reconstruction():
    policy = PolicyNetwork()
    gamma = 0.9

    trajectory = {
        "states": [[float(i)] * CONFIG.OBSERVATION_DIM for i in range(4)],
        "actions": [[0] * len(CONFIG.JOINTS) for _ in range(4)],
        "rewards": [-1.0, -0.5, -0.25, -0.1],
        "episode_length": 4,
    }

    loss = compute_trajectory_loss(policy, trajectory, gamma)

    # Manual reconstruction: sum_t G_t log pi(a_t | s_t), negated
    returns = compute_returns(trajectory["rewards"], gamma)
    expected_loss = 0.0

    for state, action, G in zip(trajectory["states"], trajectory["actions"], returns):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.long)

        logits = policy.forward(state_tensor)
        distribution = Categorical(logits=logits)
        log_prob = distribution.log_prob(action_tensor).sum()

        expected_loss += -G * log_prob.item()

    assert math.isclose(loss.item(), expected_loss, rel_tol=1e-5)


# Test the simple sequential (no-multiprocessing) debug path
def test_run_episode():
    # Initialise
    env = PlanarClawEnv(gui=False)
    policy = PolicyNetwork()

    try:
        # Collect one trajectory and its REINFORCE loss
        trajectory, loss = run_episode(env, policy, CONFIG.GAMMA)

        # Episode contains exactly the allowed number of transitions
        assert env.step_count == CONFIG.MAX_EPISODE_STEPS
        assert trajectory["episode_length"] == CONFIG.MAX_EPISODE_STEPS
        assert len(trajectory["states"]) == CONFIG.MAX_EPISODE_STEPS
        assert len(trajectory["actions"]) == CONFIG.MAX_EPISODE_STEPS
        assert len(trajectory["rewards"]) == CONFIG.MAX_EPISODE_STEPS

        # Loss is a scalar tensor that can be backpropagated
        assert loss.shape == torch.Size([])
        assert math.isfinite(loss.item())

    finally:
        env.close()


# Test training on a full batch of trajectories, sampled in parallel
def test_train_batch(pool):
    batch_size = CONFIG.REINFORCE_BATCH_SIZE

    # Use reproducible policy initialisation
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = PolicyNetwork()
        optimiser = torch.optim.Adam(policy.parameters(), lr=0.001)

        # Copy parameters before training
        parameters_before = [parameter.detach().clone() for parameter in policy.parameters()]

        # Train on one batch of trajectories
        diagnostics = train_batch(pool, policy, optimiser, CONFIG.GAMMA, batch_size, seed_start=1000)

    # Exactly N complete trajectories were sampled for a full batch
    assert diagnostics["num_episodes"] == batch_size
    assert len(diagnostics["episode_lengths"]) == batch_size

    # Each trajectory still contains exactly CONFIG.MAX_EPISODE_STEPS transitions
    assert all(length == CONFIG.MAX_EPISODE_STEPS for length in diagnostics["episode_lengths"])

    # Diagnostics are finite
    assert math.isfinite(diagnostics["mean_loss"])
    assert math.isfinite(diagnostics["mean_discounted_return"])
    assert math.isfinite(diagnostics["gradient_norm"])
    assert diagnostics["gradient_norm"] >= 0.0

    # min/max bracket the mean discounted return
    assert diagnostics["min_discounted_return"] <= diagnostics["mean_discounted_return"]
    assert diagnostics["max_discounted_return"] >= diagnostics["mean_discounted_return"]

    # Policy parameters were updated after one batch
    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, policy.parameters())
    )


# Test that a batch smaller than the pool's worker count trains correctly
def test_train_batch_smaller_than_worker_count(pool):
    policy = PolicyNetwork()
    optimiser = torch.optim.Adam(policy.parameters(), lr=0.001)

    # The shared test pool has 4 workers
    # Use a smaller batch
    diagnostics = train_batch(pool, policy, optimiser, CONFIG.GAMMA, batch_size=2, seed_start=2000)

    assert diagnostics["num_episodes"] == 2
    assert math.isfinite(diagnostics["gradient_norm"])


# Test that a final partial batch of size 1 trains correctly
def test_train_batch_final_partial_batch(pool):
    policy = PolicyNetwork()
    optimiser = torch.optim.Adam(policy.parameters(), lr=0.001)

    diagnostics = train_batch(pool, policy, optimiser, CONFIG.GAMMA, batch_size=1, seed_start=3000)

    assert diagnostics["num_episodes"] == 1
    assert math.isfinite(diagnostics["gradient_norm"])


# Test that train_batch's reported mean discounted return matches trajectories
# Sampled sequentially from the identical frozen snapshot and identical seeds
def test_train_batch_mean_return_matches_episodes(pool):
    policy = PolicyNetwork()
    optimiser = torch.optim.Adam(policy.parameters(), lr=0.001)
    batch_size = 3
    seed_start = 4000

    # Freeze the exact same snapshot train_batch will freeze internally
    policy_state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in policy.state_dict().items()
    }

    # Independently reconstruct the trajectories train_batch will sample
    # Sample sequentially from the identical snapshot and identical seeds
    # Mirror the worker task exactly, with one persistent policy object reloaded on every episode
    # A fresh PolicyNetwork() per episode would itself consume RNG state and break reproducibility
    reference_env = PlanarClawEnv(gui=False)
    sampling_policy = PolicyNetwork()
    try:
        reference_returns = []

        for episode_index in range(batch_size):
            torch.manual_seed(seed_start + episode_index)
            sampling_policy.load_state_dict(policy_state_dict)

            trajectory = sample_episode(reference_env, sampling_policy)
            reference_returns.append(compute_returns(trajectory["rewards"], CONFIG.GAMMA)[0])
    finally:
        reference_env.close()

    diagnostics = train_batch(pool, policy, optimiser, CONFIG.GAMMA, batch_size, seed_start)

    expected_mean_return = sum(reference_returns) / batch_size

    assert math.isclose(
        diagnostics["mean_discounted_return"],
        expected_mean_return,
        rel_tol=1e-5,
    )


# Test that gradient accumulation over a parallel batch matches averaging trajectory losses
# Sampled sequentially from the same snapshot and seeds
def test_train_batch_gradient_matches_averaged_losses(pool):
    batch_size = 2
    seed_start = 5000

    policy = PolicyNetwork()
    optimiser = torch.optim.Adam(policy.parameters(), lr=0.001)

    policy_state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in policy.state_dict().items()
    }

    # Reference: sample trajectories sequentially from the identical frozen snapshot and seeds
    # Mirror the worker task exactly, with one persistent policy object reloaded on every episode
    # See test_train_batch_mean_return_matches_episodes
    reference_env = PlanarClawEnv(gui=False)
    sampling_policy = PolicyNetwork()
    try:
        reference_trajectories = []

        for episode_index in range(batch_size):
            torch.manual_seed(seed_start + episode_index)
            sampling_policy.load_state_dict(policy_state_dict)

            reference_trajectories.append(sample_episode(reference_env, sampling_policy))
    finally:
        reference_env.close()

    # Average trajectory losses computed against the live (grad-tracking) policy
    # Exactly as train_batch does internally
    reference_losses = [
        compute_trajectory_loss(policy, trajectory, CONFIG.GAMMA)
        for trajectory in reference_trajectories
    ]
    averaged_loss = sum(reference_losses) / batch_size
    averaged_loss.backward()

    reference_gradients = [parameter.grad.detach().clone() for parameter in policy.parameters()]

    # train_batch: accumulate gradients from trajectories collected in parallel
    train_batch(pool, policy, optimiser, CONFIG.GAMMA, batch_size, seed_start)

    batch_gradients = [parameter.grad.detach().clone() for parameter in policy.parameters()]

    # The two gradient accumulation strategies must match
    # Allowing for floating-point summation order differences between backward() calls
    for reference_grad, batch_grad in zip(reference_gradients, batch_gradients):
        assert torch.allclose(reference_grad, batch_grad, rtol=1e-4, atol=1e-3)


# Test that exactly one optimiser step is taken per batch, not one per trajectory
def test_train_batch_single_optimiser_step(pool, monkeypatch):
    policy = PolicyNetwork()
    optimiser = torch.optim.Adam(policy.parameters(), lr=0.001)
    batch_size = 4

    step_count = 0
    original_step = optimiser.step

    def counting_step(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimiser, "step", counting_step)

    train_batch(pool, policy, optimiser, CONFIG.GAMMA, batch_size, seed_start=6000)

    assert step_count == 1
