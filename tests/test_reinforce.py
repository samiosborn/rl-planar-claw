# tests/test_reinforce.py

import math

import torch

import config.simulation as CONFIG
from src.algorithms.reinforce import PolicyNetwork, run_episode, sample_action, train_batch
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


# Test sampling and scoring a single complete episode
def test_run_episode():
    # Initialise
    env = PlanarClawEnv(gui=False)
    policy = PolicyNetwork()

    try:
        # Collect one trajectory
        episode = run_episode(env, policy, CONFIG.GAMMA)

        # Episode contains exactly the allowed number of transitions
        assert env.step_count == CONFIG.MAX_EPISODE_STEPS
        assert episode["episode_length"] == CONFIG.MAX_EPISODE_STEPS

        # Loss is a scalar tensor that can be backpropagated
        assert episode["loss"].shape == torch.Size([])
        assert math.isfinite(episode["loss"].item())

    finally:
        env.close()


# Test training on a full batch of trajectories
def test_train_batch():
    # Initialise
    env = PlanarClawEnv(gui=False)
    batch_size = CONFIG.REINFORCE_BATCH_SIZE

    try:
        # Use reproducible policy initialisation and action sampling
        with torch.random.fork_rng():
            torch.manual_seed(0)
            policy = PolicyNetwork()
            optimiser = torch.optim.Adam(policy.parameters(), lr=0.001)

            # Copy parameters before training
            parameters_before = [parameter.detach().clone() for parameter in policy.parameters()]

            # Train on one batch of trajectories
            diagnostics = train_batch(env, policy, optimiser, CONFIG.GAMMA, batch_size)

        # Exactly N complete episodes were sampled for a full batch
        assert diagnostics["num_episodes"] == batch_size
        assert len(diagnostics["episode_lengths"]) == batch_size

        # Each episode still contains exactly CONFIG.MAX_EPISODE_STEPS transitions
        assert all(length == CONFIG.MAX_EPISODE_STEPS for length in diagnostics["episode_lengths"])

        # Diagnostics are finite
        assert math.isfinite(diagnostics["mean_loss"])
        assert math.isfinite(diagnostics["mean_discounted_return"])
        assert math.isfinite(diagnostics["gradient_norm"])
        assert diagnostics["gradient_norm"] >= 0.0

        # min/max bracket the mean discounted return
        assert diagnostics["min_discounted_return"] <= diagnostics["mean_discounted_return"]
        assert diagnostics["max_discounted_return"] >= diagnostics["mean_discounted_return"]

        # Optimiser parameters were updated after one batch
        assert any(
            not torch.equal(before, after)
            for before, after in zip(parameters_before, policy.parameters())
        )

    finally:
        # Close environment
        env.close()


# Test that the reported mean discounted return matches the sampled episodes
def test_train_batch_mean_return_matches_episodes():
    # Initialise
    env = PlanarClawEnv(gui=False)
    policy = PolicyNetwork()
    optimiser = torch.optim.Adam(policy.parameters(), lr=0.001)
    batch_size = 3

    try:
        # Re-sample the same number of episodes independently, using a policy
        # snapshot equivalent to the one train_batch starts from
        with torch.random.fork_rng():
            torch.manual_seed(1)
            reference_policy = PolicyNetwork()
            reference_returns = [
                run_episode(env, reference_policy, CONFIG.GAMMA)["discounted_return"]
                for _ in range(batch_size)
            ]

        # Train a freshly-seeded policy identically, and check the reported
        # mean discounted return equals the arithmetic mean of its own episodes
        with torch.random.fork_rng():
            torch.manual_seed(1)
            policy = PolicyNetwork()
            optimiser = torch.optim.Adam(policy.parameters(), lr=0.001)
            diagnostics = train_batch(env, policy, optimiser, CONFIG.GAMMA, batch_size)

        expected_mean_return = sum(reference_returns) / batch_size

        assert math.isclose(
            diagnostics["mean_discounted_return"],
            expected_mean_return,
            rel_tol=1e-5,
        )

    finally:
        env.close()


# Test that gradient accumulation over a batch matches averaging trajectory losses
def test_train_batch_gradient_matches_averaged_losses():
    # Use a batch size small enough to keep the test fast and deterministic
    batch_size = 2

    env = PlanarClawEnv(gui=False)

    try:
        # Reference: manually average two trajectory losses sampled under the
        # same random draws and backpropagate once
        with torch.random.fork_rng():
            torch.manual_seed(2)
            reference_policy = PolicyNetwork()

            episode_losses = [
                run_episode(env, reference_policy, CONFIG.GAMMA)["loss"]
                for _ in range(batch_size)
            ]
            averaged_loss = sum(episode_losses) / batch_size
            averaged_loss.backward()

            reference_gradients = [
                parameter.grad.detach().clone()
                for parameter in reference_policy.parameters()
            ]

        # train_batch: accumulate gradients via sequential (loss / N).backward() calls
        with torch.random.fork_rng():
            torch.manual_seed(2)
            policy = PolicyNetwork()
            optimiser = torch.optim.Adam(policy.parameters(), lr=0.001)
            train_batch(env, policy, optimiser, CONFIG.GAMMA, batch_size)

            batch_gradients = [
                parameter.grad.detach().clone()
                for parameter in policy.parameters()
            ]

        # The two gradient accumulation strategies must match (allowing for
        # floating-point summation order differences between backward() calls)
        for reference_grad, batch_grad in zip(reference_gradients, batch_gradients):
            assert torch.allclose(reference_grad, batch_grad, rtol=1e-4, atol=1e-3)

    finally:
        env.close()


# Test that exactly one optimiser step is taken per batch, not one per trajectory
def test_train_batch_single_optimiser_step(monkeypatch):
    env = PlanarClawEnv(gui=False)
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

    try:
        train_batch(env, policy, optimiser, CONFIG.GAMMA, batch_size)
        assert step_count == 1

    finally:
        env.close()
