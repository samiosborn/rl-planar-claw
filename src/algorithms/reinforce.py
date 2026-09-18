# src/algorithms/reinforce.py

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import config.simulation as CONFIG


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Hidden layer
        self.hidden = nn.Linear(CONFIG.OBSERVATION_DIM, 64)

        # Output layer
        self.output = nn.Linear(64, len(CONFIG.JOINTS) * len(CONFIG.JOINT_ACTION_VELOCITIES))


    # Forward pass
    # Single state [OBSERVATION_DIM] returns logits [NUM_JOINTS, NUM_ACTIONS]
    # Batch of T states [T, OBSERVATION_DIM] returns logits [T, NUM_JOINTS, NUM_ACTIONS]
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Hidden layer (tanh activation)
        hidden = torch.tanh(self.hidden(state))

        # Output (logits)
        output = self.output(hidden)

        # Reshape into one categorical distribution per joint
        # Preserves any leading batch dimension
        logits = output.reshape(
            *state.shape[:-1],
            len(CONFIG.JOINTS),
            len(CONFIG.JOINT_ACTION_VELOCITIES),
        )

        return logits


# Sample action from policy for a single state
def sample_action(policy: PolicyNetwork, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Get policy logits
    logits = policy.forward(state)

    # Convert into probabilities
    probs = torch.softmax(logits, dim=-1)

    # Create categorical distribution for each joint
    distribution = Categorical(probs=probs)

    # Sample one action for each joint
    actions = distribution.sample()

    # Log probability of joint action
    log_prob = distribution.log_prob(actions).sum()

    return actions, log_prob


# Score already-chosen actions under the current policy
# Used to recompute log pi_theta(a_t | s_t) with autograd from stored trajectory data
# states: [T, OBSERVATION_DIM], actions: [T, NUM_JOINTS]
# Returns log pi(a_t | s_t) for each timestep: [T]
def compute_action_log_probs(policy: PolicyNetwork, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    # Logits for every timestep in one forward pass: [T, NUM_JOINTS, NUM_ACTIONS]
    logits = policy.forward(states)

    # Categorical distribution per joint, per timestep
    distribution = Categorical(logits=logits)

    # Log probability of each joint's chosen action: [T, NUM_JOINTS]
    joint_log_probs = distribution.log_prob(actions)

    # log pi(a_t | s_t) = sum_j log pi(a_t,j | s_t): [T]
    return joint_log_probs.sum(dim=-1)


# Compute returns-to-go
def compute_returns(rewards, gamma):
    # Initialise
    G = 0.0
    returns = []

    # Compute return-to-go backwards
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.append(G)

    # Restore chronological order
    returns.reverse()

    return returns


# Compute policy loss from per-timestep log-probabilities and returns-to-go
def compute_policy_loss(log_probs: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    # Monte Carlo policy objective (J) = sum_t G_t log pi(a_t | s_t)
    policy_objective = (log_probs * returns).sum()

    # Gradient descent on negative objective = gradient ascent on objective
    return -policy_objective


# Compute the REINFORCE loss for one complete trajectory
# L_i = -sum_t G_i,t log pi_theta(a_i,t | s_i,t)
# Recomputes log pi_theta(a_t | s_t) from the stored states and actions with autograd enabled
# So the loss is differentiable with respect to the current (live) policy parameters
def compute_trajectory_loss(policy: PolicyNetwork, trajectory: dict, gamma: float) -> torch.Tensor:
    # Returns-to-go for this trajectory
    returns = torch.tensor(
        compute_returns(trajectory["rewards"], gamma),
        dtype=torch.float32,
    )

    # Stored states and actions as tensors, via a single numpy array first
    # Converting a list of arrays straight to a tensor is far slower
    states = torch.tensor(np.array(trajectory["states"]), dtype=torch.float32)
    actions = torch.tensor(np.array(trajectory["actions"]), dtype=torch.long)

    # Recompute log pi_theta(a_t | s_t) for every timestep in one forward pass
    log_probs = compute_action_log_probs(policy, states, actions)

    return compute_policy_loss(log_probs, returns)


# Sequential debug path, without multiprocessing
# Useful for comparing against the parallel path
def run_episode(env, policy: PolicyNetwork, gamma: float):
    # Local import avoids a circular import
    # src.rollout imports sample_action from this module
    from src.rollout import sample_episode

    trajectory = sample_episode(env, policy)
    loss = compute_trajectory_loss(policy, trajectory, gamma)

    return trajectory, loss


# Train policy from a batch of complete trajectories (Monte Carlo REINFORCE)
# Trajectories are sampled in parallel across worker processes
# Gradient estimator: -(1/N) sum_i sum_t G_i,t grad log pi_theta(a_i,t | s_i,t)
# Gradient descent on this negative objective corresponds to policy-gradient ascent
def train_batch(pool, policy: PolicyNetwork, optimiser, gamma: float, batch_size: int, seed_start: int):
    # Local import avoids a circular import
    # src.parallel_rollout imports src.rollout, which imports sample_action from this module
    from src.parallel_rollout import collect_trajectories_parallel

    # Clear gradients once for the whole batch
    optimiser.zero_grad()

    # Freeze one explicit CPU copy of the current policy parameters
    # Every trajectory in this batch is sampled from this exact snapshot
    # This keeps the batch strictly on-policy
    policy_state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in policy.state_dict().items()
    }

    # Sample batch_size complete trajectories in parallel, under torch.no_grad()
    trajectories = collect_trajectories_parallel(
        pool,
        policy_state_dict,
        batch_size,
        seed_start,
    )

    # Initialise per-trajectory diagnostics
    losses = []
    episode_lengths = []
    undiscounted_returns = []
    discounted_returns = []

    # Recompute log pi_theta(a_t | s_t) with autograd and accumulate gradients
    for trajectory in trajectories:
        # Recompute the REINFORCE loss for this trajectory against the current (live) policy
        loss = compute_trajectory_loss(policy, trajectory, gamma)

        # Average trajectory losses by backpropagating loss / N for each trajectory
        # PyTorch accumulates the resulting gradients
        (loss / len(trajectories)).backward()

        # Record diagnostics
        losses.append(loss.item())
        episode_lengths.append(trajectory["episode_length"])
        undiscounted_returns.append(sum(trajectory["rewards"]))
        discounted_returns.append(compute_returns(trajectory["rewards"], gamma)[0])

    # Gradient norm, computed after all trajectory gradients have accumulated
    gradient_norm_squared = 0.0

    for parameter in policy.parameters():
        if parameter.grad is not None:
            gradient_norm_squared += parameter.grad.pow(2).sum().item()

    # Update policy parameters exactly once for the batch
    optimiser.step()

    # Debug outputs
    return {
        "num_episodes": len(trajectories),
        "episode_lengths": episode_lengths,
        "mean_loss": sum(losses) / len(trajectories),
        "mean_undiscounted_return": sum(undiscounted_returns) / len(trajectories),
        "mean_discounted_return": sum(discounted_returns) / len(trajectories),
        "min_discounted_return": min(discounted_returns),
        "max_discounted_return": max(discounted_returns),
        "gradient_norm": gradient_norm_squared ** 0.5,
    }
