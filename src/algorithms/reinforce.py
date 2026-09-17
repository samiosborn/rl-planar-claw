# src/algorithms/reinforce.py

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
    def forward(self, state: torch.Tensor) -> torch.Tensor: 
        # Hidden layer (tanh activation)
        hidden = torch.tanh(self.hidden(state))

        # Output (logits)
        output = self.output(hidden)

        # Reshape into one categorical distribution per joint
        logits = output.reshape(len(CONFIG.JOINTS), len(CONFIG.JOINT_ACTION_VELOCITIES)) 

        return logits


# Sample action from policy
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


# Compute policy loss
def compute_policy_loss(log_probs, returns): 
    # Convert into a single tensor
    log_probs = torch.stack(log_probs)
    returns = torch.tensor(returns, dtype=torch.float32)

    # Monte Carlo policy objective (J)
    policy_objective = (log_probs * returns).sum()

    # Gradient descent on negative objective = gradient ascent on objective
    return -policy_objective


# Sample one complete episode and compute its REINFORCE loss
def run_episode(env, policy, gamma):
    # Reset environment
    state = env.reset()

    # Initialise rewards and log-probabilities
    rewards = []
    log_probs = []

    # Reset episode
    done = False

    # Loop until done
    while not done:
        # Convert state into tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # Sample action from policy
        action, log_prob = sample_action(policy, state_tensor)

        # State transition following action (converted to list first)
        next_state, reward, done = env.step(action.tolist())

        # Append
        rewards.append(reward)
        log_probs.append(log_prob)

        # Update current state
        state = next_state

    # Compute returns
    returns = compute_returns(rewards, gamma)

    # Compute policy loss (sum over timesteps)
    loss = compute_policy_loss(log_probs, returns)

    # Debug outputs (loss kept as a tensor so it can still be backpropagated)
    return {
        "loss": loss,
        "episode_length": len(rewards),
        "undiscounted_return": sum(rewards),
        "discounted_return": returns[0],
    }


# Train policy from a batch of complete trajectories (Monte Carlo REINFORCE)
def train_batch(env, policy, optimiser, gamma, batch_size):
    # Clear gradients once for the whole batch
    optimiser.zero_grad()

    # Initialise per-episode diagnostics
    losses = []
    episode_lengths = []
    undiscounted_returns = []
    discounted_returns = []

    # Sample batch_size complete episodes
    for _ in range(batch_size):
        # Collect one trajectory and its REINFORCE loss
        episode = run_episode(env, policy, gamma)

        # Average trajectory losses by backpropagating loss / batch_size
        # for each trajectory; PyTorch accumulates the resulting gradients
        (episode["loss"] / batch_size).backward()

        # Record diagnostics
        losses.append(episode["loss"].item())
        episode_lengths.append(episode["episode_length"])
        undiscounted_returns.append(episode["undiscounted_return"])
        discounted_returns.append(episode["discounted_return"])

    # Gradient norm, computed after all trajectory gradients have accumulated
    gradient_norm_squared = 0.0

    for parameter in policy.parameters():
        if parameter.grad is not None:
            gradient_norm_squared += parameter.grad.pow(2).sum().item()

    # Update policy parameters exactly once for the batch
    optimiser.step()

    # Debug outputs
    return {
        "num_episodes": batch_size,
        "episode_lengths": episode_lengths,
        "mean_loss": sum(losses) / batch_size,
        "mean_undiscounted_return": sum(undiscounted_returns) / batch_size,
        "mean_discounted_return": sum(discounted_returns) / batch_size,
        "min_discounted_return": min(discounted_returns),
        "max_discounted_return": max(discounted_returns),
        "gradient_norm": gradient_norm_squared ** 0.5,
    }
