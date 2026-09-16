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


# Train policy from one episode
def train_episode(env, policy, optimiser, gamma): 
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

    # Compute policy loss
    loss = compute_policy_loss(log_probs, returns)

    # Clear previous episode gradients
    optimiser.zero_grad()

    # Compute gradients
    loss.backward()

    # Gradient norm
    gradient_norm_squared = 0.0

    for parameter in policy.parameters(): 
        if parameter.grad is not None: 
            gradient_norm_squared += parameter.grad.pow(2).sum().item()

    # Update policy parameters
    optimiser.step()

    # Debug outputs
    return {
        "loss": loss.item(), 
        "episode_length": len(rewards), 
        "undiscounted_return": sum(rewards), 
        "discounted_return": returns[0], 
        "gradient_norm": gradient_norm_squared ** 0.5,
    }
