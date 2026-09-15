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

