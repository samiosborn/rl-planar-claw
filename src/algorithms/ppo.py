# src/algorithms/ppo.py

import torch
import torch.nn as nn
from torch.distributions import Categorical

import config.simulation as SIM_CONFIG
from src.parallel_rollout import collect_trajectories_parallel


# Actor
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Hidden layer
        self.hidden = nn.Linear(SIM_CONFIG.OBSERVATION_DIM, 64)

        # Output layer
        self.output = nn.Linear(
            64,
            len(SIM_CONFIG.JOINTS) * len(SIM_CONFIG.JOINT_ACTION_VELOCITIES),
        )


    # Forward pass
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Hidden layer (tanh activation)
        hidden = torch.tanh(self.hidden(state))

        # Output (logits)
        output = self.output(hidden)

        # Reshape into one categorical distribution per joint
        logits = output.reshape(
            *state.shape[:-1],
            len(SIM_CONFIG.JOINTS),
            len(SIM_CONFIG.JOINT_ACTION_VELOCITIES),
        )

        return logits


# Critic
class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Hidden layer
        self.hidden = nn.Linear(SIM_CONFIG.OBSERVATION_DIM, 64)

        # Output layer
        self.output = nn.Linear(64, 1)


    # Forward pass
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Hidden layer (tanh activation)
        hidden = torch.tanh(self.hidden(state))

        # Output state value
        value = self.output(hidden)

        # Remove final dimension of size 1
        value = value.squeeze(-1)

        return value


# Sample action from policy for a single state
def sample_action(
    policy: PolicyNetwork,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute action logits
    logits = policy.forward(state)

    # Construct one categorical distribution per joint
    distribution = Categorical(logits=logits)

    # Sample one action per joint
    actions = distribution.sample()

    # Compute log-probability of each sampled joint action
    joint_log_probs = distribution.log_prob(actions)

    # Sum joint log-probabilities for the complete action
    log_prob = joint_log_probs.sum()

    return actions, log_prob


# Compute log-probabilities of stored actions, states
def compute_action_log_probs(
    policy: PolicyNetwork,
    states: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    # Compute action logits
    logits = policy.forward(states)

    # Construct one categorical distribution per joint
    distribution = Categorical(logits=logits)

    # Compute log-probability of each stored joint action
    joint_log_probs = distribution.log_prob(actions)

    # Sum joint log-probabilities for each complete action
    log_probs = joint_log_probs.sum(dim=-1)

    return log_probs


# Compute state values for a trajectory of states
def compute_values(
    value_network: ValueNetwork,
    states: torch.Tensor,
) -> torch.Tensor:
    # Compute the value estimate for each state
    values = value_network.forward(states)

    return values


# Compute generalised advantage estimates
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    final_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    # Initialise advantages
    advantages = torch.zeros_like(rewards)

    # Initialise next advantage
    next_advantage = torch.tensor(
        0.0,
        dtype=rewards.dtype,
        device=rewards.device,
    )

    # Bootstrap from final state value
    next_value = final_value

    # Compute advantages backwards through trajectory
    for t in reversed(range(len(rewards))):
        # TD residual
        delta = rewards[t] + gamma * next_value - values[t]

        # Compute GAE (recursively)
        advantage = delta + gamma * gae_lambda * next_advantage

        # Store advantage
        advantages[t] = advantage

        # Move back one timestep
        next_advantage = advantage
        next_value = values[t]

    return advantages


# Compute value targets from GAE advantages
def compute_value_targets(
    advantages: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    # Add baseline values back to advantages
    value_targets = advantages + values

    return value_targets


# Compute PPO probability ratios
def compute_probability_ratios(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> torch.Tensor:
    # Difference in log probability
    log_ratio = current_log_probs - old_log_probs

    # Convert log ratio to probability ratio
    ratios = torch.exp(log_ratio)

    return ratios


# Compute PPO clipped policy loss
def compute_policy_loss(
    ratios: torch.Tensor,
    advantages: torch.Tensor,
    epsilon_clip: float,
) -> torch.Tensor:
    # Unclipped surrogate
    unclipped_surrogate = ratios * advantages

    # Clip probability ratios
    clipped_ratios = torch.clamp(
        ratios,
        1.0 - epsilon_clip,
        1.0 + epsilon_clip,
    )

    # Clipped surrogate
    clipped_surrogate = clipped_ratios * advantages

    # PPO clipped objective
    clipped_objective = torch.minimum(
        unclipped_surrogate,
        clipped_surrogate,
    ).mean()

    # Convert maximisation objective into minimisation loss
    policy_loss = -clipped_objective

    return policy_loss


# Compute value loss (MSE)
def compute_value_loss(
    values: torch.Tensor,
    value_targets: torch.Tensor,
) -> torch.Tensor:
    # Squared value errors
    squared_errors = (values - value_targets) ** 2

    # Mean squared error
    value_loss = squared_errors.mean()

    return value_loss


# Process a trajectory into training data format
def process_trajectory(
    value_network: ValueNetwork,
    trajectory: dict,
    gamma: float,
    gae_lambda: float,
) -> dict:
    # Convert to tensor
    states = torch.tensor(trajectory["states"], dtype=torch.float32)
    actions = torch.tensor(trajectory["actions"], dtype=torch.long)
    rewards = torch.tensor(trajectory["rewards"], dtype=torch.float32)
    final_state = torch.tensor(trajectory["final_state"], dtype=torch.float32)

    with torch.no_grad():
        # Values for rollout states
        values = compute_values(
            value_network,
            states,
        )

        # Final state value as bootstrap
        final_value = compute_values(
            value_network,
            final_state,
        )

        # Advantages via GAE
        advantages = compute_gae(
            rewards,
            values,
            final_value,
            gamma,
            gae_lambda,
        )

        # Value targets for critic
        value_targets = compute_value_targets(
            advantages,
            values,
        )

    return {
        "states": states,
        "actions": actions,
        "old_log_probs": torch.tensor(
            trajectory["log_probs"],
            dtype=torch.float32,
        ),
        "advantages": advantages,
        "value_targets": value_targets,
    }


# Compute PPO losses for one optimisation epoch from the training data
def compute_ppo_losses(
    policy: PolicyNetwork,
    value_network: ValueNetwork,
    batch: dict,
    epsilon_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Current policy log-probabilities
    current_log_probs = compute_action_log_probs(
        policy,
        batch["states"],
        batch["actions"],
    )

    # Current values (critic estimate)
    current_values = compute_values(
        value_network,
        batch["states"],
    )

    # Probability ratios
    ratios = compute_probability_ratios(
        current_log_probs,
        batch["old_log_probs"],
    )

    # Actor loss (clipped objective)
    policy_loss = compute_policy_loss(
        ratios,
        batch["advantages"],
        epsilon_clip,
    )

    # Critic loss (MSE)
    value_loss = compute_value_loss(
        current_values,
        batch["value_targets"],
    )

    return policy_loss, value_loss


# Optimise actor and critic for one PPO epoch
def optimise_epoch(
    policy: PolicyNetwork,
    value_network: ValueNetwork,
    policy_optimiser,
    value_optimiser,
    batch: dict,
    epsilon_clip: float,
) -> tuple[float, float]:
    # Clear actor gradients
    policy_optimiser.zero_grad()

    # Clear critic gradients
    value_optimiser.zero_grad()

    # Current PPO losses for actor and critic
    policy_loss, value_loss = compute_ppo_losses(
        policy,
        value_network,
        batch,
        epsilon_clip,
    )

    # Actor gradients
    policy_loss.backward()

    # Critic gradients
    value_loss.backward()

    # Update actor parameters
    policy_optimiser.step()

    # Update critic parameters
    value_optimiser.step()

    return policy_loss.item(), value_loss.item()


# Process rollout trajectories into one PPO training batch
def process_trajectories(
    value_network: ValueNetwork,
    trajectories: list[dict],
    gamma: float,
    gae_lambda: float,
) -> dict:
    # Process each trajectory independently
    processed_trajectories = [
        process_trajectory(
            value_network,
            trajectory,
            gamma,
            gae_lambda,
        )
        for trajectory in trajectories
    ]

    # Concatenate
    states = torch.cat(
        [trajectory["states"] for trajectory in processed_trajectories],
        dim=0,
    )

    actions = torch.cat(
        [trajectory["actions"] for trajectory in processed_trajectories],
        dim=0,
    )

    old_log_probs = torch.cat(
        [trajectory["old_log_probs"] for trajectory in processed_trajectories],
        dim=0,
    )

    advantages = torch.cat(
        [trajectory["advantages"] for trajectory in processed_trajectories],
        dim=0,
    )

    value_targets = torch.cat(
        [trajectory["value_targets"] for trajectory in processed_trajectories],
        dim=0,
    )

    return {
        "states": states,
        "actions": actions,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
        "value_targets": value_targets,
    }


# Train on one batch of rollouts
def train_batch(
    pool,
    policy: PolicyNetwork,
    value_network: ValueNetwork,
    policy_optimiser,
    value_optimiser,
    gamma: float,
    gae_lambda: float,
    epsilon_clip: float,
    num_trajectories: int,
    optimisation_epochs: int,
    seed_start: int,
) -> dict:
    # Freeze current policy parameters (for rollout workers)
    policy_state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in policy.state_dict().items()
    }

    # Collect trajectories from fixed policy
    trajectories = collect_trajectories_parallel(
        pool,
        policy_state_dict,
        num_trajectories,
        seed_start,
    )

    # Process trajectories into one fixed batch for training
    batch = process_trajectories(
        value_network,
        trajectories,
        gamma,
        gae_lambda,
    )

    # Track optimisation losses
    policy_losses = []
    value_losses = []

    # Optimise repeatedly using the same fixed batch
    for _ in range(optimisation_epochs):
        policy_loss, value_loss = optimise_epoch(
            policy,
            value_network,
            policy_optimiser,
            value_optimiser,
            batch,
            epsilon_clip,
        )

        policy_losses.append(policy_loss)
        value_losses.append(value_loss)

    # Collect rollout diagnostics
    episode_lengths = [
        trajectory["episode_length"]
        for trajectory in trajectories
    ]

    undiscounted_returns = [
        sum(trajectory["rewards"])
        for trajectory in trajectories
    ]

    return {
        "num_trajectories": len(trajectories),
        "num_transitions": len(batch["states"]),
        "episode_lengths": episode_lengths,
        "mean_policy_loss": sum(policy_losses) / len(policy_losses),
        "mean_value_loss": sum(value_losses) / len(value_losses),
        "mean_undiscounted_return": (
            sum(undiscounted_returns) / len(undiscounted_returns)
        ),
    }
