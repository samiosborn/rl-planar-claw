# src/rollout.py

import torch

from src.algorithms.reinforce import sample_action


# Sample one complete episode under torch.no_grad() (lean training sampler)
# Returns only the training-relevant trajectory data
# No log-probabilities, no autograd graph, no returns
def sample_episode(env, policy):
    # Reset environment
    state = env.reset()

    # Initialise trajectory data (only the T states that generated actions)
    states = []
    actions = []
    rewards = []

    # Reset episode
    done = False

    # Sample the whole episode without building an autograd graph
    with torch.no_grad():
        while not done:
            # Convert state into tensor
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # Sample action from policy
            action, _ = sample_action(policy, state_tensor)

            # Convert action tensor to plain Python list
            action_list = action.tolist()

            # State transition following action
            next_state, reward, done = env.step(action_list)

            # Append the state that generated this action, not the resulting state
            states.append(state)
            actions.append(action_list)
            rewards.append(reward)

            # Update current state
            state = next_state

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "episode_length": len(rewards),
    }


# Rollout episode
def rollout_episode(env, policy, sample_action, gamma):
    # Reset environment
    state = env.reset()

    # Initialise trajectory data
    states = [state]
    actions = []
    rewards = []
    angle_errors = [env.get_angle_error()]

    # Reset episode
    done = False

    # Loop until done
    while not done: 
        # Convert state into tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # Sample action from policy
        with torch.no_grad():
            action, _ = sample_action(policy, state_tensor)

        # Convert action tensor to plain Python list
        action_list = action.tolist()

        # State transition following action (converted to list first)
        next_state, reward, done = env.step(action_list)

        # Update current state
        state = next_state

        # Append
        states.append(next_state)
        rewards.append(reward)
        actions.append(action_list)
        angle_errors.append(env.get_angle_error())

    # Discounted return
    discounted_return = sum(
        (gamma ** t) * reward
        for t, reward in enumerate(rewards)
    )
  
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "angle_errors": angle_errors,
        "discounted_return": discounted_return,
    }
