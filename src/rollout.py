# src/rollout.py

import torch

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
