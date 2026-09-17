# scripts/train_reinforce.py

import torch

import config.simulation as CONFIG
from src.algorithms.reinforce import PolicyNetwork, train_episode
from src.env import PlanarClawEnv


# Initialise environment
env = PlanarClawEnv(gui=False)

# Initialise policy
policy = PolicyNetwork()

# Initialise optimiser
optimiser = torch.optim.Adam(policy.parameters(), lr=CONFIG.LEARNING_RATE)

try:
    # Train policy
    for episode in range(CONFIG.NUM_TRAINING_EPISODES):
        diagnostics = train_episode(
            env,
            policy,
            optimiser,
            CONFIG.GAMMA,
        )

        # Print training diagnostics
        if episode % 10 == 0:
            print(
                f"Episode {episode:4d} | "
                f"return {diagnostics['discounted_return']:8.2f} | "
                f"loss {diagnostics['loss']:10.2f} | "
                f"grad {diagnostics['gradient_norm']:8.2f}"
            )

finally:
    env.close()
