# scripts/plot_rollout_reinforce.py

import matplotlib.pyplot as plt

import config.simulation as CONFIG
from src.algorithms.reinforce import PolicyNetwork, sample_action
from src.env import PlanarClawEnv
from src.rollout import rollout_episode
from src.visualisation.trajectory import plot_trajectories


# Initialise
env = PlanarClawEnv(gui=False)
policy = PolicyNetwork()

try:
    # Sample trajectories
    trajectories = []

    for _ in range(CONFIG.NUM_ROLLOUT_EPISODES):
        trajectory = rollout_episode(
            env,
            policy,
            sample_action,
            CONFIG.GAMMA,
        )
        trajectories.append(trajectory)

    # Plot trajectories
    fig, ax = plot_trajectories(trajectories)

    # Display plot
    plt.show()

finally:
    env.close()
