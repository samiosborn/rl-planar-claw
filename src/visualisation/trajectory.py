# src/visualisation/trajectory.py

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


# Standardise returns from sampled trajectories
def standardise_returns(trajectories):
    # Extract discounted returns
    returns = np.array([trajectory["discounted_return"] for trajectory in trajectories], dtype=float)

    # Compute mean and standard deviation
    mean = returns.mean()
    std = returns.std()

    # Debug standardisation
    print(f"Mean return: {mean:.4f}")
    print(f"Return std: {std:.4f}")    

    # All trajectories have the same return
    if std == 0:
        return np.zeros_like(returns)

    # Standardise
    standardised_returns = (returns - mean) / std

    return standardised_returns


# Plot trajectories angle error over time
def plot_trajectories(trajectories):
    # Create figure
    fig, ax = plt.subplots()

    # Standardise trajectory returns
    standardised_returns = standardise_returns(trajectories)

    # Colour according to returns
    norm = TwoSlopeNorm(
        vmin=standardised_returns.min(),
        vcenter=0.0,
        vmax=standardised_returns.max(),
    )

    cmap = plt.get_cmap("coolwarm")

    # Plot each trajectory
    for i, (trajectory, standardised_return) in enumerate(
    zip(trajectories, standardised_returns)
    ):
        # Debug trajectory return
        print(
            f"Trajectory {i}: "
            f"discounted return = {trajectory['discounted_return']:.6f}, "
            f"standardised return = {standardised_return:.2f}"
        )

        # Extract angle error
        angle_errors = trajectory["angle_errors"]
        steps = np.arange(len(angle_errors))

        # Map relative return to colour
        colour = cmap(norm(standardised_return))

        # Plot trajectory
        ax.plot(
            steps,
            angle_errors,
            color=colour,
        )

    # Target angle error
    ax.axhline(0.0, linestyle="--")

    # Labels
    ax.set_xlabel("Control step")
    ax.set_ylabel("Angle error (rad)")
    ax.set_title("Sampled policy trajectories")

    # Colour
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="Standardised discounted return")

    return fig, ax
