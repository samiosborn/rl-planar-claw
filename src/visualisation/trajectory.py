# src/visualisation/trajectory.py
import math

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


# Standardise returns from sampled trajectories
def standardise_returns(trajectories):
    # Extract discounted returns
    returns = np.asarray(
        [trajectory["discounted_return"] for trajectory in trajectories], dtype=float
    )
    if returns.size == 0:
        raise ValueError("At least one trajectory is required")

    # Compute standard deviation
    std = returns.std()

    # All trajectories have the same return
    if std == 0:
        return np.zeros_like(returns)

    # Standardise
    return (returns - returns.mean()) / std


# Build return colour scale
def _return_normalisation(returns):
    # Return range
    minimum = float(np.min(returns))
    maximum = float(np.max(returns))

    # Equal returns
    if minimum == maximum:
        padding = max(abs(minimum) * 0.01, 1.0)
        minimum -= padding
        maximum += padding
    return Normalize(vmin=minimum, vmax=maximum)


# Plot one trajectory panel
def _plot_trajectory_panel(ax, trajectories, norm, cmap, alpha_by_return):
    # Plot each trajectory
    for trajectory in trajectories:
        discounted_return = trajectory["discounted_return"]

        # Weight opacity by return
        alpha = 0.4 + 0.5 * norm(discounted_return) if alpha_by_return else 1.0

        # Extract angle error
        angle_errors = trajectory["angle_errors"]

        # Colour according to return
        ax.plot(
            np.arange(len(angle_errors)),
            angle_errors,
            color=cmap(norm(discounted_return)),
            alpha=alpha,
        )

    # Target angle error
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)

    # Labels
    ax.set_xlabel("Control step")
    ax.set_ylabel("Angle error (rad)")


# Plot trajectories angle error over time
def plot_trajectories(trajectories, *, alpha_by_return=False):
    if not trajectories:
        raise ValueError("At least one trajectory is required")

    # Extract discounted returns
    returns = np.asarray(
        [trajectory["discounted_return"] for trajectory in trajectories], dtype=float
    )

    # Colour according to returns
    norm = _return_normalisation(returns)
    cmap = plt.get_cmap("coolwarm")

    # Create figure
    fig, ax = plt.subplots()

    # Plot trajectories
    _plot_trajectory_panel(ax, trajectories, norm, cmap, alpha_by_return)
    ax.set_title("Sampled policy trajectories")

    # Colourbar
    colour_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(colour_map, ax=ax, label="Discounted return")
    return fig, ax


# Plot trajectories across training checkpoints
def plot_trajectory_grid(
    checkpoint_trajectories,
    *,
    titles=None,
    alpha_by_return=False,
    columns=None,
):
    """Plot trajectory samples using one return scale across every checkpoint."""
    checkpoint_trajectories = list(checkpoint_trajectories)
    if not checkpoint_trajectories:
        raise ValueError("At least one checkpoint is required")
    if any(not trajectories for trajectories in checkpoint_trajectories):
        raise ValueError("Each checkpoint must contain at least one trajectory")

    panel_count = len(checkpoint_trajectories)
    if titles is None:
        titles = [f"Checkpoint {index + 1}" for index in range(panel_count)]
    elif len(titles) != panel_count:
        raise ValueError("There must be one title per checkpoint")

    # Extract all discounted returns
    all_returns = np.asarray(
        [
            trajectory["discounted_return"]
            for trajectories in checkpoint_trajectories
            for trajectory in trajectories
        ],
        dtype=float,
    )

    # Global return colour scale
    norm = _return_normalisation(all_returns)
    cmap = plt.get_cmap("coolwarm")

    # Panel layout
    if columns is None:
        columns = min(3, panel_count)
    if columns < 1:
        raise ValueError("columns must be at least 1")
    rows = math.ceil(panel_count / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 3.8 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    active_axes = list(axes.flat[:panel_count])

    # Plot each checkpoint
    for ax, trajectories, title in zip(active_axes, checkpoint_trajectories, titles):
        _plot_trajectory_panel(ax, trajectories, norm, cmap, alpha_by_return)
        ax.set_title(title)

    # Remove unused panels
    for ax in axes.flat[panel_count:]:
        fig.delaxes(ax)

    # Shared colourbar
    colour_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(colour_map, ax=active_axes, label="Discounted return", pad=0.02)
    fig.suptitle("Sampled policy trajectories across training")
    return fig, np.asarray(active_axes, dtype=object)
