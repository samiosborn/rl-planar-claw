# tests/test_trajectory_visualisation.py
import matplotlib
import pytest

matplotlib.use("Agg")

from scripts.plot_rollout_reinforce import select_evenly_spaced
from src.visualisation.trajectory import plot_trajectory_grid


# Build a minimal trajectory
def trajectory(discounted_return):
    return {
        "discounted_return": discounted_return,
        "angle_errors": [0.5, 0.25, 0.0],
    }


# Shared colour range
def test_grid_uses_one_global_return_scale():
    # Sample trajectories
    fig, axes = plot_trajectory_grid(
        [
            [trajectory(-10.0), trajectory(-8.0)],
            [trajectory(2.0), trajectory(10.0)],
        ]
    )

    # Distinct colours and shared colourbar
    assert len(axes) == 2
    assert axes[0].lines[0].get_color() != axes[1].lines[0].get_color()
    assert fig.axes[-1].get_ylabel() == "Discounted return"


# Equal returns and unused panels
def test_grid_handles_equal_returns_and_unused_panels():
    fig, axes = plot_trajectory_grid(
        [[trajectory(3.0)], [trajectory(3.0)], [trajectory(3.0)], [trajectory(3.0)]],
        columns=3,
    )

    assert len(axes) == 4
    assert len(fig.axes) == 5


# Select checkpoint updates
def test_even_checkpoint_selection_includes_endpoints():
    paths = list(range(11))
    assert select_evenly_spaced(paths, 4) == [0, 3, 7, 10]
    assert select_evenly_spaced(paths, 1) == [10]


# Reject excess checkpoint count
def test_cannot_select_more_checkpoints_than_available():
    with pytest.raises(ValueError, match="only 2 are available"):
        select_evenly_spaced([0, 1], 3)
