"""Evaluate saved Q predictions and paired value iteration on held-out 3-D maps."""

from pathlib import Path
import re
import sys

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q_transformer.evaluate import evaluate_q_transformer
from robot_3d.robot_3d_motion_mdp import Robot3DMotionMDP


def _mdp_from_description(description):
    """Recover the unit-cell MDP and verify that its inference prompt is unchanged."""
    shape = tuple(map(int, re.search(r"size (\d+)x(\d+)x(\d+)", description).groups()))
    endpoints = re.search(r"Start at \(([^)]+)\).*goal state at \(([^)]+)\)", description)
    start, goal = (tuple(map(int, text.split(","))) for text in endpoints.groups())
    grid = description.split("0 = free space, 1 = obstacle.\n\n", 1)[1].split("Start at", 1)[0]
    cells = np.fromstring(grid, dtype=int, sep=" ")
    if cells.size != np.prod(shape) or not np.all(np.isin(cells, (0, 1))):
        raise ValueError("The saved description must contain a complete binary grid.")
    obstacles = [tuple(cell[::-1]) for cell in np.argwhere(cells.reshape(shape[::-1]))]
    mdp = Robot3DMotionMDP(
        x_max=float(shape[0]), y_max=float(shape[1]), z_max=float(shape[2]), grid_shape=shape,
        start_cell=start, goal_cell=goal, gamma=0.9, movement_reward=0.0, goal_reward=1.0,
        obstacles=obstacles,
    )
    if mdp.q_transformer_context() != description:
        raise ValueError("The reconstructed MDP does not reproduce its saved description.")
    return mdp


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "saved_results/q_transformer"
    evaluate_q_transformer(
        output_dir / "q_data_3d.npz", output_dir / "checkpoint", _mdp_from_description,
        n_maps=10, seed=42, validation_fraction=0.1, tolerance=1e-8,
        n_workers=2, cpu_threads=4, truncation=False,
    )
