"""Generate and save 2-D robust Q-table targets without training a transformer."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grid_q_data import training_maps
from q_transformer.data import generate_q_data
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP


if __name__ == "__main__":
    mdp = Robot2DMotionMDP(
        x_max=10.0, y_max=10.0, grid_shape=(10, 10),
        start_cell=(1, 1), goal_cell=(8, 8),
        gamma=0.9, movement_reward=0.0, goal_reward=1.0,
    )
    n_maps = 2000
    R_values = (0.0, 0.2, 0.4, 0.5)
    C_values = (0.0, 0.5, 0.75, 1.5, 2.0)
    n_obstacles = (25, 40)
    seed = 7
    n_workers = 8

    dataset_path = Path(__file__).resolve().parent / "saved_results/q_transformer/q_data_2d.npz"
    maps = training_maps(mdp, n_maps, n_obstacles=n_obstacles, seed=seed)
    generate_q_data(maps, R_values, C_values, dataset_path, n_maps=n_maps, n_workers=n_workers)
    print(f"Saved Q-table dataset to {dataset_path}")
