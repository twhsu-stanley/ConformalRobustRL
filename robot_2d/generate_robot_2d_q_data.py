"""Generate and save 2-D robust Q-table targets without training a transformer."""

import sys
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q_transformer.data import generate_q_data
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP
from robot_2d.utils import randomize_obstacles


def randomize_start_goal(mdp, rng, start_goal_seperation=7.5):
    """Sample distinct workspace cells with the requested physical center distance.

    Obstacles are generated afterwards, so the template's obstacle map is ignored.
    Every ordered start/goal pair satisfying the distance constraint is equally likely.
    """
    if not np.isfinite(start_goal_seperation) or start_goal_seperation < 0:
        raise ValueError("start_goal_seperation must be finite and nonnegative.")
    centers = mdp.state_coordinates
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    pairs = np.argwhere((distances >= start_goal_seperation) & (distances > 0))
    if len(pairs) == 0:
        raise ValueError("No distinct start/goal cells satisfy start_goal_seperation.")
    start_row, goal_row = pairs[rng.integers(len(pairs))]
    return tuple(map(int, mdp.states[start_row])), tuple(map(int, mdp.states[goal_row]))


def training_maps(mdp, n_maps, *, n_obstacles=(10, 30), start_goal_seperation=7.5, seed=7):
    """Randomize endpoints, then obstacles, retaining a nominal start-to-goal path."""
    if not isinstance(n_maps, (int, np.integer)) or n_maps < 2:
        raise ValueError("n_maps must be an integer of at least two.")
    low, high = n_obstacles
    if not 0 <= low <= high:
        raise ValueError("n_obstacles must be an ordered pair of nonnegative counts.")
    rng = np.random.default_rng(seed)
    for _ in range(n_maps):
        nominal_mdp = mdp.clone(nominal=True)
        nominal_mdp.start_state, nominal_mdp.goal_state = randomize_start_goal(
            nominal_mdp, rng, start_goal_seperation,
        )
        count = int(rng.integers(low, high + 1))
        if count == 0:
            nominal_mdp.set_obstacles([])
        else:
            randomize_obstacles(nominal_mdp, rng, n_obstacles=count)
        nominal_mdp.reset()
        yield nominal_mdp


def main(
    mdp, *, gamma=0.9, n_maps=3000, n_obstacles=(20, 30), start_goal_seperation=7.5, seed=7,
    R_values=(0.0, 0.1, 0.2, 0.3),
    C_values=(0.0, 1.0, 1.75, np.inf),
):
    dataset_path = Path(__file__).resolve().parent / "saved_results/q_transformer/q_data.npz"
    maps = training_maps(
        mdp, n_maps, n_obstacles=n_obstacles, start_goal_seperation=start_goal_seperation,
        seed=seed,
    )
    generate_q_data(maps, R_values, C_values, gamma, dataset_path, n_maps=n_maps)
    print(f"Saved Q-table dataset to {dataset_path}")
    return dataset_path


if __name__ == "__main__":
    # Match the online experiment. For main_robot_2d.py use movement_reward=.01 and gamma=.95.
    mdp = Robot2DMotionMDP(
        x_max=10, y_max=10, grid_shape=(10, 10), obstacles=[],
        movement_reward=0.0, goal_reward=1.0,
    )
    main(mdp, gamma=0.9, n_maps=2500, n_obstacles=(20, 30), start_goal_seperation=7.5,
        R_values=(0.0, 0.2, 0.4, 0.5), C_values=(0.0, 1.0, 1.75, 3.0, np.inf), seed=7)
