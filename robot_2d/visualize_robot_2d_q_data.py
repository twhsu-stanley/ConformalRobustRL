"""Display one saved training Q-table and its nominal greedy trajectory without saving."""

from collections import deque
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q_transformer.data import format_prompt
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP
from robot_2d.utils import plot_robot_motion


def _mdp_from_description(description):
    """Recover the unit-cell map used by generate_robot_2d_q_data."""
    nx, ny = map(int, re.search(r"size (\d+)x(\d+)", description).groups())
    endpoints = re.search(
        r"Start at \((\d+),\s*(\d+)\) and move to the goal state at \((\d+),\s*(\d+)\)",
        description,
    )
    sx, sy, gx, gy = map(int, endpoints.groups())
    grid = description.split("0 = free space, 1 = obstacle.\n\n", 1)[1].split("Start at", 1)[0]
    cells = np.fromstring(grid, dtype=int, sep=" ")
    if cells.size != nx * ny or not np.all(np.isin(cells, (0, 1))):
        raise ValueError("The saved description must contain a complete binary grid.")
    obstacles = [tuple(map(int, cell[::-1])) for cell in np.argwhere(cells.reshape(ny, nx))]
    return Robot2DMotionMDP(
        x_max=float(nx), y_max=float(ny), grid_shape=(nx, ny),
        start_cell=(sx, sy), goal_cell=(gx, gy), gamma=0.9,
        movement_reward=0.0, goal_reward=1.0, obstacles=obstacles,
    )


def _nominal_policy_trajectory(mdp, Q):
    """Break exact greedy ties toward the goal, then follow the resulting nominal policy."""
    if Q.shape != (mdp.n_state, mdp.action_space.n) or not np.all(np.isfinite(Q)):
        raise ValueError("The saved Q-table must be finite and match the map dimensions.")
    policy = np.argmax(Q, axis=1)
    predecessors = [[] for _ in mdp.states]
    for state in mdp.trainable_states:
        row = mdp.state_index(state)
        for action in np.flatnonzero(Q[row] == Q[row].max()):
            successor = mdp.state_index(mdp.nominal_next_state(state, int(action)))
            predecessors[successor].append((row, int(action)))
    goal_row = mdp.state_index(mdp.goal_state)
    reached = {goal_row}
    queue = deque([goal_row])
    while queue:
        for row, action in predecessors[queue.popleft()]:
            if row not in reached:
                policy[row] = action
                reached.add(row)
                queue.append(row)
    if mdp.state_index(mdp.start_state) not in reached:
        raise ValueError("The saved Q-table has no goal-reaching nominal greedy policy.")
    state = mdp.start_state
    trajectory = [mdp.coordinates(state)]
    while state != mdp.goal_state:
        state = mdp.nominal_next_state(state, int(policy[mdp.state_index(state)]))
        trajectory.append(mdp.coordinates(state))
    return policy, np.asarray(trajectory)


if __name__ == "__main__":
    index = 49
    show = True

    data_dir = Path(__file__).resolve().parent / "saved_results/q_transformer"
    dataset_path = data_dir / "q_data_2d.npz"
    with np.load(dataset_path, allow_pickle=False) as data:
        n_datapoints = len(data["map_index"])
        if not isinstance(index, (int, np.integer)) or not 1 <= index <= n_datapoints:
            raise ValueError(f"index must be an integer from 1 to {n_datapoints}.")
        row = index - 1
        description = str(data["descriptions"][data["map_index"][row]])
        Q = data["Q"][row].copy()
        R, C = float(data["R"][row]), float(data["C"][row])
    print(format_prompt(description, R, C), end="")
    mdp = _mdp_from_description(description)
    policy, trajectory = _nominal_policy_trajectory(mdp, Q)
    fig = plot_robot_motion(
        mdp, Q, policy, trajectory,
        title=f"Training data point {index}/{n_datapoints}: R={R}, C={C}", value_format=".2g",
    )
    fig.axes[1].set_title("Greedy policy and nominal trajectory")
    if show:
        plt.show()
