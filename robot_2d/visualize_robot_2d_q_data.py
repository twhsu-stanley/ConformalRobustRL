"""Display one saved training Q-table and its nominal greedy trajectory without saving."""

from collections import deque
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q_transformer.data import format_prompt, mdp_parameters
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP
from robot_2d.utils_2d import plot_robot_motion


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
        mdp = Robot2DMotionMDP(**mdp_parameters(data, data["map_index"][row]))
        Q = data["Q"][row].copy()
        R, C = float(data["R"][row]), float(data["C"][row])
    print(format_prompt(mdp, R, C), end="")
    policy, trajectory = _nominal_policy_trajectory(mdp, Q)
    fig = plot_robot_motion(
        mdp, Q, policy, trajectory,
        title=f"Training data point {index}/{n_datapoints}: R={R}, C={C}", value_format=".2g",
    )
    fig.axes[1].set_title("Greedy policy and nominal trajectory")
    if show:
        plt.show()
