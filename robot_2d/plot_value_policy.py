"""Plot robot values and greedy policies from saved robust Q-learning agents."""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils_tabular import plot_robot_motion


def plot_robot_2d_tabular(mdp, agent, *, show=True):
    """Plot V(s)=max_a Q(s,a) and the greedy policy on the physical robot grid."""
    if agent.Q.shape != (mdp.n_state, mdp.action_space.n):
        raise ValueError("The agent Q-table must match the MDP state and action spaces.")
    title = f"Robust Q-Learning for 2-D Robot Motion: R={agent.R}, C={agent.C}"
    fig = plot_robot_motion(mdp, agent.Q, np.argmax(agent.Q, axis=1), title=title)
    fig.axes[1].set_title("Greedy policy from learned Q")
    if show:
        plt.show()
    return fig


def main(*, R=0.2, C=1.0, trial=0, data_dir=None, show=True):
    """Load one saved trial, plot its values and policy, and save the figure."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "saved_results"
    data_dir = Path(data_dir)
    filename = data_dir / f"robot_2d_R{R}_C{C}_agents.pkl"
    with open(filename, "rb") as f:
        saved = pickle.load(f)
    if not isinstance(trial, (int, np.integer)) or not 0 <= trial < len(saved["agents"]):
        raise ValueError("trial must be a valid zero-based saved trial index.")
    fig = plot_robot_2d_tabular(saved["mdp"], saved["agents"][trial], show=False)
    filename = data_dir / f"robot_2d_R{R}_C{C}_trial{trial + 1}_value_policy.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved value and policy plot to {filename.resolve()}")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    main()
