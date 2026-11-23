"""Plot robot values and greedy policies from saved robust Q-learning agents."""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_3d.utils_3d import plot_robot_3d_tabular, plot_robot_motion


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "saved_results/fixedRC"
    trial = 0
    show = True
    combinations = ((0.2, 1.0), (0.2, 2.0))

    for R, C in combinations:
        filename = data_dir / f"robot_3d_R{R}_C{C}_agents.pkl"
        with open(filename, "rb") as f:
            saved = pickle.load(f)
        if not isinstance(trial, (int, np.integer)) or not 0 <= trial < len(saved["agents"]):
            raise ValueError("trial must be a valid zero-based saved trial index.")
        fig = plot_robot_3d_tabular(saved["mdp"], saved["agents"][trial], show=False)
        filename = data_dir / f"robot_3d_R{R}_C{C}_trial{trial + 1}_value_policy.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved value and policy plot to {filename.resolve()}")
        agent = saved["agents"][trial]
        overview = plot_robot_motion(saved["mdp"], agent.Q, agent.Q.argmax(axis=1),
                                     title=f"Robust Q-learning: R={R}, C={C}")
        filename = data_dir / f"robot_3d_R{R}_C{C}_trial{trial + 1}_value_policy_3d.png"
        overview.savefig(filename, dpi=200, bbox_inches="tight")
        print(f"Saved 3-D overview to {filename.resolve()}")
        plt.close(overview)

    if show:
        plt.show()
