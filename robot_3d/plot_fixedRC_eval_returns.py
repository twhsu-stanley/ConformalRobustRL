"""Plot the fixed-R,C comparisons saved by main_robot_3d.py."""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils_tabular import calc_evaluation_return_mean_std


def main(*, data_dir=None, show=True):
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "saved_results/fixedRC"
    data_dir = Path(data_dir)
    comparisons = (
        (
            "Fixed C = 1.0", ((0.0, 1.0), (0.1, 1.0), (0.2, 1.0)),
            "robot_3d_fixed_C1.0_compare_R.png",
        ),
        (
            "Fixed R = 0.2", ((0.2, 0.0), (0.2, 1.0), (0.2, 2.0)),
            "robot_3d_fixed_R0.2_compare_C.png",
        ),
    )
    figures = []
    for title, combinations, output_name in comparisons:
        fig, (axis, steps_axis) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        labels = []
        for R, C in combinations:
            filename = data_dir / f"robot_3d_R{R}_C{C}.pkl"
            with open(filename, "rb") as f:
                evaluation_return = pickle.load(f)
            mean, std, _ = calc_evaluation_return_mean_std(evaluation_return)
            steps = np.arange(len(mean))
            line, = axis.plot(steps, mean, label=f"R = {R}, C = {C}")
            axis.fill_between(steps, mean - std, mean + std, color=line.get_color(), alpha=0.2)
            with (data_dir / f"robot_3d_R{R}_C{C}_agents.pkl").open("rb") as file:
                agents = pickle.load(file)["agents"]
            totals = [agent.training_diagnostics["transitions"] for agent in agents]
            position = len(labels)
            labels.append(f"R={R}\nC={C}")
            steps_axis.bar(position, np.mean(totals), color=line.get_color(), alpha=0.7)
            steps_axis.plot(np.full(len(totals), position), totals, "k.", alpha=0.6)
        axis.set(
            xlabel="Cumulative Training Steps", ylabel="Evaluation Return: V(initial state)",
            title=f"Robust Q-Learning for 3-D Robot Motion: {title}",
        )
        axis.set_xlim(left=0)
        axis.grid()
        axis.legend()
        steps_axis.set(title="Training steps to convergence", ylabel="Global training steps")
        steps_axis.set_xticks(np.arange(len(labels)), labels)
        steps_axis.grid(axis="y", alpha=0.25)
        filename = data_dir / output_name
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {filename.resolve()}")
        figures.append(fig)
    if show:
        plt.show()
    return figures


if __name__ == "__main__":
    main()
