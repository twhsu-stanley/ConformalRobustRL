"""Plot the fixed-R,C comparisons saved by main_robot_2d.py."""

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
        data_dir = Path(__file__).resolve().parent / "saved_results"
    data_dir = Path(data_dir)
    comparisons = (
        (
            "Fixed C = 1.0", ((0.0, 1.0), (0.1, 1.0), (0.2, 1.0)),
            "robot_2d_fixed_C1.0_compare_R.png",
        ),
        (
            "Fixed R = 0.2", ((0.2, 0.0), (0.2, 1.0), (0.2, 1.5)),
            "robot_2d_fixed_R0.2_compare_C.png",
        ),
    )
    figures = []
    for title, combinations, output_name in comparisons:
        fig, axis = plt.subplots()
        for R, C in combinations:
            filename = data_dir / f"robot_2d_R{R}_C{C}.pkl"
            with open(filename, "rb") as f:
                evaluation_return = pickle.load(f)
            mean, std, _ = calc_evaluation_return_mean_std(evaluation_return)
            steps = np.arange(len(mean))
            line, = axis.plot(steps, mean, label=f"R = {R}, C = {C}")
            axis.fill_between(steps, mean - std, mean + std, color=line.get_color(), alpha=0.2)
        axis.set(
            xlabel="Cumulative Training Steps", ylabel="Evaluation Return: V(initial state)",
            title=f"Robust Q-Learning for 2-D Robot Motion: {title}",
        )
        axis.set_xlim(left=0)
        axis.grid()
        axis.legend()
        filename = data_dir / output_name
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {filename.resolve()}")
        figures.append(fig)
    if show:
        plt.show()
    return figures


if __name__ == "__main__":
    main()
