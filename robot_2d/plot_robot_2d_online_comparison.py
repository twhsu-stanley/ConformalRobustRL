"""Plot saved robot experiments; no training or deployment takes place here."""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_2d.main_robot_2d_online import METHOD_SETTINGS
from robot_2d.utils_2d import mismatch_probabilities, plot_trajectories
from utils import plot_R, plot_acp, plot_performance


plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})


METHOD_LABELS = {method: settings[2] for method, settings in METHOD_SETTINGS.items()}


if __name__ == "__main__":
    """Read saved runs and plot R, ACP, performance, and all deployment trajectories."""
    output_dir = Path(__file__).resolve().parent / "saved_results/online"
    runs = []
    for method, (R, C, _) in METHOD_SETTINGS.items():
        for filename in sorted(output_dir.glob(f"results_{method}_R{R}_C{C}_trial_*.pkl")):
            with filename.open("rb") as file:
                runs.append(pickle.load(file))
    if not runs:
        raise FileNotFoundError(f"No online comparison results found in {output_dir}.")
    estimated = [run for run in runs if run["fixed_R"] is None]
    calibrated = [run for run in runs if run["fixed_C"] is None]
    figures = {"policy_comparison": plot_performance(runs, METHOD_LABELS)}
    if estimated:
        probabilities = []
        for run in estimated:
            probability = run.get("R_star")
            if probability is None:
                mdp = run["mdp"]
                probability = mismatch_probabilities(mdp)[mdp.state_index(mdp.start_state), 0]
            probabilities.append(probability)
        figures["r_estimation"] = plot_R(estimated, probabilities)
    if calibrated:
        figures["acp_calibration"] = plot_acp(calibrated)
    for name, fig in figures.items():
        for axis in fig.axes:
            axis.grid(alpha=0.25, axis="y")
            if axis.get_legend_handles_labels()[0]:
                axis.legend()
        fig.savefig(output_dir / f"{name}.png", dpi=180)
        plt.close(fig)
    for run in runs:
        fig = plot_trajectories(run, METHOD_LABELS[run["method"]])
        name = (f"trajectories_{run['method']}_R{run['fixed_R']}_C{run['fixed_C']}"
                f"_trial_{run['trial']}")
        fig.savefig(output_dir / f"{name}.png", dpi=180)
        plt.close(fig)