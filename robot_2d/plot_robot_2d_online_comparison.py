"""Plot saved robot experiments; no training or deployment takes place here."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_2d.main_robot_2d_online import METHOD_SETTINGS
from robot_2d.utils_2d import plot_trajectories
from utils import load_online_results, plot_R, plot_acp, plot_R_acp, plot_performance
from utils import plot_acp_empirical_coverage, print_and_save_summary, print_online_summary


plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


METHOD_LABELS = {method: settings[2] for method, settings in METHOD_SETTINGS.items()}


if __name__ == "__main__":
    """Read saved runs and plot R, ACP, performance, and all deployment trajectories."""
    output_dir = Path(__file__).resolve().parent / "saved_results/online"
    runs = load_online_results(output_dir, METHOD_SETTINGS)
    if not runs:
        raise FileNotFoundError(f"No online comparison results found in {output_dir}.")
    print_and_save_summary(
        output_dir / "online_summary.txt", print_online_summary, runs, METHOD_LABELS,
    )
    episode_xlim = (0, max(len(run["history"]) for run in runs))
    r_estimation_runs = [run for run in runs if run["fixed_R"] is None]
    acp_runs = [run for run in runs if run["fixed_C"] is None]
    # Set limits here; for plot_acp, keys 0/1 select the scores/quantile and delta figures.
    figures = {
        "policy_comparison": plot_performance(runs, METHOD_LABELS, xlim=episode_xlim, ylim=(0, 0.5))
    }
    if r_estimation_runs:
        figures["r_estimation"] = plot_R(r_estimation_runs, xlim=episode_xlim, ylim=(0, 1))
    if acp_runs:
        figures["acp_scores_quantile"], figures["acp_delta_k"] = plot_acp(
            acp_runs, xlim=episode_xlim, ylim=None,
        )
        figures["acp_empirical_converage"] = plot_acp_empirical_coverage(
            acp_runs, xlim=episode_xlim, ylim=(0, 1),
        )
    if r_estimation_runs and acp_runs:
        figures["r_acp"] = plot_R_acp(
            r_estimation_runs, acp_runs, xlim=episode_xlim, ylim={0: (0, 1), 1: (0, 1)},
        )
    for name, fig in figures.items():
        for axis in fig.axes:
            axis.grid(alpha=0.25, axis="y")
            if axis.get_legend_handles_labels()[0]:
                axis.legend()
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"{name}.{extension}", dpi=180)
        plt.close(fig)
    for method, label in METHOD_LABELS.items():
        method_runs = [run for run in runs if run["method"] == method]
        if not method_runs:
            continue
        fig = plot_trajectories(method_runs, label)
        name = f"trajectories_{method}"
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"{name}.{extension}", dpi=180)
        plt.close(fig)
    print(f"Saved plots to {output_dir}")
