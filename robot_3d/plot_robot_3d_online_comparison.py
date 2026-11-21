"""Plot saved robot experiments; no training or deployment takes place here."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_3d.main_robot_3d_online import METHOD_SETTINGS
from utils import load_online_results, plot_R, plot_acp, plot_performance
from utils import print_online_summary


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
    """Read saved runs and plot R, ACP, and performance."""
    output_dir = Path(__file__).resolve().parent / "saved_results/online"
    runs = load_online_results(output_dir, METHOD_SETTINGS)
    if not runs:
        raise FileNotFoundError(f"No online comparison results found in {output_dir}.")
    print_online_summary(runs, METHOD_LABELS)
    episode_xlim = (0, max(len(run["history"]) for run in runs))
    estimated = [run for run in runs if run["fixed_R"] is None]
    calibrated = [run for run in runs if run["fixed_C"] is None]
    # Set (min, max) limits here; ACP also accepts a dict keyed by subplot index.
    figures = {
        "policy_comparison": plot_performance(runs, METHOD_LABELS, xlim=episode_xlim, ylim=(0, 0.2))
    }
    if estimated:
        figures["r_estimation"] = plot_R(estimated, xlim=episode_xlim, ylim=(0, 0.6))
    if calibrated:
        figures["acp_calibration"] = plot_acp(calibrated, xlim=episode_xlim, ylim=None)
    for name, fig in figures.items():
        for axis in fig.axes:
            axis.grid(alpha=0.25, axis="y")
            if axis.get_legend_handles_labels()[0]:
                axis.legend()
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"{name}.{extension}", dpi=180)
        plt.close(fig)
    print(f"Saved plots to {output_dir}")
