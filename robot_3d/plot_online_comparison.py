"""Plot saved robot experiments; no training or deployment takes place here."""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_3d.main_robot_3d_online import METHOD_SETTINGS
from robot_3d.utils import plot_robot_map


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


def plot_R(runs):
    """Compare cumulative R estimates with the common true R-star."""
    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    estimates, probabilities = [], []
    for run in runs:
        probability = run["R_star"]
        probabilities.append(probability)
        estimate = np.asarray([row["R_hat_next"] for row in run["history"]])
        estimates.append(estimate)
        axis.plot(np.arange(1, len(estimate) + 1), estimate, color="C0", alpha=0.25)
    episodes = np.arange(1, len(estimates[0]) + 1)
    axis.plot(episodes, np.mean(estimates, axis=0), label=r"Estimated $\hat{R}_k$")
    axis.axhline(np.mean(probabilities), color="black", linestyle=":", label=r"True $R^\star$")
    axis.set(title="R estimation", xlabel="Deployment episode k", ylabel="Probability")
    return fig


def plot_acp(runs):
    """Show episode scores, radii, and adaptive delta for the first trial."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
    run = runs[0]
    history = run["history"]
    episodes = np.arange(1, len(history) + 1)
    radii = np.asarray([row["C_hat"] for row in history])
    scores = np.asarray([row["score"] for row in history], dtype=float)
    axes[0].plot(episodes, scores, ".", label="nonconformity scores")
    axes[0].plot(
        episodes, np.where(np.isfinite(radii), radii, np.nan), label=r"ACP quantile $\hat{C}_k$",
    )
    support_bounds = [row.get("C_star_k", run["mdp"].noise_radius) for row in history]
    if "C_star_k" in history[0]:
        axes[0].plot(episodes, support_bounds, "k:", label=r"True support bound $C^\star$")
    ceiling = max(1, max(support_bounds), np.nanmax(scores, initial=0)) * 1.1
    unbounded = np.isposinf(radii)
    if np.any(unbounded):
        axes[0].scatter(
            episodes[unbounded], np.full(sum(unbounded), ceiling), marker="^", label="Unbounded",
        )
    axes[0].set(title="Scores and predictions, first trial",
                xlabel="Deployment episode k", ylabel="Distance")

    axes[1].plot(episodes, [row["delta_k"] for row in history], label=r"$\delta_k$")
    axes[1].axhline(run["delta"], color="black", linestyle="--", label=r"Target $\delta$")
    axes[1].set(title="Adaptive failure parameter, first trial",
                    xlabel="Deployment episode k", ylabel="Failure parameter")

    return fig


def plot_performance(runs):
    """Compare the rewards and outcomes already recorded during online deployment."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
    methods = list(dict.fromkeys(run["method"] for run in runs))
    outcomes = {"goal_reached": "Goal", "collision": "Obstacle", "workspace_exit": "Exit",
                "timeout": "Step limit"}
    width = 0.8 / len(methods)
    for index, method in enumerate(methods):
        color = f"C{index}"
        label = METHOD_LABELS[method]
        histories = [run["history"] for run in runs if run["method"] == method]
        returns = np.array([[row["discounted_return"] for row in history] for history in histories])
        episodes = np.arange(1, returns.shape[1] + 1)
        averages = np.cumsum(returns, axis=1) / episodes
        axes[0].plot(episodes, averages.T, color=color, alpha=0.2)
        axes[0].plot(episodes, np.mean(averages, axis=0), color=color, label=label)
        rates = np.asarray([[np.mean([row[key] for row in history]) for key in outcomes]
                            for history in histories])
        positions = np.arange(len(outcomes)) + (index - (len(methods) - 1) / 2) * width
        axes[1].bar(positions, np.mean(rates, axis=0), width, color=color, label=label)
        axes[1].plot(positions, rates.T, "k.", alpha=0.6)
    axes[1].plot([], [], "k.", alpha=0.6, label="Individual trials")
    axes[0].set(title="Cumulative mean discounted reward", xlabel="Online episode", ylabel="Reward")
    axes[1].set(title="Outcomes across online episodes", ylabel="Episode fraction", ylim=(0, 1.02))
    axes[1].set_xticks(np.arange(len(outcomes)), list(outcomes.values()))
    fig.suptitle("Averaged cumulative returns and success rate")
    return fig


def plot_trajectories(run):
    """Overlay every recorded deployment episode on its trial's fixed obstacle map."""
    mdp, history = run["mdp"], run["history"]
    if not history:
        raise ValueError("At least one deployment episode is required for plotting.")
    fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    axis = fig.add_subplot(projection="3d")
    plot_robot_map(mdp, axis)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.5, vmax=len(history) + 0.5)
    trajectories = [np.asarray(row["positions"]) for row in history]
    for episode, trajectory in enumerate(trajectories, 1):
        axis.plot(*trajectory.T, color=cmap(norm(episode)),
                  linewidth=1, alpha=0.5, marker=".", markersize=3)
    positions = np.concatenate(trajectories)
    lo = np.minimum(positions.min(axis=0), 0) - 0.2 * mdp.cell_widths
    hi = np.maximum(positions.max(axis=0), mdp.bounds) + 0.2 * mdp.cell_widths
    axis.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
    ticks = np.unique(np.linspace(1, len(history), min(6, len(history)), dtype=int))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis,
                 ticks=ticks, label="Deployment episode")
    axis.set_title(f"{METHOD_LABELS[run['method']]}, trial {run['trial']}\n"
                   f"{len(history)} deployment trajectories")
    return fig


def plot_comparison():
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
    figures = {"policy_comparison": plot_performance(runs)}
    if estimated:
        figures["r_estimation"] = plot_R(estimated)
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
        fig = plot_trajectories(run)
        name = (f"trajectories_{run['method']}_R{run['fixed_R']}_C{run['fixed_C']}"
                f"_trial_{run['trial']}")
        fig.savefig(output_dir / f"{name}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    plot_comparison()
