"""Plot saved robot experiments; no training or deployment takes place here."""

import math
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acp import ACP
from robot_2d.main_robot_2d_online import METHOD_SETTINGS
from robot_2d.utils import plot_robot_map


METHOD_LABELS = {method: settings[2] for method, settings in METHOD_SETTINGS.items()}


def disk_rectangle_area(radius, xlo, xhi, ylo, yhi):
    """Disk/rectangle intersection, used only for the offline mismatch reference."""
    xlo, xhi = max(-radius, xlo), min(radius, xhi)
    ylo, yhi = max(-radius, ylo), min(radius, yhi)
    if xlo >= xhi or ylo >= yhi:
        return 0.0
    splits = [xlo, xhi]
    for height in (ylo, yhi):
        if abs(height) < radius:
            root = math.sqrt(radius**2 - height**2)
            splits.extend(x for x in (-root, root) if xlo < x < xhi)
    splits = sorted(set(splits))

    def primitive(x):
        root = math.sqrt(max(0.0, radius**2 - x**2))
        return 0.5 * (x * root + radius**2 * math.asin(x / radius))

    area = 0.0
    for left, right in zip(splits[:-1], splits[1:]):
        height = math.sqrt(max(0.0, radius**2 - ((left + right) / 2)**2))
        if min(yhi, height) <= max(ylo, -height):
            continue
        upper_const, upper_arc = (yhi, 0) if yhi < height else (0, 1)
        lower_const, lower_arc = (ylo, 0) if ylo > -height else (0, -1)
        area += (upper_const - lower_const) * (right - left)
        area += (upper_arc - lower_arc) * (primitive(right) - primitive(left))
    return max(0.0, area)


def mismatch_probabilities(mdp):
    """Common mismatch probability for every grid source, including the goal."""
    if mdp.noise_sampler is not None:
        raise ValueError("This offline reference assumes the default zero/uniform-disk noise.")
    probability = 0.0
    if mdp.noise_radius > 0:
        hx, hy = mdp.cell_widths / 2
        area = disk_rectangle_area(mdp.noise_radius, -hx, hx, -hy, hy)
        nominal_mass = area / (np.pi * mdp.noise_radius**2)
        probability = mdp.noise_probability * (1 - nominal_mass)
    result = np.full((mdp.n_state, mdp.action_space.n), probability)
    result[mdp.exit_state] = 0  # Outside outcomes are never decision states.
    return np.clip(result, 0, 1)


def plot_R(runs):
    """Compare cumulative R estimates with the common true R-star."""
    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    estimates, probabilities = [], []
    for run in runs:
        mdp = run["mdp"]
        probability = run.get("R_star")
        if probability is None:
            probability = mismatch_probabilities(mdp)[mdp.start_state, 0]
        probabilities.append(probability)
        estimate = np.asarray([row["R_hat_next"] for row in run["history"]])
        estimates.append(estimate)
        axis.plot(np.arange(1, len(estimate) + 1), estimate, color="C0", alpha=0.25)
    episodes = np.arange(1, len(estimates[0]) + 1)
    axis.plot(episodes, np.mean(estimates, axis=0), label="Estimated R")
    axis.axhline(np.mean(probabilities), color="black", linestyle=":", label="True R*")
    axis.set(title="R estimation", xlabel="Deployment episode k", ylabel="Probability")
    return fig


def plot_calibration(runs, window):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    history = runs[0]["history"]
    episodes = np.arange(1, len(history) + 1)
    radii = np.asarray([row["C_hat"] for row in history])
    scores = [row["score"] for row in history]
    axes[0, 0].plot(episodes, scores, ".", label="Episode maximum continuous deviation")
    axes[0, 0].plot(episodes, np.where(np.isfinite(radii), radii, np.nan), label="Predicted C")
    support_bounds = [row.get("C_star_k", runs[0]["mdp"].noise_radius) for row in history]
    if "C_star_k" in history[0]:
        axes[0, 0].plot(episodes, support_bounds, "k:", label="True support bound C*")
    ceiling = max(1, max(support_bounds), max(scores)) * 1.1
    unbounded = np.isposinf(radii)
    if np.any(unbounded):
        axes[0, 0].scatter(
            episodes[unbounded], np.full(sum(unbounded), ceiling), marker="^", label="Unbounded",
        )
    all_misses, rolling_misses, frozen_misses, sizes = [], [], [], []
    for run in runs:
        quantile = ACP(
            run["initial_calibration_scores"], delta=run["delta"],
            delta_1=run["delta_1"], window_size=run["calibration_window"],
        )
        frozen_radius = quantile.quantile()
        quantile.delta_k = run["delta"]
        fixed, frozen, support = [], [], []
        mdp = run["mdp"].clone(nominal=True)
        for row in run["history"]:
            fixed.append(int(row["score"] > quantile.quantile()))
            frozen.append(int(row["score"] > frozen_radius))
            quantile.scores.append(row["score"])
            if "obstacles" in row:
                mdp.set_obstacles(row["obstacles"])
            radius = row["C_used"]
            counts = [len(mdp.localized_uncertainty_states(s, radius)) for s in range(mdp.n_state)]
            nominal_targets = mdp.nominal_transitions[mdp.trainable_states]
            support.append(np.mean(np.asarray(counts)[nominal_targets]))
        all_misses.append([row["miscoverage"] for row in run["history"]])
        rolling_misses.append(fixed)
        frozen_misses.append(frozen)
        sizes.append(support)
    for misses, label, style in (
        (all_misses, "ACP", "-"), (rolling_misses, "Rolling fixed delta", "--"),
        (frozen_misses, "Frozen pilot quantile", ":"),
    ):
        misses = np.asarray(misses)
        axes[0, 1].plot(episodes, np.mean(1 - np.cumsum(misses, axis=1) / episodes, axis=0),
                         style, label=label)
        coverage = []
        for row in misses:
            coverage.append([1 - np.mean(row[max(0, k - window + 1):k + 1])
                             for k in range(len(row))])
        axes[1, 0].plot(episodes, np.mean(coverage, axis=0), style, label=label)
    for axis in (axes[0, 1], axes[1, 0]):
        axis.axhline(1 - runs[0]["delta"], color="black", linestyle=":", label="Target")
        axis.set_ylim(0, 1.02)
    axes[1, 1].plot(episodes, np.mean(sizes, axis=0), label="Localized support")
    axes[1, 1].axhline(runs[0]["mdp"].n_state, color="black", linestyle=":", label="Global support")
    titles = ("Scores and predictions, first seed", "Cumulative coverage, same score stream",
              f"Trailing {window}-episode coverage", "Mean neighborhood size")
    for axis, title in zip(axes.ravel(), titles):
        axis.set(title=title, xlabel="Episode")
    return fig


def plot_performance(runs):
    """Compare the rewards and outcomes already recorded during online deployment."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
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
    axes[0].set(title="Cumulative mean discounted reward", xlabel="Online episode", ylabel="Reward")
    axes[1].set(title="Outcomes across online episodes", ylabel="Episode fraction", ylim=(0, 1.02))
    axes[1].set_xticks(np.arange(len(outcomes)), list(outcomes.values()))
    fig.suptitle("Recorded online deployment episodes; dots denote individual trials")
    return fig


def plot_trajectories(run):
    """Overlay every recorded deployment episode on its trial's fixed obstacle map."""
    mdp, history = run["mdp"], run["history"]
    if not history:
        raise ValueError("At least one deployment episode is required for plotting.")
    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    plot_robot_map(mdp, axis)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.5, vmax=len(history) + 0.5)
    trajectories = [np.asarray(row["positions"]) for row in history]
    for episode, trajectory in enumerate(trajectories, 1):
        axis.plot(trajectory[:, 0], trajectory[:, 1], color=cmap(norm(episode)),
                  linewidth=1, alpha=0.5, marker=".", markersize=3)
    positions = np.concatenate(trajectories)
    lo = np.minimum(positions.min(axis=0), 0) - 0.2 * mdp.cell_widths
    hi = np.maximum(positions.max(axis=0), mdp.bounds) + 0.2 * mdp.cell_widths
    axis.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
    ticks = np.unique(np.linspace(1, len(history), min(6, len(history)), dtype=int))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis,
                 ticks=ticks, label="Deployment episode")
    axis.set_title(f"{METHOD_LABELS[run['method']]}, trial {run['trial']}: "
                   f"{len(history)} deployment trajectories")
    return fig


def plot_comparison(window=10):
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
        figures["acp_calibration"] = plot_calibration(calibrated, window)
    for name, fig in figures.items():
        for axis in fig.axes:
            axis.grid(alpha=0.25, axis="y")
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=8)
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
