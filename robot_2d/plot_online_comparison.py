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
    result[mdp.failure_state] = 0  # Outside outcomes are never decision states.
    return np.clip(result, 0, 1)


def plot_R(runs):
    """Compare the H-transition estimator with the fixed common R-star."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    estimates, probabilities = [], []
    for run in runs:
        mdp = run["mdp"]
        probabilities.append(mismatch_probabilities(mdp)[mdp.start_state, 0])
        estimate = [row["R_next"] for row in run["history"]]
        estimates.append(estimate)
        axes[0].plot(np.arange(1, len(estimate) + 1), estimate, color="C0", alpha=0.25)
    episodes = np.arange(1, len(estimates[0]) + 1)
    counts = [row["transition_count"] for row in runs[0]["history"]]
    axes[0].plot(episodes, np.mean(estimates, axis=0), label="Estimated R")
    axes[0].axhline(np.mean(probabilities), color="black", linestyle=":", label="True R*")
    errors = np.asarray(estimates) - np.asarray(probabilities)[:, None]
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    axes[1].loglog(counts, np.maximum(rmse, 1e-12), label="RMSE over seeds")
    if np.any(rmse > 0):
        axes[1].loglog(counts, max(rmse) * np.sqrt(counts[0] / np.asarray(counts)),
                       "--", label=r"$n^{-1/2}$ slope")
    axes[0].set(title="R estimation", xlabel="Episode", ylabel="Probability")
    axes[1].set(title="Error around true R*", xlabel="Transitions", ylabel="RMSE")
    fig.suptitle("H noisy transitions per episode; restarts are not counted")
    return fig


def plot_calibration(runs, window):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    history = runs[0]["history"]
    episodes = np.arange(1, len(history) + 1)
    radii = np.asarray([row["conformal_radius"] for row in history])
    scores = [row["score"] for row in history]
    axes[0, 0].plot(episodes, scores, ".", label="Episode maximum continuous deviation")
    axes[0, 0].plot(episodes, np.where(np.isfinite(radii), radii, np.nan), label="Predicted C")
    ceiling = max(1, runs[0]["mdp"].noise_radius) * 1.1
    for tail, level, marker, label in (
        (np.isposinf(radii), ceiling, "^", "Unbounded"),
        (np.isneginf(radii), 0, "x", "Empty; planner uses C=0"),
    ):
        if np.any(tail):
            axes[0, 0].scatter(
                episodes[tail], np.full(sum(tail), level), marker=marker, label=label,
            )
    all_misses, rolling_misses, frozen_misses, sizes = [], [], [], []
    for run in runs:
        quantile = ACP(
            run["initial_calibration_scores"], target_failure=run["target_failure"],
            initial_failure=run["initial_failure"], window_size=run["calibration_window"],
        )
        frozen_radius = quantile.quantile()
        quantile.delta = run["target_failure"]
        fixed, frozen, support = [], [], []
        for row in run["history"]:
            fixed.append(int(row["score"] > quantile.quantile()))
            frozen.append(int(row["score"] > frozen_radius))
            quantile.scores.append(row["score"])
            mdp, radius = run["mdp"], row["planning_radius"]
            counts = [len(mdp.localized_states_around(s, radius)) for s in range(mdp.n_state)]
            nominal_targets = mdp.nominal_transitions[mdp.trainable_states]
            support.append(np.mean(np.asarray(counts)[nominal_targets]))
        all_misses.append([row["miscoverage"] for row in run["history"]])
        rolling_misses.append(fixed)
        frozen_misses.append(frozen)
        sizes.append(support)
    prediction_label = "ACP" if runs[0].get("fixed_C") is None else "Fixed C"
    for misses, label, style in (
        (all_misses, prediction_label, "-"), (rolling_misses, "Rolling fixed delta", "--"),
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
    planned = [[row["planned_miscoverage"] for row in run["history"]] for run in runs]
    if np.any(np.asarray(planned) != all_misses):
        coverage = 1 - np.cumsum(planned, axis=1) / episodes
        axes[0, 1].plot(episodes, np.mean(coverage, axis=0), "-.", label="Planning-radius coverage")
    for axis in (axes[0, 1], axes[1, 0]):
        axis.axhline(1 - runs[0]["target_failure"], color="black", linestyle=":", label="Target")
        axis.set_ylim(0, 1.02)
    axes[1, 1].plot(episodes, np.mean(sizes, axis=0), label="Localized support")
    axes[1, 1].axhline(runs[0]["mdp"].n_state, color="black", linestyle=":", label="Global support")
    titles = ("Scores and predictions, first seed", "Cumulative coverage, same score stream",
              f"Trailing {window}-episode coverage", "Mean neighborhood size")
    for axis, title in zip(axes.ravel(), titles):
        axis.set(title=title, xlabel="Episode")
    return fig


def plot_performance(runs):
    """Show seed means and individual seed results from fresh final-policy evaluations."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    methods = list(dict.fromkeys(run["method"] for run in runs))
    labels = {"proposed": "Proposed", "nominal": "Nominal", "global": "Global R-contamination"}
    metrics = {
        "success_rate": "Goal success", "collision_rate": "Obstacle collision",
        "workspace_exit_rate": "Workspace exit", "timeout_rate": "Goal not reached by H",
        "discounted_return": "Actual discounted reward",
        "discounted_goal_arrival": "Discounted actual goal arrival",
    }
    for axis, (metric, title) in zip(axes.ravel(), metrics.items()):
        for index, method in enumerate(methods):
            values = [run["evaluation"][metric] for run in runs if run["method"] == method]
            axis.bar(index, np.mean(values), color=f"C{index}", alpha=0.6)
            axis.scatter(np.full(len(values), index), values, color="black", s=15)
        axis.set_xticks(np.arange(len(methods)), [labels[method] for method in methods])
        axis.set_title(title)
        if metric != "discounted_return":
            axis.set_ylim(0, 1.02)
    fig.suptitle("Independent random rollouts; dots denote independent seeds")
    return fig


def plot_comparison(data_file=None, window=100):
    """Load comparison.pkl and save the three requested figures as PNG and PDF."""
    if data_file is None:
        data_file = Path(__file__).resolve().parent / "saved_results/online_comparison"
        data_file = data_file / "comparison.pkl"
    data_file = Path(data_file)
    with data_file.open("rb") as file:
        runs = pickle.load(file)
    proposed = [run for run in runs if run["method"] == "proposed"]
    figures = {}
    if proposed:
        figures["r_estimation"] = plot_R(proposed)
        figures["acp_calibration"] = plot_calibration(proposed, window)
    if all(run["evaluation"] is not None for run in runs):
        figures["policy_comparison"] = plot_performance(runs)
    for name, fig in figures.items():
        for axis in fig.axes:
            axis.grid(alpha=0.25, axis="y")
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=8)
        for extension in ("png", "pdf"):
            fig.savefig(data_file.parent / f"{name}.{extension}", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    plot_comparison()
