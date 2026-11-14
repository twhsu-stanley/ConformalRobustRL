"""Shared tabular experiment statistics and plots."""

import numpy as np


def calc_evaluation_return_mean_std(evaluation_return):
    """Average full curves, holding each stopped learner at its final value."""
    curves = [np.asarray(curve, dtype=float) for curve in evaluation_return]
    if not curves or any(curve.ndim != 1 or len(curve) == 0 for curve in curves):
        raise ValueError("Provide nonempty one-dimensional learning curves.")
    max_len = max(map(len, curves))
    evaluation_return_np = np.stack([
        np.pad(curve, (0, max_len - len(curve)), mode="edge") for curve in curves
    ])
    evaluation_return_mean = np.mean(evaluation_return_np, axis=0)
    evaluation_return_std = np.std(evaluation_return_np, axis=0)
    return evaluation_return_mean, evaluation_return_std, evaluation_return_np


def plot_conformal_history(history, delta):
    """Plot deployment uncertainty, showing estimation and ACP only when enabled."""
    import matplotlib.pyplot as plt

    if not history:
        raise ValueError("At least one deployment episode is required for plotting.")
    has_acp = "C_hat" in history[0]
    fig, axes = plt.subplots(3 if has_acp else 2, 2, figsize=(12, 11 if has_acp else 7),
                             constrained_layout=True)
    episodes = np.arange(1, len(history) + 1)
    axes = axes.ravel()
    axes[0].plot(episodes, [item["R_used"] for item in history], label="R used for training")
    if "R_hat_next" in history[0]:
        axes[0].plot(episodes, [item["R_hat_next"] for item in history], label="Updated R estimate")
    axes[0].set(title="Planning R", ylabel="Probability", ylim=(0, 1))
    scores = np.asarray([item["score"] for item in history], dtype=float)
    radii = np.asarray([item["C_used"] for item in history])
    finite_radii = np.where(np.isfinite(radii), radii, np.nan)
    ceiling = max(1.0, np.nanmax(scores, initial=0),
                  np.max(radii[np.isfinite(radii)], initial=0)) * 1.1
    axes[1].plot(episodes, scores, label="Maximum deviation")
    axes[1].plot(episodes, finite_radii, label="Conformal radius" if has_acp else "Fixed radius")
    if "C_star_k" in history[0]:
        support_bounds = [item["C_star_k"] for item in history]
        axes[1].plot(episodes, support_bounds, "k:", label="True C*")
        ceiling = max(ceiling, max(support_bounds) * 1.1)
    if np.any(np.isposinf(radii)):
        axes[1].scatter(
            episodes[np.isposinf(radii)], np.full(np.isposinf(radii).sum(), ceiling),
            marker="^", label="Unbounded prediction",
        )
    axes[1].set(title="Continuous uncertainty", ylabel="Distance")
    if has_acp:
        axes[2].plot(episodes, [item["delta_k"] for item in history], label="Adaptive delta")
        axes[2].axhline(delta, color="black", linestyle="--", label="Target failure")
        axes[2].set(title="ACP update", ylabel="Failure parameter")
        miscoverage = np.asarray([item["miscoverage"] for item in history], dtype=float)
        coverage = 1 - np.cumsum(miscoverage) / episodes
        axes[3].plot(episodes, coverage, label="Conformal coverage")
        axes[3].axhline(1 - delta, color="black", linestyle="--", label="Coverage target")
        axes[3].set(title="Cumulative empirical coverage", ylabel="Fraction", ylim=(0, 1.05))
    returns_axis, residual_axis = axes[-2:]
    returns = [item["discounted_return"] for item in history]
    returns_axis.plot(episodes, returns, label="Discounted r(s,a)")
    for key, marker, color in (
        ("goal_reached", "o", "green"), ("collision", "x", "red"),
        ("workspace_exit", "x", "purple"),
    ):
        indices = np.flatnonzero([item[key] for item in history])
        returns_axis.scatter(episodes[indices], np.asarray(returns)[indices], marker=marker,
                             color=color, label=key.replace("_", " "))
    returns_axis.set(title="Deployment outcomes", ylabel="Return")
    residuals = [max(item["training"]["bellman_residual"], 1e-15) for item in history]
    residual_axis.semilogy(episodes, residuals, label="Robust Bellman residual")
    residual_axis.set(title="Finite training accuracy", ylabel="Residual")
    for axis in axes:
        axis.set_xlabel("Deployment episode")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    return fig


def plot_acp(runs):
    """Show episode scores, radii, and adaptive delta for the first trial."""
    import matplotlib.pyplot as plt

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


def plot_R(runs, R_star=None):
    """Compare cumulative R estimates with the common true R-star."""
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    estimates = []
    if R_star is None:
        R_star = [run["R_star"] for run in runs]
    for run in runs:
        estimate = np.asarray([row["R_hat_next"] for row in run["history"]])
        estimates.append(estimate)
        axis.plot(np.arange(1, len(estimate) + 1), estimate, color="C0", alpha=0.25)
    episodes = np.arange(1, len(estimates[0]) + 1)
    axis.plot(episodes, np.mean(estimates, axis=0), label=r"Estimated $\hat{R}_{k+1}$")
    axis.axhline(np.mean(R_star), color="black", linestyle=":", label=r"True $R^\star$")
    axis.set(title="R estimation", xlabel="Deployment episode k", ylabel="Probability")
    return fig


def plot_performance(runs, method_labels, *, return_ylim=None):
    """Compare recorded returns and outcomes; optionally zoom returns with a full-range inset."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
    methods = list(dict.fromkeys(run["method"] for run in runs))
    outcomes = {"goal_reached": "Goal", "collision": "Obstacle", "workspace_exit": "Exit",
                "timeout": "Step limit"}
    width = 0.8 / len(methods)
    for index, method in enumerate(methods):
        color = f"C{index}"
        label = method_labels[method]
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
    if return_ylim is not None:
        inset = axes[0].inset_axes([0.25, 0.62, 0.24, 0.30])
        for line in axes[0].lines:
            inset.plot(
                line.get_xdata(), line.get_ydata(), color=line.get_color(),
                alpha=line.get_alpha(), linewidth=line.get_linewidth(),
            )
        inset.set_title("Full range", fontsize=9)
        inset.tick_params(labelsize=7)
        inset.grid(alpha=0.2)
        axes[0].set_ylim(*return_ylim)
    return fig
