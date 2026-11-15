"""Shared robot disturbances, tabular experiment statistics, and plots."""

import numpy as np


def sample_deviation(nominal_position, rng, *, radius, R_star, min_radius=None):
    """Mix zero error with a uniform length and a uniform direction in 2-D or 3-D.

    The nonzero length is uniform on [min_radius, radius], rather than uniform
    over the shell's area or volume. Normalizing independent Gaussian coordinates
    gives a uniform direction. The default inner radius is the unit cell's half
    diagonal, sqrt(d)/2. A continuous length draw exceeds this bound almost surely,
    so every active disturbance changes the discrete successor almost surely.
    Only the dimension of nominal_position is used; the law is source-independent.
    """
    dimension = len(nominal_position)
    if dimension not in (2, 3):
        raise ValueError("The robot disturbance requires a 2-D or 3-D position.")
    if not np.isfinite(R_star) or not 0 <= R_star <= 1:
        raise ValueError("R_star must be in [0, 1].")
    half_diagonal = np.sqrt(dimension) / 2
    min_radius = half_diagonal if min_radius is None else min_radius
    if not np.isfinite(min_radius) or min_radius < half_diagonal:
        raise ValueError("min_radius must be at least the unit cell's half diagonal.")
    if not np.isfinite(radius) or radius <= min_radius:
        raise ValueError("The outer radius must be finite and exceed min_radius.")
    if rng.random() >= R_star:
        return np.zeros(dimension)
    direction = rng.normal(size=dimension)
    length = np.linalg.norm(direction)
    while length == 0:
        direction = rng.normal(size=dimension)
        length = np.linalg.norm(direction)
    return rng.uniform(min_radius, radius) * direction / length


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


def _set_plot_limits(axes, *, xlim=None, ylim=None):
    """Set optional limits; dict keys select zero-based subplot indices."""
    for index, axis in enumerate(np.atleast_1d(axes).flat):
        x_limits = xlim.get(index) if isinstance(xlim, dict) else xlim
        y_limits = ylim.get(index) if isinstance(ylim, dict) else ylim
        if x_limits is not None:
            axis.set_xlim(x_limits)
        if y_limits is not None:
            axis.set_ylim(y_limits)


def plot_acp(runs, *, xlim=None, ylim=None):
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
    if "C_star_k" in history[0]:
        support_bounds = [row["C_star_k"] for row in history]
        axes[0].plot(episodes, support_bounds, "k:", label=r"True support bound $C^\star$")
    finite_radii = radii[np.isfinite(radii)]
    ceiling = max(1, np.nanmax(scores, initial=0), np.max(finite_radii, initial=0))
    if "C_star_k" in history[0]:
        ceiling = max(ceiling, max(support_bounds))
    ceiling *= 1.1
    unbounded = np.isposinf(radii)
    if np.any(unbounded):
        axes[0].scatter(
            episodes[unbounded], np.full(sum(unbounded), ceiling), marker="^",
            label=r"$\hat{C}_k=\infty$",
        )
    axes[0].set(title="Scores and predictions, first trial",
                xlabel="Deployment episode k", ylabel="Distance")

    axes[1].plot(episodes, [row["delta_k"] for row in history], label=r"$\delta_k$")
    axes[1].axhline(run["delta"], color="black", linestyle="--", label=r"Target $\delta$")
    axes[1].set(title="Adaptive failure parameter, first trial",
                    xlabel="Deployment episode k", ylabel="Failure parameter")

    _set_plot_limits(axes, xlim=xlim, ylim=ylim)
    return fig


def plot_R(runs, R_star=None, *, xlim=None, ylim=None):
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
    _set_plot_limits(axis, xlim=xlim, ylim=ylim)
    return fig


def plot_performance(runs, method_labels, *, xlim=None, ylim=None):
    """Plot averaged cumulative returns and print goal-reaching and obstacle-hitting rates."""
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    methods = list(dict.fromkeys(run["method"] for run in runs))
    for index, method in enumerate(methods):
        color = f"C{index}"
        label = method_labels[method]
        histories = [run["history"] for run in runs if run["method"] == method]
        returns = np.array([[row["discounted_return"] for row in history] for history in histories])
        episodes = np.arange(1, returns.shape[1] + 1)
        averages = np.cumsum(returns, axis=1) / episodes
        axis.plot(episodes, averages.T, color=color, alpha=0.2)
        axis.plot(episodes, np.mean(averages, axis=0), color=color, label=label)
        goal_rate = np.mean([row["goal_reached"] for history in histories for row in history])
        collision_rate = np.mean([row["collision"] for history in histories for row in history])
        print(f"{label}: goal-reaching rate = {goal_rate:.1%}, "
              f"obstacle-hitting rate = {collision_rate:.1%}")
    axis.set(title="Cumulative mean discounted reward", xlabel="Online episode", ylabel="Reward")
    _set_plot_limits(axis, xlim=xlim, ylim=ylim)
    return fig
