"""Shared tabular experiment statistics and plots."""

import numpy as np


def print_and_save_summary(output_path, print_summary, *args, **kwargs):
    """Print a final summary and optionally save the same output as plain text."""
    from contextlib import redirect_stdout
    from io import StringIO
    from pathlib import Path

    output = StringIO()
    with redirect_stdout(output):
        result = print_summary(*args, **kwargs)
    summary = output.getvalue()
    print(summary, end="")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(summary, encoding="utf-8")
    return result


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


def load_online_results(output_dir, method_settings):
    """Read saved runs matching the methods and uncertainty settings being plotted."""
    import pickle
    from pathlib import Path

    runs = []
    for method, (R, C, _) in method_settings.items():
        for filename in sorted(Path(output_dir).glob(f"results_{method}_R{R}_C{C}_trial_*.pkl")):
            with filename.open("rb") as file:
                runs.append(pickle.load(file))
    return runs


def _format_mean_std(values):
    """Format sample standard deviation; one trial cannot estimate that deviation."""
    spread = f"{np.std(values, ddof=1):.4f}" if len(values) > 1 else "N/A"
    return f"{np.mean(values):.4f} +/- {spread}"


def _acp_coverage_data(acp_runs):
    """Compute trial coverage curves and their common finite-K lower bound."""
    settings = np.asarray([
        (run["delta"], run["eta"], run["history"][0]["delta_k"]) for run in acp_runs
    ])
    if not np.allclose(settings, settings[0], rtol=0, atol=1e-12):
        raise ValueError("ACP coverage comparison requires common delta, eta, and delta_1.")
    delta, eta, delta_1 = settings[0]
    indicators = np.asarray([
        [row["score"] <= row["C_hat"] for row in run["history"]] for run in acp_runs
    ])
    episodes = np.arange(1, indicators.shape[1] + 1)
    cumulative_coverage = np.cumsum(indicators, axis=1) / episodes
    lower_bound = 1 - delta - (eta + delta_1) / (episodes * eta)
    return episodes, cumulative_coverage, lower_bound, settings[0]


def print_online_summary(runs, method_labels):
    """Print setup, policy performance, training convergence, R accuracy, and ACP coverage."""
    mdp = runs[0]["mdp"]
    episode_counts = [len(run["history"]) for run in runs]
    episodes = str(min(episode_counts))
    if min(episode_counts) != max(episode_counts):
        episodes += f"..{max(episode_counts)}"
    print(f"\n{len(mdp.grid_shape)}D online experiment")
    print(f"Grid: {mdp.grid_shape}; cell widths: {tuple(map(float, mdp.cell_widths))}")
    print(f"Start: {tuple(map(float, mdp.coordinates(mdp.start_state)))}; "
          f"goal: {tuple(map(float, mdp.coordinates(mdp.goal_state)))}")
    print(f"Obstacles: {mdp.obstacle_mask.sum()}; gamma: {mdp.gamma:g}; "
          f"transition limit: {runs[0]['horizon']}")
    if "map_seed" in runs[0]:
        print(f"Map seed: {runs[0]['map_seed']}")
    print(f"Deployment episodes/trial: {episodes}; saved runs: {len(runs)}")
    print(f"True R*: {runs[0]['R_star']:g}")
    print("Returns: mean +/- sample SD of trial averages; goal/hit rates pool all episodes.")
    print(f"{'Method':<28} {'Trials':>6} {'Episodes':>9} "
          f"{'Return mean +/- SD':>22} {'Goal (%)':>10} {'Obstacle (%)':>9}")
    for method in dict.fromkeys(run["method"] for run in runs):
        selected = [run for run in runs if run["method"] == method]
        returns = [np.mean([row["discounted_return"] for row in run["history"]])
                   for run in selected]
        history = [row for run in selected for row in run["history"]]
        goal = 100 * np.mean([row["goal_reached"] for row in history])
        hit = 100 * np.mean([row["collision"] for row in history])
        print(f"{method_labels[method]:<28} {len(selected):>6} {len(history):>9,} "
              f"{_format_mean_std(returns):>22} {goal:>10.1f} {hit:>9.1f}")

    training_records = [row["training"] for run in runs for row in run["history"]
                        if not row["training_reused"]]
    converged_run_count = sum(training_record["converged"] for training_record in training_records)
    max_bellman_residual = max(training_record["bellman_residual"]
                              for training_record in training_records)
    convergence_tols = ", ".join(f"{value:g}" for value in sorted({
        training_record.get("convergence_tol", training_record.get("tolerance"))
        for training_record in training_records
    }))
    print(f"Q-learning runs converged: {converged_run_count:,}/{len(training_records):,}; "
          f"convergence_tol={convergence_tols}; maximum residual={max_bellman_residual:.6g}")
    r_estimation_runs = [run for run in runs if run["fixed_R"] is None]
    if r_estimation_runs:
        final_R = [run["history"][-1]["R_hat_next"] for run in r_estimation_runs]
        print(f"Final R estimate: {_format_mean_std(final_R)} "
              f"across {len(r_estimation_runs)} trials; "
              f"initial estimate={r_estimation_runs[0]['history'][0]['R_used']:g}")
    acp_runs = [run for run in runs if run["fixed_C"] is None]
    if acp_runs:
        episodes, cumulative_coverage, lower_bound, settings = _acp_coverage_data(acp_runs)
        delta, eta, delta_1 = settings
        print(f"ACP cumulative coverage at K={episodes[-1]}: "
              f"{_format_mean_std(cumulative_coverage[:, -1])} "
              "(trial mean +/- sample SD)")
        print(f"ACP target={1 - delta:.4f}; finite-K lower bound={lower_bound[-1]:.4f}")
        print(f"ACP settings: delta={delta:g}, eta={eta:g}, delta_1={delta_1:g}")

def _set_plot_limits(axes, *, xlim=None, ylim=None):
    """Set optional limits; dict keys select zero-based axes."""
    for index, axis in enumerate(np.atleast_1d(axes).flat):
        x_limits = xlim.get(index) if isinstance(xlim, dict) else xlim
        y_limits = ylim.get(index) if isinstance(ylim, dict) else ylim
        if x_limits is not None:
            axis.set_xlim(x_limits)
        if y_limits is not None:
            axis.set_ylim(y_limits)


def _plot_acp_scores_quantile(run, axis):
    """Draw first-trial scores and ACP predictions on the supplied axis."""
    history = run["history"]
    episodes = np.arange(1, len(history) + 1)
    radii = np.asarray([row["C_hat"] for row in history])
    scores = np.asarray([row["score"] for row in history], dtype=float)
    axis.plot(episodes, scores, ".", label="nonconformity scores")
    axis.plot(
        episodes, np.where(np.isfinite(radii), radii, np.nan), label=r"ACP quantile $\hat{C}_k$",
    )
    finite_radii = radii[np.isfinite(radii)]
    ceiling = 1.1 * max(1, np.nanmax(scores, initial=0), np.max(finite_radii, initial=0))
    unbounded = np.isposinf(radii)
    if np.any(unbounded):
        axis.scatter(
            episodes[unbounded], np.full(sum(unbounded), ceiling), marker="^",
            label=r"$\hat{C}_k=\infty$",
        )
    axis.set(title="Scores and predictions, first trial",
             xlabel="Deployment episode k", ylabel="Distance")


def plot_acp(runs, *, xlim=None, ylim=None):
    """Return independent score/quantile and adaptive-delta figures for the first trial."""
    import matplotlib.pyplot as plt

    scores_quantile, scores_axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    delta, delta_axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    run = runs[0]
    _plot_acp_scores_quantile(run, scores_axis)
    history = run["history"]
    episodes = np.arange(1, len(history) + 1)

    delta_axis.plot(episodes, [row["delta_k"] for row in history], label=r"$\delta_k$")
    delta_axis.axhline(run["delta"], color="black", linestyle="--", label=r"Target $\delta$")
    delta_axis.set(title="Adaptive failure parameter, first trial",
                   xlabel="Deployment episode k", ylabel="Failure parameter")

    _set_plot_limits((scores_axis, delta_axis), xlim=xlim, ylim=ylim)
    return scores_quantile, delta


def plot_acp_empirical_coverage(runs, *, xlim=None, ylim=None, axis=None):
    """Plot cumulative episode coverage across trials against the finite-K lower bound."""
    import matplotlib.pyplot as plt

    episodes, cumulative_coverage, lower_bound, settings = _acp_coverage_data(runs)
    delta = settings[0]
    if axis is None:
        _, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.plot(episodes, cumulative_coverage.T, color="C0", alpha=0.2)
    axis.plot(episodes, cumulative_coverage.mean(axis=0), color="C0", label="Mean coverage")
    axis.axhline(1 - delta, color="black", linestyle="--", label=r"Target $1-\delta$")
    axis.plot(episodes, lower_bound, color="C3", linestyle=":", label="Theoretical lower bound")
    axis.set(title="ACP empirical coverage", xlabel="Deployment episodes K", ylabel="Coverage")
    _set_plot_limits(axis, xlim=xlim, ylim=ylim)
    return axis.figure


def plot_R(runs, R_star=None, *, xlim=None, ylim=None, axis=None):
    """Compare cumulative R estimates with the common true R-star."""
    import matplotlib.pyplot as plt

    if axis is None:
        _, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
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
    return axis.figure


def plot_R_acp(r_estimation_runs, acp_runs, *, xlim=None, ylim=None):
    """Combine R estimates and cumulative ACP coverage across trials in a 1-by-2 figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    plot_R(r_estimation_runs, axis=axes[0])
    plot_acp_empirical_coverage(acp_runs, axis=axes[1])
    _set_plot_limits(axes, xlim=xlim, ylim=ylim)
    return fig


def plot_performance(runs, method_labels, *, xlim=None, ylim=None):
    """Plot averaged cumulative returns; print_online_summary reports the outcome statistics."""
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
    axis.set(title="Cumulative mean discounted reward", xlabel="Online episode", ylabel="Reward")
    _set_plot_limits(axis, xlim=xlim, ylim=ylim)
    return fig
