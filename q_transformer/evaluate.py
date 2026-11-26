"""Compare initial Q errors and convergence updates on fresh map/R/C cases."""

from pathlib import Path

import numpy as np

from generate_grid_maps import training_maps
from q_transformer.data import MDP_SHARED_FIELDS, PROMPT_DECIMALS, mdp_data
from Tabular_Agent import Tabular_Agent
from utils import print_and_save_summary


ALGORITHMS = ("robust_value_iteration", "robust_q_learning")
ERROR_METRICS = ("initial_error_inf", "initial_error_rmse")


def _map_key(start, goal, obstacles):
    return tuple(start), tuple(goal), np.asarray(obstacles, dtype=bool).tobytes()


def sample_test_cases(template, dataset_path, *, n_cases, n_obstacles, seed):
    """Sample unseen maps and uniformly sample the two-decimal R/C grid in dataset bounds."""
    if not isinstance(n_cases, (int, np.integer)) or isinstance(n_cases, bool) or n_cases < 1:
        raise ValueError("n_cases must be a positive integer.")
    parameters = mdp_data(template)
    with np.load(dataset_path, allow_pickle=False) as data:
        if any(not np.array_equal(data[k], parameters[k]) for k in MDP_SHARED_FIELDS):
            raise ValueError("Template geometry, rewards and discount must match the dataset.")
        bounds = np.array([(data[k].min(), data[k].max()) for k in ("R", "C")])
        excluded = {_map_key(s, g, o) for s, g, o in zip(
            data["start_states"], data["goal_states"], data["obstacle_masks"],
        )}
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Uniform sampling requires finite dataset R/C bounds.")
    for R, C in bounds.T:
        Tabular_Agent._validate_uncertainty(R, C)
    scale = 10**PROMPT_DECIMALS
    low, high = np.ceil(bounds[:, 0] * scale), np.floor(bounds[:, 1] * scale)
    if np.any(low > high):
        raise ValueError("The dataset bounds contain no values at the prompt's precision.")
    map_stream, parameter_stream, learning_stream = np.random.SeedSequence(seed).spawn(3)
    parameter_rng = np.random.default_rng(parameter_stream)
    learning_rng = np.random.default_rng(learning_stream)
    map_seed = int(map_stream.generate_state(1)[0])
    cases = []
    maps = training_maps(template, n_cases + len(excluded), n_obstacles=n_obstacles, seed=map_seed)
    for mdp in maps:
        key = _map_key(mdp.start_state, mdp.goal_state, mdp.obstacle_mask)
        if key in excluded:
            continue
        excluded.add(key)
        R, C = parameter_rng.integers(low.astype(int), high.astype(int) + 1) / scale
        cases.append({
            "mdp": mdp, "R": float(R), "C": float(C),
            "seed": int(learning_rng.integers(2**32)),
        })
        if len(cases) == n_cases:
            return cases, bounds
    raise RuntimeError("Could not sample enough distinct maps outside the training dataset.")


def _q_errors(Q, reference, active):
    error = Q[active] - reference[active]
    return dict(initial_error_inf=float(np.abs(error).max()),
                initial_error_rmse=float(np.sqrt(np.mean(error**2))))


def compare_initializations(
    mdp, checkpoint_path, *, R, C, lr_init, step_start_decay_lr, epsilon_init, epsilon_lb,
    epsilon_decay_rate, seed, algorithm="robust_value_iteration", convergence_tol=1e-5,
    q_star_reference_tol=1e-8, max_iterations=10000, max_training_episodes=200000,
    check_every=100, predictor=None, accelerated=True, exploring_starts=True,
):
    """Measure paired solves against an independent, more accurate VI reference.

    Q-learning counts nominal training transitions; VI counts nonterminal
    state-action backups. Reference generation, inference and residual checks
    do not enter either update counter. Terminal Q entries are fixed at zero.
    Both initializations use the same Q-learning settings and episode-start mode.
    Exploring starts are the default; False resets to the configured start state.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}.")
    if not np.isfinite(q_star_reference_tol) or not 0 < q_star_reference_tol < convergence_tol:
        raise ValueError("q_star_reference_tol must be positive and stricter than convergence_tol.")
    if seed is None:
        seed = int(np.random.default_rng().integers(2**32))
    if predictor is None:
        from q_transformer.model import QTransformerPredictor

        predictor = QTransformerPredictor(checkpoint_path)
    reference_agent = Tabular_Agent(mdp.clone(nominal=True), R=R, C=C)
    reference, _ = reference_agent.robust_value_iteration(
        convergence_tol=q_star_reference_tol, max_iterations=max_iterations,
    )
    prediction = np.asarray(predictor(mdp, R, C), dtype=float).copy()
    if prediction.shape != reference.shape or not np.all(np.isfinite(prediction)):
        raise ValueError("The transformer must return a finite Q-table matching the MDP.")
    prediction[mdp.terminal_mask] = 0.0
    result = {"R": R, "C": C, "seed": seed,
              "reference_residual": reference_agent.diagnostics["bellman_residual"]}
    for name, initial in (("zero", np.zeros_like(reference)), ("warm", prediction)):
        agent = Tabular_Agent(mdp.clone(nominal=True, seed=seed), R=R, C=C)
        failure = None
        try:
            if algorithm == "robust_value_iteration":
                agent.robust_value_iteration(
                    initial_Q=initial, convergence_tol=convergence_tol,
                    max_iterations=max_iterations,
                )
            else:
                agent.robust_q_learning(
                    max_training_episodes, initial_Q=initial, convergence_tol=convergence_tol,
                    check_every=check_every, seed=seed, exploring_starts=exploring_starts,
                    record_every=max_training_episodes * mdp.max_episode_steps,
                    lr_init=lr_init, step_start_decay_lr=step_start_decay_lr,
                    epsilon_init=epsilon_init, epsilon_lb=epsilon_lb,
                    epsilon_decay_rate=epsilon_decay_rate, accelerated=accelerated,
                )
        except RuntimeError as error:
            if agent.diagnostics.get("converged") is not False:
                raise
            failure = str(error)
        result[name] = {
            **agent.diagnostics, **_q_errors(initial, reference, ~mdp.terminal_mask),
            "final_error_inf": float(np.abs(agent.Q - reference).max()), "failure": failure,
        }
    return result


def summarize_initializations(results):
    """Print mean, sample SD, medians and paired wins; retain all failed cases."""
    cases = results["cases"]
    complete = [c for c in cases if c["zero"]["converged"] and c["warm"]["converged"]]
    summary = {"complete_pairs": len(complete), "n_cases": len(cases)}
    configuration = results.get("configuration", {})
    if cases and cases[0]["zero"].get("algorithm") == "robust_q_learning":
        starts = "exploring starts" if cases[0]["zero"]["exploring_starts"] else "start state"
        print(f"Episode starts: {starts}.")
        keys = (
            "lr_init", "step_start_decay_lr", "epsilon_init", "epsilon_lb", "epsilon_decay_rate",
        )
        settings = ", ".join(f"{key}={configuration[key]}" for key in keys if key in configuration)
        if settings:
            print(f"Q-learning (both initializations): {settings}")
    print("\nErrors use nonterminal Q entries; terminal entries are fixed at zero.")
    print(f"Complete convergence pairs: {len(complete)}/{len(cases)}; failures "
          f"zero/warm: {sum(not c['zero']['converged'] for c in cases)}/"
          f"{sum(not c['warm']['converged'] for c in cases)}")
    print(f"{'Metric':<22} {'Zero mean +/- SD':>24} {'Warm mean +/- SD':>24} "
          f"{'Warm wins/ties/losses':>23}")
    for metric, label in zip((*ERROR_METRICS, "global_steps"),
                             ("Initial Q max error", "Initial Q RMSE", "Convergence steps")):
        rows = complete if metric == "global_steps" else cases
        if not rows:
            continue
        zero, warm = (np.array([c[m][metric] for c in rows]) for m in ("zero", "warm"))
        values = [f"{v.mean():.6g} +/- {v.std(ddof=1) if len(v) > 1 else 0:.3g}"
                  for v in (zero, warm)]
        ties = np.isclose(zero, warm, rtol=0, atol=1e-12)
        wins, losses = int(((warm < zero) & ~ties).sum()), int(((warm > zero) & ~ties).sum())
        summary[metric] = {
            m: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.,
                "median": float(np.median(v))} for m, v in (("zero", zero), ("warm", warm))
        }
        summary[metric].update(wins=wins, ties=int(ties.sum()), losses=losses)
        print(f"{label:<22} {values[0]:>24} {values[1]:>24} "
              f"{wins:>7}/{ties.sum()}/{losses}")
    if complete:
        zero = np.array([c["zero"]["global_steps"] for c in complete])
        warm = np.array([c["warm"]["global_steps"] for c in complete])
        reduction = float(100 * (1 - warm.sum() / zero.sum())) if zero.sum() else None
        summary["step_reduction_percent"] = reduction
        print(f"Median convergence steps, zero/warm: {np.median(zero):,.0f}/{np.median(warm):,.0f}")
        if reduction is not None:
            print(f"Aggregate step reduction: {reduction:+.2f}% (positive favors warm start).")
    else:
        print("No complete convergence pairs; step reduction is undefined.")
    print("Step statistics use complete pairs only; initial-error statistics use every case.")
    return summary


def plot_initialization_comparison(results):
    """Show paired errors and convergence work; points below the diagonal favor warm start."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for axis, metric, title in zip(axes, ("initial_error_rmse", "global_steps"),
                                   ("RMSE(Q_init, Q*)", "Total Steps to Convergence")):
        cases = results["cases"]
        if metric == "global_steps":
            cases = [c for c in cases if c["zero"]["converged"] and c["warm"]["converged"]]
        zero, warm = (np.array([c[m][metric] for c in cases]) for m in ("zero", "warm"))
        limit = max(float(zero.max(initial=0)), float(warm.max(initial=0))) * 1.05 or 1.0
        axis.scatter(zero, warm, s=25, alpha=.8)
        if metric == "global_steps" and not cases:
            axis.text(.5, .5, "No complete convergence pairs", transform=axis.transAxes,
                      ha="center", va="center")
        axis.plot([0, limit], [0, limit], "k--", linewidth=1)
        axis.set(title=f"{title} (n={len(cases)})", xlabel="Zero initialization",
                 ylabel="Transformer warm start", xlim=(0, limit), ylim=(0, limit))
        axis.set_aspect("equal", adjustable="box")
    figure.suptitle("Below the diagonal favors the transformer")
    return figure


def evaluate_initializations(
    template, dataset_path, checkpoint_path, *, n_cases=10, n_obstacles, seed=2026,
    algorithm="robust_value_iteration", convergence_tol=1e-5, q_star_reference_tol=1e-8,
    max_iterations=10000, max_training_episodes=200000, check_every=100,
    lr_init=.9, step_start_decay_lr=10000000, epsilon_init=.1, epsilon_lb=.01,
    epsilon_decay_rate=.999, accelerated=True, exploring_starts=True, output_dir=None,
):
    """Sample a configurable number of cases, load the predictor once, and report paired results."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}.")
    cases, bounds = sample_test_cases(
        template, dataset_path, n_cases=n_cases, n_obstacles=n_obstacles, seed=seed,
    )
    from q_transformer.model import QTransformerPredictor

    predictor = QTransformerPredictor(checkpoint_path)
    options = dict(
        algorithm=algorithm, convergence_tol=convergence_tol,
        q_star_reference_tol=q_star_reference_tol,
        max_iterations=max_iterations, max_training_episodes=max_training_episodes,
        check_every=check_every, lr_init=lr_init, step_start_decay_lr=step_start_decay_lr,
        epsilon_init=epsilon_init, epsilon_lb=epsilon_lb, epsilon_decay_rate=epsilon_decay_rate,
        accelerated=accelerated, exploring_starts=exploring_starts,
    )
    results = {"configuration": {**options, "n_cases": n_cases, "seed": seed,
                                 "R_range": bounds[0], "C_range": bounds[1]}, "cases": []}
    print(f"{algorithm}: {n_cases} unseen maps; R in {bounds[0]}, C in {bounds[1]}; "
          f"parameters sampled at {PROMPT_DECIMALS} decimals.")
    print(f"Convergence residual <= {convergence_tol:g}; "
          f"reference residual <= {q_star_reference_tol:g}.")
    print("Global steps count nominal transitions for Q-learning and individual backups for VI.")
    for index, case in enumerate(cases, 1):
        result = compare_initializations(checkpoint_path=checkpoint_path, predictor=predictor,
                                         **case, **options)
        results["cases"].append(result)
        zero, warm = result["zero"], result["warm"]
        print(f"Case {index:>3}/{n_cases}: R={case['R']:.2f}, C={case['C']:.2f}; "
              f"steps zero/warm={zero['global_steps']:,}/{warm['global_steps']:,}; "
              f"converged={zero['converged']}/{warm['converged']}", flush=True)
    summary_path = None
    if output_dir is not None:
        summary_path = Path(output_dir) / f"transformer_evaluation_{algorithm}.txt"
    results["summary"] = print_and_save_summary(summary_path, summarize_initializations, results)
    if output_dir is not None:
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        figure = plot_initialization_comparison(results)
        for extension in ("png", "pdf"):
            figure.savefig(output_dir / f"transformer_evaluation_{algorithm}.{extension}", dpi=180)
        plt.close(figure)
        print(f"Saved comparison plots to {output_dir}")
    return results
