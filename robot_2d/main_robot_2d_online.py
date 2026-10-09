"""Run Algorithm 2 or its nominal/global baselines, then save the observations."""

import pickle
import sys
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformally_robust_controller import ConformallyRobustController
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP


def evaluate_policy(mdp, policy, gamma, n_episodes, rng):
    """Evaluate a frozen policy with fresh random rollouts; do not update R or ACP."""
    records = [mdp.deploy_policy(policy, gamma, int(rng.integers(0, 2**32 - 1)))
               for _ in range(n_episodes)]
    returns = np.asarray([row["discounted_return"] for row in records])
    goals = np.asarray([row["goal_reached"] for row in records])
    collisions = np.asarray([row["collision"] for row in records])
    exits = np.asarray([row["workspace_exit"] for row in records])
    arrivals = [gamma**(row["goal_step"] - 1) if row["goal_reached"] else 0 for row in records]
    ordered = np.sort(returns)
    tail_mass = 0.1 * n_episodes
    complete = int(tail_mass)
    tail_total = np.sum(ordered[:complete]) + (tail_mass - complete) * ordered[complete]
    return {
        "success_rate": float(np.mean(goals)), "collision_rate": float(np.mean(collisions)),
        "workspace_exit_rate": float(np.mean(exits)),
        "timeout_rate": float(np.mean(~goals)),
        "discounted_return": float(np.mean(returns)),
        "discounted_goal_arrival": float(np.mean(arrivals)),
        "lower_tail_return": float(tail_total / tail_mass),
        "returns": returns, "goals": goals, "collisions": collisions, "workspace_exits": exits,
    }


def main(
    *, n_episodes=30, gamma=0.95, horizon=100, training_episodes=4000, initial_R=0.1,
    target_failure=0.1, initial_failure=None, eta=0.01, calibration_window=100,
    pilot_episodes=20, calibration_scores=None, seed=7, save_plots=True, output_dir=None,
    method="proposed", R=None, C=None, noise_probability=0.3, noise_radius=1.2,
    evaluation_episodes=0, verbose=True,
):
    """R/C=None selects online estimation/calibration; numeric R/C keeps it fixed.

    method='nominal' plans with R=0; method='global' plans with C=infinity.
    The stationary simulator noise law is not supplied to the learning agent.
    """
    if not isinstance(evaluation_episodes, (int, np.integer)) or evaluation_episodes < 0:
        raise ValueError("evaluation_episodes must be a nonnegative integer.")
    mdp = Robot2DMotionMDP(
        x_max=8.0, y_max=8.0, grid_shape=(8, 8),
        obstacles=[(2, 1), (5, 1), (1, 3), (4, 3), (6, 4), (2, 5), (5, 6)],
        start_cell=(0, 0), goal_cell=(7, 7), movement_reward=0.01, goal_reward=1.0,
        noise_probability=noise_probability, noise_radius=noise_radius,
        max_episode_steps=horizon, episode_mode="fixed_horizon", seed=seed,
    )
    controller = ConformallyRobustController(
        mdp, gamma=gamma, horizon=horizon, training_horizon=horizon,
        training_episodes=training_episodes, initial_R=initial_R,
        target_failure=target_failure, initial_failure=initial_failure, eta=eta,
        calibration_scores=calibration_scores, calibration_window=calibration_window,
        pilot_episodes=pilot_episodes, seed=seed, method=method, R=R, C=C,
    )
    controller.run(n_episodes, verbose=verbose)
    evaluation = None
    if evaluation_episodes:
        evaluation = evaluate_policy(
            controller.deployment_env, controller.policy, gamma,
            evaluation_episodes, controller.rng,
        )
    results = {
        "method": method, "seed": seed, "mdp": mdp, "gamma": gamma,
        "fixed_R": R, "fixed_C": C,
        "target_failure": target_failure, "initial_failure": initial_failure,
        "eta": eta, "calibration_window": calibration_window,
        "pilot_scores": controller.pilot_scores,
        "initial_calibration_scores": controller.initial_calibration_scores,
        "final_R": controller.R_estimate, "final_delta": controller.calibrator.delta,
        "history": controller.history, "evaluation": evaluation,
        "Q": controller.Q, "policy": controller.policy,
    }
    controller.results = results
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "saved_results" / "online"
        if method != "proposed":
            output_dir = output_dir / method
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.pkl").open("wb") as file:
        pickle.dump(results, file)
    np.savez_compressed(
        output_dir / "final_policy.npz", Q=controller.Q, policy=controller.policy,
        visit_counts=controller.agent.visit_counts,
    )
    if save_plots:
        import matplotlib
        matplotlib.use("Agg")
        from robot_2d.utils import plot_conformal_history, plot_robot_motion
        import matplotlib.pyplot as plt
        fig = plot_conformal_history(controller.history, controller.target_failure)
        fig.savefig(output_dir / "online_uncertainty.png", dpi=160)
        plt.close(fig)
        fig = plot_robot_motion(
            mdp, controller.Q, controller.policy, controller.history[-1]["positions"],
            f"Final {method} policy", source_positions=controller.history[-1]["source_positions"],
        )
        fig.savefig(output_dir / "final_policy.png", dpi=160)
        plt.close(fig)
    if verbose:
        print(f"Saved results to {output_dir.resolve()}")
    return controller


if __name__ == "__main__":
    # Change these ordinary settings for the numerical experiment.
    seeds = (7, 8, 9)
    methods = ("proposed", "nominal", "global")
    n_episodes = 30
    training_episodes = 4000
    pilot_episodes = 100
    evaluation_episodes = 500
    output_dir = Path(__file__).resolve().parent / "saved_results" / "online_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        shared_scores = None
        for index, method in enumerate(methods):
            controller = main(
                method=method, seed=seed + 100000 * index, save_plots=False,
                n_episodes=n_episodes, training_episodes=training_episodes,
                pilot_episodes=pilot_episodes, calibration_scores=shared_scores,
                evaluation_episodes=evaluation_episodes,
                output_dir=output_dir / f"{method}_seed_{seed}",
            )
            shared_scores = controller.initial_calibration_scores
            controller.results["trial_seed"] = seed
            runs.append(controller.results)
    with (output_dir / "comparison.pkl").open("wb") as file:
        pickle.dump(runs, file)
    from robot_2d.plot_online_comparison import plot_comparison
    plot_comparison(output_dir / "comparison.pkl")
