"""Run Algorithm 2 or its nominal/R-contamination baselines, then save the observations."""

import pickle
import sys
from functools import partial
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformally_robust_controller import ConformallyRobustController
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP
from robot_2d.utils_2d import sample_deviation_2d

METHOD_SETTINGS = {
    "proposed": (None, None, "Proposed"),
    "nominal": (0.0, 0.0, "Nominal"),
    "R_contamination_0.1": (0.1, np.inf, "R-contamination (R=0.1)"),
    "R_contamination_0.5": (0.5, np.inf, "R-contamination (R=0.5)"),
}


if __name__ == "__main__":
    mdp = Robot2DMotionMDP(
        x_max=10.0, y_max=10.0, grid_shape=(10, 10),
        start_cell=(1, 1), goal_cell=(7, 6), gamma=0.9, movement_reward=0.0, goal_reward=1.0,
        obstacles=[
            (0, 7), (0, 8), (0, 9),
            (1, 8), (1, 9),
            (2, 9),
            (4, 1), (4, 3), (4, 4),
            (5, 0), (5, 1), (5, 3), (5, 4), (5, 5), (5, 6), (5, 9),
            (6, 0), (6, 3), (6, 4),
            (8, 0), (8, 1), (8, 4), (8, 9),
            (9, 3), (9, 4), (9, 5), (9, 8), (9, 9)
        ],
    )

    n_deployment_episodes = 100
    horizon = 100
    lr_init = 0.9
    step_start_decay_lr = 40000
    epsilon_init = 0.1
    epsilon_lb = 0.01
    epsilon_decay_rate = 0.999
    convergence_tol = 1e-5
    max_training_episodes = 2000000
    check_every = 100
    R_hat_init = 0.1
    delta = 0.1
    delta_1 = None
    eta = 0.02
    calibration_window = 100
    quantile_max = 1.75
    pilot_episodes = 100
    calibration_scores = None
    save_plots = False
    R = None
    C = None
    R_star = 0.4
    C_star = 1.3
    min_radius = np.linalg.norm(mdp.cell_widths) / 2
    verbose = True
    seeds = (23, 67, 87, 101, 922, 1009, 2027, 3037, 4049, 5051,
             6067, 7079, 8089, 9091, 10093)
    first_trial = 1
    methods = ("proposed", "nominal", "R_contamination_0.1", "R_contamination_0.5")

    checkpoint_path = Path(__file__).resolve().parent / "saved_results/q_transformer/checkpoint"
    q_initializer = None
    if checkpoint_path is not None:
        import torch
        from q_transformer.model import QTransformerPredictor
        torch.set_num_threads(4)
        q_initializer = QTransformerPredictor(checkpoint_path)

    if not np.allclose(mdp.cell_widths, 1.0, rtol=0, atol=1e-12):
        raise ValueError("This directional-noise experiment requires unit grid cells.")
    if not np.isfinite(R_star) or not 0 < R_star < 1:
        raise ValueError("R_star must be in (0, 1) for the directional-noise experiment.")

    for trial, trial_seed in enumerate(seeds, first_trial):
        for index, method in enumerate(methods):
            seed = trial_seed + 100000 * index
            if method not in METHOD_SETTINGS:
                raise ValueError(f"Unknown method: {method}.")
            # Each run owns one peak-angle sequence, continued across all its episodes.
            mdp.noise_sampler = partial(
                sample_deviation_2d, radius=C_star, R_star=R_star, min_radius=min_radius,
                peak_angles=2 * np.pi * np.arange(5) / 5,
            )
            fixed_R, fixed_C = (R, C) if method == "proposed" else METHOD_SETTINGS[method][:2]
            controller = ConformallyRobustController(
                mdp, horizon=horizon, training_horizon=horizon,
                max_training_episodes=max_training_episodes, R_hat_init=R_hat_init,
                convergence_tol=convergence_tol, training_check_every=check_every,
                lr_init=lr_init, step_start_decay_lr=step_start_decay_lr,
                epsilon_init=epsilon_init, epsilon_lb=epsilon_lb,
                epsilon_decay_rate=epsilon_decay_rate,
                delta=delta, delta_1=delta_1, eta=eta, quantile_max=quantile_max,
                calibration_scores=calibration_scores, calibration_window=calibration_window,
                pilot_episodes=pilot_episodes, seed=seed, R=fixed_R, C=fixed_C,
                q_initializer=q_initializer,
                accelerated_training=True,
            )
            controller.run(n_deployment_episodes, verbose=verbose)
            deployment_mdp = controller.deployment_env
            results = {
                "method": method, "seed": seed, "trial": trial, "mdp": deployment_mdp,
                "horizon": horizon,
                "R_star": R_star,
                "C_star": C_star,
                "min_radius": min_radius,
                "fixed_R": controller.fixed_R, "fixed_C": controller.fixed_C,
                "history": controller.history,
                "Q": controller.Q, "policy": controller.policy,
                "visit_counts": controller.agent.visit_counts,
            }
            if controller.fixed_R is None:
                results["final_R"] = controller.R_hat
            if controller.acp is not None:
                results.update({
                    "delta": delta, "delta_1": delta_1,
                    "eta": eta, "calibration_window": calibration_window,
                    "quantile_max": quantile_max,
                    "pilot_scores": controller.pilot_scores,
                    "initial_calibration_scores": controller.initial_calibration_scores,
                    "final_delta": controller.acp.delta_k,
                })
            output_dir = Path(__file__).resolve().parent / "saved_results/online"
            output_dir.mkdir(parents=True, exist_ok=True)
            test_name = f"{method}_R{controller.fixed_R}_C{controller.fixed_C}_trial_{trial}"
            filename = output_dir / f"results_{test_name}.pkl"
            with filename.open("wb") as file:
                pickle.dump(results, file)

            if save_plots:
                import matplotlib
                matplotlib.use("Agg")
                from robot_2d.utils_2d import plot_robot_motion
                import matplotlib.pyplot as plt
                fig = plot_robot_motion(
                    deployment_mdp, controller.Q, controller.policy,
                    controller.history[-1]["positions"],
                    f"Final {method} policy",
                )
                fig.savefig(output_dir / f"final_policy_{test_name}.png", dpi=160)
                plt.close(fig)
            if verbose:
                print(f"Saved results to {filename.resolve()}")
