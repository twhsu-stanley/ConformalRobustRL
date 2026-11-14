"""Run Algorithm 2 or its nominal/R-contamination baselines, then save the observations."""

import pickle
import sys
from functools import partial
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformally_robust_controller import ConformallyRobustController
from generate_grid_maps import training_maps
from robot_3d.robot_3d_motion_mdp import Robot3DMotionMDP
from robot_3d.utils_3d import sample_wind_deviation, wind_support_3d

METHOD_SETTINGS = {
    "proposed": (None, None, "Proposed"),
    "nominal": (0.0, 0.0, "Nominal"),
    "R_contamination_0.1": (0.1, np.inf, "R-contamination (R=0.1)"),
    "R_contamination_0.5": (0.5, np.inf, "R-contamination (R=0.5)"),
}


if __name__ == "__main__":
    map_seed = 23
    mdp = next(training_maps(
        Robot3DMotionMDP(
            x_max=10.0, y_max=10.0, z_max=10.0, grid_shape=(10, 10, 10),
            start_cell=(1, 1, 1), goal_cell=(8, 8, 8),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0,
        ),
        1, n_obstacles=(250, 400), seed=map_seed,
    ))

    n_deployment_episodes = 100
    horizon = 100
    max_training_episodes = 200000
    R_hat_init = 0.1
    delta = 0.1
    delta_1 = None
    eta = 0.1
    calibration_window = 100
    quantile_max = 3.0
    pilot_episodes = 100
    calibration_scores = None
    save_plots = False
    R = None
    C = None
    R_star = 0.3
    shift_period = 100
    verbose = True
    training_tolerance = 1e-6
    training_check_every = 1000
    seeds = (23, 67, 87, 101, 922)
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
        raise ValueError("This shifting-noise experiment requires unit grid cells.")
    if not np.isfinite(R_star) or not 0 < R_star < 1:
        raise ValueError("R_star must be in (0, 1) for the shifting-support experiment.")

    def prepare_episode(controller, episode):
        radius = wind_support_3d(episode, shift_period)
        controller.deployment_env.noise_radius = radius
        controller.deployment_env.noise_sampler = partial(
            sample_wind_deviation, radius=radius, R_star=R_star,
        )
        controller.deployment_env.is_nominal = False
        return {"C_star_k": radius}

    for trial, trial_seed in enumerate(seeds, first_trial):
        for index, method in enumerate(methods):
            seed = trial_seed + 100000 * index
            if method not in METHOD_SETTINGS:
                raise ValueError(f"Unknown method: {method}.")
            fixed_R, fixed_C = (R, C) if method == "proposed" else METHOD_SETTINGS[method][:2]
            controller = ConformallyRobustController(
                mdp, horizon=horizon, training_horizon=horizon,
                max_training_episodes=max_training_episodes, R_hat_init=R_hat_init,
                step_start_decay_lr=10000000,
                training_tolerance=training_tolerance, training_check_every=training_check_every,
                delta=delta, delta_1=delta_1, eta=eta, quantile_max=quantile_max,
                calibration_scores=calibration_scores, calibration_window=calibration_window,
                pilot_episodes=pilot_episodes, seed=seed, R=fixed_R, C=fixed_C,
                episode_setup=prepare_episode, q_initializer=q_initializer,
                accelerated_training=True,
            )
            controller.run(n_deployment_episodes, verbose=verbose)
            deployment_mdp = controller.deployment_env
            results = {
                "method": method, "seed": seed, "trial": trial, "mdp": deployment_mdp,
                "horizon": horizon,
                "R_star": R_star, "shift_period": shift_period,
                "map_seed": map_seed, "disturbance_model": "bounded_integrated_wind",
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
                from robot_3d.utils_3d import plot_robot_motion
                from utils import plot_conformal_history
                import matplotlib.pyplot as plt
                if controller.fixed_R is None or controller.acp is not None:
                    fig = plot_conformal_history(controller.history, controller.delta)
                    fig.savefig(output_dir / f"online_uncertainty_{test_name}.png", dpi=160)
                    plt.close(fig)
                fig = plot_robot_motion(
                    deployment_mdp, controller.Q, controller.policy,
                    controller.history[-1]["positions"],
                    f"Final {method} policy",
                )
                fig.savefig(output_dir / f"final_policy_{test_name}.png", dpi=160)
                plt.close(fig)
            if verbose:
                print(f"Saved results to {filename.resolve()}")
