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
from robot_2d.utils import C_star_shift, sample_deviation

METHOD_SETTINGS = {
    "proposed": (None, None, "Proposed"),
    "nominal": (0.0, 0.0, "Nominal"),
    "R_contamination_0.1": (0.1, np.inf, "R-contamination (R=0.1)"),
    "R_contamination_0.5": (0.5, np.inf, "R-contamination (R=0.5)"),
}


def main(
    mdp, n_deployment_episodes=30, gamma=0.95, horizon=100, training_episodes=4000, R_hat_init=0.1,
    delta=0.1, delta_1=None, eta=0.01, calibration_window=100, quantile_max = 3.0,
    pilot_episodes=20, calibration_scores=None, seed=7, save_plots=True,
    method="proposed", R=None, C=None, R_star=0.2, shift_period=100,
    verbose=True, trial=1,
):
    """R/C=None selects online estimation/calibration; numeric R/C keeps it fixed.

    method='nominal' plans with R=C=0. Methods 'R_contamination_0.1' and
    'R_contamination_0.5' fix R_used at 0.1 and 0.5, respectively, with C_used=infinity.
    R=None enables mismatch estimation; C=None enables ACP and calibration pilots.
    Fixed R/C disable the corresponding updates and allow policy reuse when both are fixed.
    horizon caps episode length; goal, collision, and exit end deployment earlier.
    The supplied obstacle map stays fixed throughout the run. Each episode has
    a fixed disturbance law whose support bound varies slowly within [0.5, 2.0],
    while R_star stays constant.
    These true simulator parameters are not supplied to the Q-learning agent.
    The trial loop passes the fixed map to every method and trial. Matching
    shift_period gives methods the same support schedule even when pilots are skipped.
    """
    if method not in METHOD_SETTINGS:
        raise ValueError(f"Unknown method: {method}.")
    if method != "proposed":
        R, C, _ = METHOD_SETTINGS[method]
    if not np.allclose(mdp.cell_widths, 1.0, rtol=0, atol=1e-12):
        raise ValueError("This shifting-noise experiment requires unit grid cells.")
    if not np.isfinite(R_star) or not 0 < R_star < 1:
        raise ValueError("R_star must be in (0, 1) for the shifting-support experiment.")

    def prepare_episode(controller, episode):
        radius = C_star_shift(episode, shift_period)
        controller.deployment_env.noise_radius = radius
        controller.deployment_env.noise_sampler = partial(
            sample_deviation, radius=radius, R_star=R_star,
        )
        controller.deployment_env.is_nominal = False
        return {"C_star_k": radius}

    controller = ConformallyRobustController(
        mdp, gamma=gamma, horizon=horizon, training_horizon=horizon,
        training_episodes=training_episodes, R_hat_init=R_hat_init,
        delta=delta, delta_1=delta_1, eta=eta, quantile_max=quantile_max,
        calibration_scores=calibration_scores, calibration_window=calibration_window,
        pilot_episodes=pilot_episodes, seed=seed, R=R, C=C,
        episode_setup=prepare_episode,
    )
    controller.run(n_deployment_episodes, verbose=verbose)
    mdp = controller.deployment_env
    results = {
        "method": method, "seed": seed, "trial": trial, "mdp": mdp,
        "gamma": gamma, "horizon": horizon,
        "R_star": R_star, "shift_period": shift_period,
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
        from robot_2d.utils import plot_conformal_history, plot_robot_motion
        import matplotlib.pyplot as plt
        if controller.fixed_R is None or controller.acp is not None:
            fig = plot_conformal_history(controller.history, controller.delta)
            fig.savefig(output_dir / f"online_uncertainty_{test_name}.png", dpi=160)
            plt.close(fig)
        fig = plot_robot_motion(
            mdp, controller.Q, controller.policy, controller.history[-1]["positions"],
            f"Final {method} policy",
        )
        fig.savefig(output_dir / f"final_policy_{test_name}.png", dpi=160)
        plt.close(fig)
    if verbose:
        print(f"Saved results to {filename.resolve()}")
    return controller


if __name__ == "__main__":
    mdp = Robot2DMotionMDP(
        x_max=10.0, y_max=10.0, grid_shape=(10, 10),
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
        start_cell=(1, 1), goal_cell=(7, 6), movement_reward=0.0, goal_reward=1.0,
    )

    seeds = (23, 67, 87, 101, 922)
    methods = ("proposed", "nominal", "R_contamination_0.1", "R_contamination_0.5")
    n_deployment_episodes = 100
    training_episodes = 5000
    pilot_episodes = 100
    R_star = 0.4
    shift_period = 100

    for trial, seed in enumerate(seeds, 1):
        for index, method in enumerate(methods):
            main(
                mdp=mdp, gamma=0.9, method=method, seed=seed + 100000 * index,
                trial=trial, save_plots=False,
                n_deployment_episodes=n_deployment_episodes, training_episodes=training_episodes,
                eta=0.1, calibration_window=100, pilot_episodes=pilot_episodes, quantile_max = 3.0,
                R_star=R_star, shift_period=shift_period,
            )
