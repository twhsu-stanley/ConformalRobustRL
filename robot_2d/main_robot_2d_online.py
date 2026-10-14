"""Run Algorithm 2 or its nominal/R-contamination baselines, then save the observations."""

import pickle
import sys
from functools import partial
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformally_robust_controller import ConformallyRobustController
from robot_2d.online_settings import randomize_obstacles, sample_deviation, C_star_shift
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP


def main(
    mdp, n_deployment_episodes=30, gamma=0.95, horizon=100, training_episodes=4000, R_hat_init=0.1,
    delta=0.1, delta_1=None, eta=0.01, calibration_window=100,
    pilot_episodes=20, calibration_scores=None, seed=7, save_plots=True,
    method="proposed", R=None, C=None, R_star=0.2, shift_period=100,
    verbose=True, trial=1,
):
    """R/C=None selects online estimation/calibration; numeric R/C keeps it fixed.

    method='nominal' plans with R=C=0. Methods 'R_contamination_0.1' and
    'R_contamination_0.3' fix R_used at 0.1 and 0.3, respectively, with C_used=infinity.
    horizon caps episode length; goal, collision, and exit end deployment earlier.
    The supplied obstacle map stays fixed throughout the run. Each episode has
    a fixed disturbance law whose support bound varies slowly within [0.5, 2.0],
    while R_star stays constant.
    These true simulator parameters are not supplied to the Q-learning agent.
    The trial loop generates one map and passes it to every method. Matching
    shift_period gives methods the same support schedule even when pilots are skipped.
    """
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
        delta=delta, delta_1=delta_1, eta=eta,
        calibration_scores=calibration_scores, calibration_window=calibration_window,
        pilot_episodes=pilot_episodes, seed=seed, method=method, R=R, C=C,
        episode_setup=prepare_episode,
    )
    controller.run(n_deployment_episodes, verbose=verbose)
    mdp = controller.deployment_env
    results = {
        "method": method, "seed": seed, "trial": trial, "mdp": mdp,
        "gamma": gamma, "horizon": horizon,
        "R_star": R_star, "shift_period": shift_period,
        "fixed_R": R, "fixed_C": C,
        "delta": delta, "delta_1": delta_1,
        "eta": eta, "calibration_window": calibration_window,
        "pilot_scores": controller.pilot_scores,
        "initial_calibration_scores": controller.initial_calibration_scores,
        "final_R": controller.R_hat, "final_delta": controller.calibrator.delta_k,
        "history": controller.history,
        "Q": controller.Q, "policy": controller.policy,
        "visit_counts": controller.agent.visit_counts,
    }
    output_dir = Path(__file__).resolve().parent / "saved_results/online"
    output_dir.mkdir(parents=True, exist_ok=True)
    test_name = f"{method}_R{R}_C{C}_trial_{trial}"
    filename = output_dir / f"results_{test_name}.pkl"
    with filename.open("wb") as file:
        pickle.dump(results, file)

    if save_plots:
        import matplotlib
        matplotlib.use("Agg")
        from robot_2d.utils import plot_conformal_history, plot_robot_motion
        import matplotlib.pyplot as plt
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
        obstacles=[],
        start_cell=(0, 0), goal_cell=(9, 9), movement_reward=0.001, goal_reward=1.0,
    )

    seeds = (67,)
    methods = ("proposed", "nominal")#, "R_contamination_0.1", "R_contamination_0.3")
    n_deployment_episodes = 40
    training_episodes = 5000
    pilot_episodes = 200
    R_star = 0.2
    shift_period = 100
    for trial, seed in enumerate(seeds, 1):
        mdp.set_obstacles([])
        randomize_obstacles(mdp, np.random.default_rng(seed), 20)
        shared_scores = None
        for index, method in enumerate(methods):
            controller = main(
                mdp=mdp, gamma=0.9, method=method, seed=seed + 100000 * index, trial=trial, save_plots=False,
                n_deployment_episodes=n_deployment_episodes, training_episodes=training_episodes,
                eta=0.1, calibration_window=200, pilot_episodes=pilot_episodes, calibration_scores=shared_scores,
                R_star=R_star, shift_period=shift_period,
            )
            shared_scores = controller.initial_calibration_scores
