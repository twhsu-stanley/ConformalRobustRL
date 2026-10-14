"""Run Algorithm 2 or its nominal/global baselines, then save the observations."""

import pickle
import sys
from functools import partial
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformally_robust_controller import ConformallyRobustController
from robot_2d.online_settings import randomize_obstacles, sample_deviation, support_radius
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP


def main(
    mdp, n_episodes=30, gamma=0.95, horizon=100, training_episodes=4000, initial_R=0.1,
    target_failure=0.1, initial_failure=None, eta=0.01, calibration_window=100,
    pilot_episodes=20, calibration_scores=None, seed=7, save_plots=True,
    method="proposed", R=None, C=None, true_R=0.2, shift_period=100,
    verbose=True, trial=1,
):
    """R/C=None selects online estimation/calibration; numeric R/C keeps it fixed.

    method='nominal' plans with R=0; method='global' plans with C=infinity.
    horizon caps episode length; goal, collision, and exit end deployment earlier.
    The supplied obstacle map stays fixed throughout the run. Each episode has
    a fixed disturbance law whose support bound varies slowly within [0.5, 2.0],
    while true_R stays constant.
    These true simulator parameters are not supplied to the Q-learning agent.
    The trial loop generates one map and passes it to every method. Matching
    shift_period gives methods the same support schedule even when pilots are skipped.
    """
    if not np.allclose(mdp.cell_widths, 1.0, rtol=0, atol=1e-12):
        raise ValueError("This shifting-noise experiment requires unit grid cells.")
    if not np.isfinite(true_R) or not 0 < true_R < 1:
        raise ValueError("true_R must be in (0, 1) for the shifting-support experiment.")
    support_radius(0, shift_period)

    def prepare_episode(controller, episode):
        radius = support_radius(episode, shift_period)
        controller.deployment_env.noise_radius = radius
        controller.deployment_env.noise_sampler = partial(
            sample_deviation, radius=radius, true_R=true_R,
        )
        controller.deployment_env.is_nominal = False
        return {"true_support": radius}

    controller = ConformallyRobustController(
        mdp, gamma=gamma, horizon=horizon, training_horizon=horizon,
        training_episodes=training_episodes, initial_R=initial_R,
        target_failure=target_failure, initial_failure=initial_failure, eta=eta,
        calibration_scores=calibration_scores, calibration_window=calibration_window,
        pilot_episodes=pilot_episodes, seed=seed, method=method, R=R, C=C,
        episode_setup=prepare_episode,
    )
    controller.run(n_episodes, verbose=verbose)
    mdp = controller.deployment_env
    results = {
        "method": method, "seed": seed, "trial": trial, "mdp": mdp,
        "gamma": gamma, "horizon": horizon,
        "true_R": true_R, "shift_period": shift_period,
        "fixed_R": R, "fixed_C": C,
        "target_failure": target_failure, "initial_failure": initial_failure,
        "eta": eta, "calibration_window": calibration_window,
        "pilot_scores": controller.pilot_scores,
        "initial_calibration_scores": controller.initial_calibration_scores,
        "final_R": controller.R_estimate, "final_delta": controller.calibrator.delta,
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
        fig = plot_conformal_history(controller.history, controller.target_failure)
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
        start_cell=(0, 0), goal_cell=(9, 9), movement_reward=0.01, goal_reward=1.0,
    )

    seeds = (87,)
    methods = ("proposed", "nominal", "global")
    n_episodes = 100
    training_episodes = 4000
    pilot_episodes = 100
    true_R = 0.2
    shift_period = 100
    for trial, seed in enumerate(seeds, 1):
        mdp.set_obstacles([])
        randomize_obstacles(mdp, np.random.default_rng(seed))
        shared_scores = None
        for index, method in enumerate(methods):
            controller = main(
                mdp=mdp, method=method, seed=seed + 100000 * index, trial=trial, save_plots=False,
                n_episodes=n_episodes, training_episodes=training_episodes,
                eta=0.1, pilot_episodes=pilot_episodes, calibration_scores=shared_scores,
                true_R=true_R, shift_period=shift_period,
            )
            shared_scores = controller.initial_calibration_scores
