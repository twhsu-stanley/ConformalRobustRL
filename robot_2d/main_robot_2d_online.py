"""Run Algorithm 2 or its nominal/global baselines, then save the observations."""

import pickle
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformally_robust_controller import ConformallyRobustController
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP


def main(
    *, n_episodes=30, gamma=0.95, horizon=100, training_episodes=4000, initial_R=0.1,
    target_failure=0.1, initial_failure=None, eta=0.01, calibration_window=100,
    pilot_episodes=20, calibration_scores=None, seed=7, save_plots=True,
    method="proposed", R=None, C=None, noise_probability=0.3, noise_radius=1.2,
    verbose=True, trial=1,
):
    """R/C=None selects online estimation/calibration; numeric R/C keeps it fixed.

    method='nominal' plans with R=0; method='global' plans with C=infinity.
    horizon caps episode length; goal, collision, and exit end deployment earlier.
    The simulator noise law is not supplied to the learning agent.
    """
    mdp = Robot2DMotionMDP(
        x_max=8.0, y_max=8.0, grid_shape=(8, 8),
        obstacles=[(5, 1), (1, 3), (4, 3), (6, 4), (2, 5), (5, 6)],
        start_cell=(0, 0), goal_cell=(7, 7), movement_reward=0.01, goal_reward=1.0,
        noise_probability=noise_probability, noise_radius=noise_radius,
        max_episode_steps=horizon, seed=seed,
    )
    controller = ConformallyRobustController(
        mdp, gamma=gamma, horizon=horizon, training_horizon=horizon,
        training_episodes=training_episodes, initial_R=initial_R,
        target_failure=target_failure, initial_failure=initial_failure, eta=eta,
        calibration_scores=calibration_scores, calibration_window=calibration_window,
        pilot_episodes=pilot_episodes, seed=seed, method=method, R=R, C=C,
    )
    controller.run(n_episodes, verbose=verbose)
    results = {
        "method": method, "seed": seed, "trial": trial, "mdp": mdp,
        "gamma": gamma, "horizon": horizon,
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
    # Change these ordinary settings for the numerical experiment.
    seeds = (87,)
    methods = ("proposed", "nominal", "global")
    n_episodes = 30
    training_episodes = 4000
    pilot_episodes = 100
    for trial, seed in enumerate(seeds, 1):
        shared_scores = None
        for index, method in enumerate(methods):
            controller = main(
                method=method, seed=seed + 100000 * index, trial=trial, save_plots=False,
                n_episodes=n_episodes, training_episodes=training_episodes,
                pilot_episodes=pilot_episodes, calibration_scores=shared_scores,
            )
            shared_scores = controller.initial_calibration_scores
