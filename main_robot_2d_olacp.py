"""Run Algorithm 2 with fresh robust Q-learning for every deployment episode."""

import pickle
from pathlib import Path

import numpy as np

from conformally_robust_controller import ConformallyRobustController
from robot_2d_motion_mdp import Robot2DMotionMDP


def main(
    *, n_episodes=30, gamma=0.95, horizon=100, training_episodes=2000, initial_R=0.1,
    target_failure=0.1, initial_failure=None, eta=0.01, calibration_window=100,
    pilot_episodes=20, calibration_scores=None, seed=7, save_plots=True,
    output_dir="results/conformal_robot_motion",
):
    if not isinstance(n_episodes, (int, np.integer)) or n_episodes < 1:
        raise ValueError("n_episodes must be a positive integer.")
    mdp = Robot2DMotionMDP(
        x_max=8.0, y_max=8.0, grid_shape=(8, 8),
        obstacles=[(2, 1), (5, 1), (1, 3), (4, 3), (6, 4), (2, 5), (5, 6)],
        start_cell=(0, 0), goal_cell=(7, 7), movement_reward=0.01, goal_reward=1.0,
        noise_probability=0.15, noise_radius=1.2, noise_sampler=None,
        max_episode_steps=horizon, episode_mode="fixed_horizon", seed=seed,
    )
    controller = ConformallyRobustController(
        mdp, gamma=gamma, horizon=horizon, training_horizon=horizon,
        training_episodes=training_episodes, initial_R=initial_R,
        target_failure=target_failure, initial_failure=initial_failure, eta=eta,
        calibration_scores=calibration_scores, calibration_window=calibration_window,
        pilot_episodes=pilot_episodes, seed=seed,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(n_episodes):
        record = controller.run_episode()
        print(
            f"Episode {record['episode']}: R={record['R_used']:.3f} -> {record['R_next']:.3f}, "
            f"C={record['planning_radius']:.3f}, score={record['score']:.3f}, "
            f"miscoverage={record['miscoverage']}, goal={record['goal_reached']}",
        )
    results = {
        "pilot_scores": controller.pilot_scores,
        "initial_calibration_scores": controller.initial_calibration_scores,
        "final_R": controller.R_estimate, "final_delta": controller.calibrator.delta,
        "history": controller.history,
    }
    with (output_dir / "results.pkl").open("wb") as file:
        pickle.dump(results, file)
    np.savez_compressed(
        output_dir / "final_policy.npz", Q=controller.Q, policy=controller.policy,
        visit_counts=controller.agent.visit_counts,
    )
    if save_plots:
        import matplotlib
        matplotlib.use("Agg")
        from utils_tabular import plot_conformal_history, plot_robot_motion
        import matplotlib.pyplot as plt
        fig = plot_conformal_history(controller.history, controller.target_failure)
        fig.savefig(output_dir / "online_uncertainty.png", dpi=160)
        plt.close(fig)
        fig = plot_robot_motion(
            mdp, controller.Q, controller.policy, controller.history[-1]["positions"],
            "Final conformally robust policy",
        )
        fig.savefig(output_dir / "final_policy.png", dpi=160)
        plt.close(fig)
    print(f"Saved results to {output_dir.resolve()}")
    return controller


if __name__ == "__main__":
    main()
