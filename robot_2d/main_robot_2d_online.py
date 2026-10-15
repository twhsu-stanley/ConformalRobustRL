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


METHOD_SETTINGS = {
    "proposed": (None, None, "Proposed"),
    "nominal": (0.0, 0.0, "Nominal"),
    "R_contamination_0.1": (0.1, np.inf, "R-contamination (R=0.1)"),
    "R_contamination_0.3": (0.3, np.inf, "R-contamination (R=0.3)"),
}


def C_star_shift(episode, period=100):
    """Slowly cycle within [0.5, 2.0], staying strictly above the half-cell threshold."""
    if not isinstance(period, (int, np.integer)) or period < 2:
        raise ValueError("The distribution-shift period must be an integer of at least 2.")
    phase = (episode % period + 0.5) / period
    return float(0.5 + 1.5 * np.cos(np.pi * phase)**2)


def sample_deviation(nominal_position, rng, *, radius, R_star):
    """Mix zero with axis displacements that always leave the nominal unit cell.

    The law ignores nominal_position, so it is the same at every state and action.
    Every nonzero draw has norm strictly above 0.5 and at most radius. Its support
    reaches radius, and its mismatch probability is exactly R_star, even at exits.
    """
    if rng.random() >= R_star:
        return np.zeros(2)
    direction = int(rng.integers(4))
    magnitude = rng.uniform((0.5 + radius) / 2, radius)
    deviation = np.zeros(2)
    deviation[direction // 2] = magnitude if direction % 2 == 0 else -magnitude
    return deviation


def randomize_obstacles(mdp, rng, n_obstacles=10):
    """Place ten obstacles at least two cell indices from start and goal, with a path."""
    candidates = [
        (ix, iy) for iy in range(mdp.ny) for ix in range(mdp.nx)
        if all(max(abs(ix - sx), abs(iy - sy)) >= 2
               for sx, sy in (mdp.start_cell, mdp.goal_cell))
    ]
    if len(candidates) < n_obstacles:
        raise ValueError(f"The grid has too few eligible cells for {n_obstacles} obstacles.")
    previous = set(map(tuple, np.argwhere(mdp.obstacle_mask)[:, ::-1]))
    for _ in range(1000):
        indices = rng.choice(len(candidates), size=n_obstacles, replace=False)
        obstacles = [candidates[index] for index in indices]
        if set(obstacles) == previous:
            continue
        try:
            mdp.set_obstacles(obstacles)
        except ValueError:
            continue
        return obstacles
    raise ValueError(f"Could not sample a {n_obstacles}-obstacle map with a start-to-goal path.")


def main(
    mdp, n_deployment_episodes=30, gamma=0.95, horizon=100, training_episodes=4000, R_hat_init=0.1,
    delta=0.1, delta_1=None, eta=0.01, calibration_window=100, quantile_max = 3.0,
    pilot_episodes=20, calibration_scores=None, seed=7, save_plots=True,
    method="proposed", R=None, C=None, R_star=0.2, shift_period=100,
    verbose=True, trial=1,
):
    """R/C=None selects online estimation/calibration; numeric R/C keeps it fixed.

    method='nominal' plans with R=C=0. Methods 'R_contamination_0.1' and
    'R_contamination_0.3' fix R_used at 0.1 and 0.3, respectively, with C_used=infinity.
    R=None enables mismatch estimation; C=None enables ACP and calibration pilots.
    Fixed R/C disable the corresponding updates and allow policy reuse when both are fixed.
    horizon caps episode length; goal, collision, and exit end deployment earlier.
    The supplied obstacle map stays fixed throughout the run. Each episode has
    a fixed disturbance law whose support bound varies slowly within [0.5, 2.0],
    while R_star stays constant.
    These true simulator parameters are not supplied to the Q-learning agent.
    The trial loop generates one map and passes it to every method. Matching
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
        obstacles=[],
        start_cell=(1, 1), goal_cell=(8, 8), movement_reward=0.00001, goal_reward=1.0,
    )

    seeds = (23,)
    methods = ("proposed", "nominal")#, "R_contamination_0.1", "R_contamination_0.3")
    n_deployment_episodes = 80
    training_episodes = 5000
    pilot_episodes = 200
    R_star = 0.45
    shift_period = 60
    n_obstacles = 10

    for trial, seed in enumerate(seeds, 1):
        mdp.set_obstacles([])
        randomize_obstacles(mdp, np.random.default_rng(seed), n_obstacles)
        for index, method in enumerate(methods):
            main(
                mdp=mdp, gamma=0.9, method=method, seed=seed + 100000 * index,
                trial=trial, save_plots=False,
                n_deployment_episodes=n_deployment_episodes, training_episodes=training_episodes,
                eta=0.1, calibration_window=200, pilot_episodes=pilot_episodes, quantile_max = 3.0,
                R_star=R_star, shift_period=shift_period,
            )
