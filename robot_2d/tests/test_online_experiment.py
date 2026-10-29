"""Check continuous deployment, uncertainty updates, and online controller settings."""

from pathlib import Path
from functools import partial
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from conformally_robust_controller import ConformallyRobustController
from robot_2d.main_robot_2d_online import main as run_online
from robot_2d.utils import randomize_obstacles
from robot_2d.plot_online_comparison import mismatch_probabilities
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP


# These tests cover deployment bookkeeping; strict convergence is tested in test_tabular_agent.
main = partial(run_online, training_tolerance=1.0)


class ContinuousDeploymentTests(unittest.TestCase):
    def test_noise_is_continuous_and_positions_are_retained_before_discretization(self):
        sampler = lambda position, rng: np.array([0.2, 0.1])
        mdp = Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0, noise_sampler=sampler,
            max_episode_steps=2,
        )
        policy = np.zeros(mdp.n_state, dtype=int)
        row = mdp.deploy_policy(policy, start_position=(3.2, 2.7))
        np.testing.assert_allclose(row["positions"], [(3.2, 2.7), (4.7, 2.6), (5.7, 2.6)])
        np.testing.assert_allclose(row["nominal_positions"], [(4.5, 2.5), (5.5, 2.5)])
        np.testing.assert_allclose(row["deviations"], np.sqrt(0.05))
        np.testing.assert_array_equal(row["mismatches"], [0, 0])
        for position, state in zip(row["positions"], row["states"]):
            np.testing.assert_array_equal(mdp.discretize(position), state)
        self.assertIs(mdp.noise_sampler, sampler)

    def test_exit_records_the_terminal_disturbance_and_ends_the_episode(self):
        samples = []

        def sampler(position, rng):
            samples.append(position.copy())
            return np.array([-1.0, 0.0])

        mdp = Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0, noise_sampler=sampler,
            max_episode_steps=4,
        )
        row = mdp.deploy_policy(np.ones(mdp.n_state, dtype=int))
        np.testing.assert_allclose(row["positions"], [(0.5, 0.5), (-0.5, 0.5)])
        np.testing.assert_allclose(row["deviation_vectors"], [(-1, 0)])
        np.testing.assert_array_equal(row["states"][1:], [(-1, 0)])
        self.assertEqual(row["score"], 1.0)
        self.assertEqual(row["episode_mismatches"], 1)
        self.assertEqual(row["boundary_violations"], 1)
        self.assertEqual(row["workspace_exit_step"], 1)
        self.assertTrue(row["workspace_exit"])
        self.assertFalse(row["timeout"])
        self.assertEqual(len(samples), 1)
        self.assertEqual(row["episode_length"], 1)
        np.testing.assert_allclose(mdp.position, (-0.5, 0.5))

    def test_upper_boundary_landing_is_a_distinct_outside_cell(self):
        mdp = Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0,
            noise_sampler=lambda position, rng: np.array([0.5, 0.0]), max_episode_steps=1,
        )
        row = mdp.deploy_policy(lambda state: 0, start_position=(8.5, 0.5))
        np.testing.assert_array_equal(row["positions"], [(8.5, 0.5), (10, 0.5)])
        np.testing.assert_array_equal(row["deviation_vectors"], [(0.5, 0)])
        np.testing.assert_array_equal(row["states"][-1], (10, 0))
        self.assertEqual(row["episode_mismatches"], 1)
        self.assertEqual(row["score"], 0.5)
        self.assertTrue(row["workspace_exit"])
        self.assertFalse(row["timeout"])

    def test_obstacle_landing_is_recorded_as_the_last_transition(self):
        mdp = Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0, obstacles=[(1, 0)],
            noise_sampler=lambda position, rng: np.array([1.0, 0.0]), max_episode_steps=3,
        )
        seen = []

        def policy(state):
            seen.append(state)
            return 1

        row = mdp.deploy_policy(policy)
        self.assertEqual(seen, [mdp.start_state])
        np.testing.assert_allclose(row["positions"], [(0.5, 0.5), (1.0, 0.5)])
        self.assertTrue(row["collision"])
        self.assertEqual(row["episode_mismatches"], 1)
        self.assertEqual(row["episode_length"], mdp.elapsed_steps)

    def test_goal_arrival_retains_its_nonzero_disturbance_without_padding(self):
        samples = []

        def sampler(position, rng):
            samples.append(position.copy())
            return np.array([0.2, 0.0])

        mdp = Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0, noise_sampler=sampler,
            max_episode_steps=3,
        )
        row = mdp.deploy_policy(
            np.zeros(mdp.n_state, dtype=int), start_position=(8.2, 9.7),
        )
        np.testing.assert_allclose(row["nominal_positions"], [(9.5, 9.5)])
        np.testing.assert_allclose(row["positions"], [(8.2, 9.7), (9.0, 9.593333333333334)])
        np.testing.assert_allclose(row["deviation_vectors"], [(0.2, 0)])
        self.assertEqual(row["score"], 0.2)
        np.testing.assert_array_equal(row["mismatches"], [0])
        self.assertTrue(row["goal_reached"])
        self.assertEqual(row["goal_step"], 1)
        self.assertEqual(row["episode_length"], 1)
        self.assertAlmostEqual(row["deviations"][0], 0.2)
        self.assertEqual(len(samples), 1)

    def test_reference_is_uniform_including_goal_obstacles_and_boundaries(self):
        mdp = Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0, noise_probability=0.3,
            noise_radius=1.2,
        )
        probabilities = mismatch_probabilities(mdp)
        cases = [((2, 2), 0), ((0, 2), 1), ((0, 0), 1), ((7, 7), 0), ((2, 1), 0)]
        for cell, action in cases:
            self.assertAlmostEqual(probabilities[mdp.state_index(cell), action], 0.2336854404)
        self.assertEqual(probabilities.shape, (mdp.n_state, 4))

    def test_time_limit_counts_only_nonterminal_transitions(self):
        mdp = Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0,
            noise_sampler=lambda position, rng: np.array([0.2, 0.0]), max_episode_steps=3,
        )
        row = mdp.deploy_policy(np.ones(mdp.n_state, dtype=int))
        self.assertTrue(row["timeout"])
        self.assertEqual(row["episode_length"], 3)
        self.assertFalse(row["goal_reached"] or row["collision"] or row["workspace_exit"])
        np.testing.assert_array_equal(row["states"], [mdp.start_state] * 4)
        np.testing.assert_allclose(row["deviations"], 0.2)


class OnlineSettingsTests(unittest.TestCase):
    def test_baselines_skip_ACP_pilots_and_calibration_outputs(self):
        with (
            tempfile.TemporaryDirectory(prefix="robot_no_acp_") as directory,
            patch("robot_2d.main_robot_2d_online.__file__", str(Path(directory) / "main.py")),
            patch("conformally_robust_controller.ACP", side_effect=AssertionError("ACP called")),
            patch("robot_2d.utils.plot_conformal_history", side_effect=AssertionError("ACP plot")),
        ):
            output_dir = Path(directory) / "saved_results/online"
            for method in ("nominal", "R_contamination_0.1", "R_contamination_0.5"):
                with self.subTest(method=method):
                    controller = main(
                        Robot2DMotionMDP(
                            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0),
                            goal_cell=(9, 9), gamma=0.9, movement_reward=0.0, goal_reward=1.0,
                        ), method=method, n_deployment_episodes=2,
                        max_training_episodes=10, horizon=5, pilot_episodes=20,
                        calibration_scores=[], delta=2, eta=0, calibration_window=0,
                        save_plots=True, verbose=False,
                    )
                    controller.initialize_calibration()
                    controller.run_episode()
                    self.assertIsNone(controller.acp)
                    self.assertFalse(controller.pilot_scores)
                    self.assertFalse(controller.initial_calibration_scores)
                    name = f"{method}_R{controller.fixed_R}_C{controller.fixed_C}_trial_1"
                    with (output_dir / f"results_{name}.pkl").open("rb") as file:
                        saved = pickle.load(file)
                    for key in ("delta", "delta_1", "eta", "calibration_window", "final_delta",
                                "pilot_scores", "initial_calibration_scores", "final_R"):
                        self.assertNotIn(key, saved)
                    for row in saved["history"]:
                        for key in ("C_hat", "miscoverage", "delta_k", "delta_next",
                                    "R_hat_next", "transition_count", "mismatch_count"):
                            self.assertNotIn(key, row)
                    self.assertFalse((output_dir / f"online_uncertainty_{name}.png").exists())
                    self.assertTrue((output_dir / f"final_policy_{name}.png").exists())
            self.assertEqual(len(list(output_dir.iterdir())), 6)

    def test_trial_map_stays_fixed_across_methods_pilots_and_continuation(self):
        with (
            tempfile.TemporaryDirectory(prefix="robot_paired_maps_") as directory,
            patch("robot_2d.main_robot_2d_online.__file__", str(Path(directory) / "main.py")),
        ):
            mdp = Robot2DMotionMDP(
                x_max=10, y_max=10, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
                gamma=0.9, movement_reward=0.0, goal_reward=1.0,
            )
            randomize_obstacles(mdp, np.random.default_rng(87))
            trial_map = mdp.obstacle_mask.copy()
            observed_maps = []
            deploy_policy = Robot2DMotionMDP.deploy_policy

            def observe_map(env, *args, **kwargs):
                observed_maps.append(env.obstacle_mask.copy())
                return deploy_policy(env, *args, **kwargs)

            map_observer = patch.object(Robot2DMotionMDP, "deploy_policy", observe_map)
            self.addCleanup(map_observer.stop)
            map_observer.start()
            proposed = main(
                mdp=mdp, n_deployment_episodes=4, max_training_episodes=20,
                horizon=7, pilot_episodes=3,
                seed=87, shift_period=10, save_plots=False, verbose=False,
            )
            methods = ("nominal", "R_contamination_0.1", "R_contamination_0.5")
            for index, method in enumerate(methods, 1):
                other = main(
                    mdp=mdp, method=method, n_deployment_episodes=2, max_training_episodes=10,
                    horizon=7, pilot_episodes=0,
                    seed=87 + 100000 * index, shift_period=10,
                    save_plots=False, verbose=False,
                )
                other.run(2)
                for expected, actual in zip(proposed.history, other.history):
                    self.assertEqual(actual["C_star_k"], expected["C_star_k"])
                self.assertFalse(other.pilot_scores)
                filename = f"results_{method}_R{other.fixed_R}_C{other.fixed_C}_trial_1.pkl"
                with (Path(directory) / "saved_results/online" / filename).open("rb") as file:
                    saved = pickle.load(file)
                np.testing.assert_array_equal(saved["mdp"].obstacle_mask, trial_map)
                self.assertEqual(saved["seed"], 87 + 100000 * index)
            self.assertEqual(len(proposed.pilot_scores), 3)
            self.assertEqual(len(observed_maps), 19)
            for observed in observed_maps:
                np.testing.assert_array_equal(observed, trial_map)
            np.testing.assert_array_equal(mdp.obstacle_mask, trial_map)

    def test_runner_maps_methods_to_fixed_parameters_and_adaptive_updates(self):
        with (
            tempfile.TemporaryDirectory(prefix="robot_online_settings_") as directory,
            patch("robot_2d.main_robot_2d_online.__file__", str(Path(directory) / "main.py")),
        ):
            methods = {"proposed": None, "nominal": 0.0,
                       "R_contamination_0.1": 0.1, "R_contamination_0.5": 0.5}
            for method, expected_R in methods.items():
                with self.subTest(method=method):
                    controller = main(
                        mdp=Robot2DMotionMDP(
                            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0),
                            goal_cell=(9, 9), gamma=0.9, movement_reward=0.0, goal_reward=1.0,
                        ),
                        method=method, n_deployment_episodes=2, max_training_episodes=10, horizon=5,
                        calibration_scores=[0.1] * 20, R_hat_init=0.8,
                        save_plots=False, verbose=False,
                    )
                    first, last = controller.history
                    if method == "proposed":
                        count = first["episode_length"] + last["episode_length"]
                        self.assertEqual(last["transition_count"], count)
                        self.assertAlmostEqual(last["R_hat_next"], last["mismatch_count"] / count)
                        self.assertEqual(controller.acp.n_updates, 2)
                    else:
                        self.assertIsNone(controller.acp)
                        self.assertIsNone(controller.R_hat)
                        self.assertEqual(controller.transition_count, 0)
                        self.assertEqual(controller.mismatch_count, 0)
                        self.assertFalse(controller.initial_calibration_scores)
                        for row in controller.history:
                            self.assertNotIn("C_hat", row)
                            self.assertNotIn("miscoverage", row)
                            self.assertNotIn("delta_next", row)
                            self.assertNotIn("R_hat_next", row)
                            self.assertNotIn("transition_count", row)
                    self.assertTrue(all(r["training"]["zero_initialized"]
                                        for r in controller.history))
                    if method == "nominal":
                        self.assertEqual(last["R_used"], 0)
                        self.assertEqual(last["C_used"], 0)
                        self.assertTrue(last["training_reused"])
                    elif method == "proposed":
                        self.assertAlmostEqual(last["R_used"], first["R_hat_next"])
                        self.assertFalse(last["training_reused"])
                    else:
                        for row in controller.history:
                            self.assertEqual(row["R_used"], expected_R)
                            self.assertEqual(row["C_used"], np.inf)
                        self.assertEqual(controller.agent.R, expected_R)
                        self.assertEqual(controller.agent.C, np.inf)
                        self.assertTrue(last["training_reused"])

    def test_fixed_parameters_and_ACP_update_order(self):
        with (
            tempfile.TemporaryDirectory(prefix="robot_online_fixed_") as directory,
            patch("robot_2d.main_robot_2d_online.__file__", str(Path(directory) / "main.py")),
        ):
            fixed = main(
                mdp=Robot2DMotionMDP(
                    x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0),
                    goal_cell=(9, 9), gamma=0.9, movement_reward=0.0, goal_reward=1.0,
                ),
                n_deployment_episodes=2, max_training_episodes=10, horizon=5, R=0.2, C=0.7,
                calibration_scores=[0.1] * 20, save_plots=False, verbose=False,
            )
            self.assertTrue(all(row["R_used"] == 0.2 for row in fixed.history))
            self.assertTrue(all(row["C_used"] == 0.7 for row in fixed.history))
            self.assertIsNone(fixed.acp)
            self.assertIsNone(fixed.R_hat)
            self.assertTrue(fixed.history[-1]["training_reused"])
            adaptive = main(
                mdp=Robot2DMotionMDP(
                    x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0),
                    goal_cell=(9, 9), gamma=0.9, movement_reward=0.0, goal_reward=1.0,
                ),
                n_deployment_episodes=2, max_training_episodes=10, horizon=5,
                calibration_scores=[0.1] * 20,
                save_plots=False, verbose=False,
            )
            self.assertEqual(adaptive.history[0]["C_hat"], 0.1)
            coverage = 1 - np.mean([row["miscoverage"] for row in adaptive.history])
            change = adaptive.acp.delta_k - adaptive.acp.delta_1
            self.assertAlmostEqual(coverage, 0.9 + change / (adaptive.acp.eta * 2))

    def test_controller_continuation_preserves_experiment_settings(self):
        cases = {
            "nominal": dict(method="nominal"),
            "R_contamination_0.1": dict(method="R_contamination_0.1"),
            "R_contamination_0.5": dict(method="R_contamination_0.5"),
            "fixed": dict(R=0.2, C=0.7),
            "R_fixed": dict(R=0.2),
            "C_fixed": dict(C=0.7),
        }
        with (
            tempfile.TemporaryDirectory(prefix="robot_online_continue_") as directory,
            patch("robot_2d.main_robot_2d_online.__file__", str(Path(directory) / "main.py")),
        ):
            for name, settings in cases.items():
                with self.subTest(setting=name):
                    controller = main(
                        mdp=Robot2DMotionMDP(
                            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0),
                            goal_cell=(9, 9), gamma=0.9, movement_reward=0.0, goal_reward=1.0,
                        ),
                        n_deployment_episodes=1, max_training_episodes=10, horizon=5,
                        calibration_scores=[0.1] * 20, save_plots=False, verbose=False,
                        **settings,
                    )
                    previous_agent = controller.agent
                    previous_count = controller.transition_count
                    record = controller.run_episode()
                    self.assertEqual(record["episode"], 2)
                    if controller.fixed_R is not None and controller.fixed_C is not None:
                        self.assertIs(controller.agent, previous_agent)
                        self.assertTrue(record["training_reused"])
                    else:
                        self.assertIsNot(controller.agent, previous_agent)
                        self.assertFalse(record["training_reused"])
                    if controller.fixed_R is None:
                        self.assertEqual(record["transition_count"],
                                         previous_count + record["episode_length"])
                        self.assertEqual(record["R_used"], controller.history[0]["R_hat_next"])
                    else:
                        self.assertEqual(record["R_used"], controller.fixed_R)
                        self.assertEqual(controller.transition_count, 0)
                        self.assertNotIn("R_hat_next", record)
                    if controller.fixed_C is None:
                        self.assertEqual(controller.acp.n_updates, 2)
                        self.assertEqual(record["C_used"], record["C_hat"])
                    else:
                        self.assertIsNone(controller.acp)
                        self.assertFalse(controller.pilot_scores)
                        self.assertEqual(record["C_used"], controller.fixed_C)

    def test_mixed_adaptive_inputs_save_and_plot_only_enabled_components(self):
        with (
            tempfile.TemporaryDirectory(prefix="robot_mixed_updates_") as directory,
            patch("robot_2d.main_robot_2d_online.__file__", str(Path(directory) / "main.py")),
        ):
            output_dir = Path(directory) / "saved_results/online"
            for settings in (dict(R=0.3), dict(C=0.7)):
                with self.subTest(settings=settings):
                    controller = main(
                        Robot2DMotionMDP(
                            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0),
                            goal_cell=(9, 9), gamma=0.9, movement_reward=0.0, goal_reward=1.0,
                        ), n_deployment_episodes=2, max_training_episodes=10,
                        horizon=5, pilot_episodes=2, save_plots=True, verbose=False, **settings,
                    )
                    name = f"proposed_R{controller.fixed_R}_C{controller.fixed_C}_trial_1"
                    with (output_dir / f"results_{name}.pkl").open("rb") as file:
                        saved = pickle.load(file)
                    self.assertEqual("final_R" in saved, controller.fixed_R is None)
                    self.assertEqual("final_delta" in saved, controller.fixed_C is None)
                    self.assertEqual("pilot_scores" in saved, controller.fixed_C is None)
                    if controller.fixed_C is None:
                        self.assertEqual(len(controller.pilot_scores), 2)
                        self.assertEqual(controller.acp.n_updates, 2)
                        self.assertEqual(controller.transition_count, 0)
                    else:
                        self.assertFalse(controller.pilot_scores)
                        self.assertIsNone(controller.acp)
                    for row in saved["history"]:
                        self.assertEqual("R_hat_next" in row, controller.fixed_R is None)
                        self.assertEqual("C_hat" in row, controller.fixed_C is None)
                        self.assertFalse(row["training_reused"])
                    self.assertTrue((output_dir / f"online_uncertainty_{name}.png").exists())
                    self.assertTrue((output_dir / f"final_policy_{name}.png").exists())
            self.assertEqual(len(list(output_dir.iterdir())), 6)

    def test_episode_maps_noise_and_saved_truth_match_the_actual_rollouts(self):
        with (
            tempfile.TemporaryDirectory(prefix="robot_shifting_noise_") as directory,
            patch("robot_2d.main_robot_2d_online.__file__", str(Path(directory) / "main.py")),
        ):
            mdp = Robot2DMotionMDP(
                x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
                gamma=0.9, movement_reward=0.0, goal_reward=1.0,
            )
            randomize_obstacles(mdp, np.random.default_rng(7))
            controller = main(
                mdp=mdp, n_deployment_episodes=3, max_training_episodes=10, horizon=20,
                calibration_scores=[0.6] * 20, shift_period=10,
                save_plots=True, verbose=False,
            )
            for row in controller.history:
                self.assertNotIn("obstacles", row)
                self.assertLessEqual(row["score"], row["C_star_k"])
                np.testing.assert_array_equal(row["mismatches"], row["deviations"] > 0)
                nonterminal = slice(None) if row["timeout"] else slice(None, -1)
                np.testing.assert_allclose(
                    row["positions"][1:][nonterminal],
                    (row["nominal_positions"] + row["deviation_vectors"])[nonterminal],
                )
                self.assertEqual(row["score"], max(row["deviations"]))
            np.testing.assert_array_equal(
                controller.training_env.obstacle_mask, controller.deployment_env.obstacle_mask,
            )
            np.testing.assert_array_equal(
                mdp.obstacle_mask, controller.deployment_env.obstacle_mask,
            )
            self.assertTrue(controller.training_env.is_nominal)
            np.testing.assert_array_equal(controller.Q[controller.training_env.terminal_mask], 0)
            self.assertFalse(any(row["training_reused"] for row in controller.history))
            output_dir = Path(directory) / "saved_results/online"
            with (output_dir / "results_proposed_RNone_CNone_trial_1.pkl").open("rb") as file:
                saved = pickle.load(file)
            self.assertEqual(saved["R_star"], 0.2)
            self.assertEqual(saved["shift_period"], 10)
            np.testing.assert_array_equal(saved["mdp"].obstacle_mask, mdp.obstacle_mask)
            self.assertEqual(np.count_nonzero(saved["mdp"].obstacle_mask), 10)
            saved_mdp = saved["mdp"]
            deviation = saved_mdp._sample_deviation(saved_mdp.coordinates(saved_mdp.start_state))
            self.assertLessEqual(np.linalg.norm(deviation), saved_mdp.noise_radius)

    def test_first_step_exit_counts_for_R_and_updates_ACP(self):
        mdp = Robot2DMotionMDP(
            x_max=2, y_max=2, grid_shape=(2, 2), start_cell=(0, 0), goal_cell=(1, 1), gamma=0.9,
            movement_reward=0.0, goal_reward=1.0, noise_radius=10,
            noise_sampler=lambda position, rng: np.array([10.0, 0.0]),
        )
        controller = ConformallyRobustController(
            mdp, horizon=5, max_training_episodes=2000, calibration_scores=[0.1] * 20,
            R_hat_init=0, seed=7,
        )
        record = controller.run_episode()
        self.assertTrue(record["workspace_exit"])
        self.assertEqual(record["transition_count"], 1)
        self.assertEqual(record["episode_mismatches"], 1)
        self.assertEqual(record["R_hat_next"], 1)
        self.assertEqual(record["deviations"].tolist(), [10])
        self.assertEqual(record["score"], 10)
        self.assertEqual(record["miscoverage"], 1)
        self.assertEqual(controller.acp.n_updates, 1)
        self.assertAlmostEqual(controller.acp.delta_k, controller.acp.delta_1 - 0.009)
        self.assertEqual(controller.Q.shape, (mdp.n_state, 4))
        self.assertEqual(mdp.values_at(record["states"][-1], controller.Q.max(axis=1)), 0)


if __name__ == "__main__":
    unittest.main()
