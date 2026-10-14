"""Check continuous deployment, uncertainty updates, and online controller settings."""

from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from conformally_robust_controller import ConformallyRobustController
from robot_2d.main_robot_2d_online import main
from robot_2d.online_settings import randomize_obstacles
from robot_2d.plot_online_comparison import mismatch_probabilities
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP


class ContinuousDeploymentTests(unittest.TestCase):
    def test_noise_is_continuous_and_positions_are_retained_before_discretization(self):
        sampler = lambda position, rng: np.array([0.2, 0.1])
        mdp = Robot2DMotionMDP(
            noise_sampler=sampler, max_episode_steps=2,
        )
        policy = np.zeros(mdp.n_state, dtype=int)
        row = mdp.deploy_policy(policy, 0.95, start_position=(3.2, 2.7))
        np.testing.assert_allclose(row["positions"], [(3.2, 2.7), (4.7, 2.6), (5.7, 2.6)])
        np.testing.assert_allclose(row["nominal_positions"], [(4.5, 2.5), (5.5, 2.5)])
        np.testing.assert_allclose(row["deviations"], np.sqrt(0.05))
        np.testing.assert_array_equal(row["mismatches"], [0, 0])
        for position, state in zip(row["positions"], row["states"]):
            self.assertEqual(mdp.discretize(position), state)
        self.assertIs(mdp.noise_sampler, sampler)

    def test_exit_records_the_terminal_disturbance_and_ends_the_episode(self):
        samples = []

        def sampler(position, rng):
            samples.append(position.copy())
            return np.array([-1.0, 0.0])

        mdp = Robot2DMotionMDP(
            noise_sampler=sampler, max_episode_steps=4,
        )
        row = mdp.deploy_policy(np.ones(mdp.n_state, dtype=int), 0.95)
        np.testing.assert_allclose(row["positions"], [(0.5, 0.5), (-0.5, 0.5)])
        np.testing.assert_allclose(row["deviation_vectors"], [(-1, 0)])
        np.testing.assert_array_equal(row["states"][1:], mdp.exit_state)
        np.testing.assert_allclose(row["quantized_positions"], [(-0.5, 0.5)])
        self.assertAlmostEqual(row["score"], 1.0)
        self.assertEqual(row["episode_mismatches"], 1)
        self.assertEqual(row["boundary_violations"], 1)
        self.assertEqual(row["workspace_exit_step"], 1)
        self.assertTrue(row["workspace_exit"])
        self.assertFalse(row["timeout"])
        self.assertEqual(len(samples), 1)
        self.assertEqual(row["episode_length"], 1)
        np.testing.assert_allclose(mdp.position, (-0.5, 0.5))

    def test_upper_boundary_landing_retains_the_observation_inside_closed_workspace(self):
        mdp = Robot2DMotionMDP(
            noise_sampler=lambda position, rng: np.array([0.5, 0.0]), max_episode_steps=1,
        )
        row = mdp.deploy_policy(lambda state: 0, 0.95, start_position=(6.5, 0.5))
        np.testing.assert_array_equal(row["positions"], [(6.5, 0.5), (8, 0.5)])
        np.testing.assert_array_equal(row["quantized_positions"], [(7.5, 0.5)])
        np.testing.assert_array_equal(row["deviation_vectors"], [(0.5, 0)])
        self.assertEqual(row["states"][-1], mdp.cell_to_state((7, 0)))
        self.assertEqual(row["episode_mismatches"], 0)
        self.assertEqual(row["score"], 0.5)
        self.assertFalse(row["workspace_exit"])
        self.assertTrue(row["timeout"])

    def test_obstacle_landing_is_recorded_as_the_last_transition(self):
        mdp = Robot2DMotionMDP(
            obstacles=[(1, 0)], noise_sampler=lambda position, rng: np.array([1.0, 0.0]),
            max_episode_steps=3,
        )
        seen = []

        def policy(state):
            seen.append(state)
            return 1

        row = mdp.deploy_policy(policy, 0.95)
        self.assertEqual(seen, [mdp.start_state])
        np.testing.assert_allclose(row["positions"], [(0.5, 0.5), (1.5, 0.5)])
        self.assertTrue(row["collision"])
        self.assertEqual(row["episode_mismatches"], 1)
        self.assertEqual(row["episode_length"], mdp.elapsed_steps)

    def test_goal_arrival_retains_its_nonzero_disturbance_without_padding(self):
        samples = []

        def sampler(position, rng):
            samples.append(position.copy())
            return np.array([0.2, 0.0])

        mdp = Robot2DMotionMDP(
            noise_sampler=sampler, max_episode_steps=3,
        )
        row = mdp.deploy_policy(
            np.zeros(mdp.n_state, dtype=int), 0.95, start_position=(6.2, 7.7),
        )
        np.testing.assert_allclose(row["nominal_positions"], [(7.5, 7.5)])
        np.testing.assert_allclose(row["positions"], [(6.2, 7.7), (7.7, 7.5)])
        np.testing.assert_array_equal(row["mismatches"], [0])
        self.assertTrue(row["goal_reached"])
        self.assertEqual(row["goal_step"], 1)
        self.assertEqual(row["episode_length"], 1)
        self.assertAlmostEqual(row["score"], 0.2)
        self.assertEqual(len(samples), 1)

    def test_reference_is_uniform_including_goal_obstacles_and_boundaries(self):
        mdp = Robot2DMotionMDP(noise_probability=0.3, noise_radius=1.2)
        probabilities = mismatch_probabilities(mdp)
        cases = [((2, 2), 0), ((0, 2), 1), ((0, 0), 1), ((7, 7), 0), ((2, 1), 0)]
        for cell, action in cases:
            self.assertAlmostEqual(probabilities[mdp.cell_to_state(cell), action], 0.2336854404)
        np.testing.assert_array_equal(probabilities[mdp.exit_state], 0)

    def test_time_limit_counts_only_nonterminal_transitions(self):
        mdp = Robot2DMotionMDP(
            noise_sampler=lambda position, rng: np.array([0.2, 0.0]), max_episode_steps=3,
        )
        row = mdp.deploy_policy(np.ones(mdp.n_state, dtype=int), 0.95)
        self.assertTrue(row["timeout"])
        self.assertEqual(row["episode_length"], 3)
        self.assertFalse(row["goal_reached"] or row["collision"] or row["workspace_exit"])
        np.testing.assert_array_equal(row["states"], mdp.start_state)
        np.testing.assert_allclose(row["deviations"], 0.2)


class OnlineSettingsTests(unittest.TestCase):
    def test_trial_map_stays_fixed_across_methods_pilots_and_continuation(self):
        with (
            tempfile.TemporaryDirectory(prefix="robot_paired_maps_") as directory,
            patch("robot_2d.main_robot_2d_online.__file__", str(Path(directory) / "main.py")),
        ):
            mdp = Robot2DMotionMDP(x_max=10, y_max=10, grid_shape=(10, 10))
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
                mdp=mdp, n_deployment_episodes=4, training_episodes=20, horizon=7, pilot_episodes=3,
                seed=87, shift_period=10, save_plots=False, verbose=False,
            )
            for index, method in enumerate(("nominal", "global"), 1):
                other = main(
                    mdp=mdp, method=method, n_deployment_episodes=2, training_episodes=10,
                    horizon=7,
                    calibration_scores=proposed.initial_calibration_scores, pilot_episodes=0,
                    seed=87 + 100000 * index, shift_period=10,
                    save_plots=False, verbose=False,
                )
                other.run(2)
                for expected, actual in zip(proposed.history, other.history):
                    self.assertEqual(actual["C_star_k"], expected["C_star_k"])
                self.assertFalse(other.pilot_scores)
                filename = f"results_{method}_RNone_CNone_trial_1.pkl"
                with (Path(directory) / "saved_results/online" / filename).open("rb") as file:
                    saved = pickle.load(file)
                np.testing.assert_array_equal(saved["mdp"].obstacle_mask, trial_map)
                self.assertEqual(saved["seed"], 87 + 100000 * index)
            self.assertEqual(len(proposed.pilot_scores), 3)
            self.assertEqual(len(observed_maps), 15)
            for observed in observed_maps:
                np.testing.assert_array_equal(observed, trial_map)
            np.testing.assert_array_equal(mdp.obstacle_mask, trial_map)

    def test_baseline_settings_preserve_estimator_counts(self):
        with (
            tempfile.TemporaryDirectory(prefix="robot_online_settings_") as directory,
            patch("robot_2d.main_robot_2d_online.__file__", str(Path(directory) / "main.py")),
        ):
            for method in ("proposed", "nominal", "global"):
                with self.subTest(method=method):
                    controller = main(
                        mdp=Robot2DMotionMDP(),
                        method=method, n_deployment_episodes=2, training_episodes=10, horizon=5,
                        calibration_scores=[0.1] * 20,
                        save_plots=False, verbose=False,
                    )
                    first, last = controller.history
                    count = first["episode_length"] + last["episode_length"]
                    self.assertEqual(last["transition_count"], count)
                    self.assertAlmostEqual(last["R_hat_next"], last["mismatch_count"] / count)
                    self.assertEqual(controller.calibrator.n_updates, 2)
                    self.assertTrue(all(r["training"]["zero_initialized"]
                                        for r in controller.history))
                    if method == "nominal":
                        self.assertEqual(last["R_used"], 0)
                        self.assertTrue(last["training_reused"])
                    else:
                        self.assertAlmostEqual(last["R_used"], first["R_hat_next"])
                        self.assertFalse(last["training_reused"])
                    if method == "global":
                        self.assertTrue(np.isposinf(last["C_used"]))
                        self.assertEqual(controller.agent.C, np.inf)

    def test_fixed_parameters_and_ACP_update_order(self):
        with (
            tempfile.TemporaryDirectory(prefix="robot_online_fixed_") as directory,
            patch("robot_2d.main_robot_2d_online.__file__", str(Path(directory) / "main.py")),
        ):
            fixed = main(
                mdp=Robot2DMotionMDP(),
                n_deployment_episodes=2, training_episodes=10, horizon=5, R=0.2, C=0.7,
                calibration_scores=[0.1] * 20, save_plots=False, verbose=False,
            )
            self.assertTrue(all(row["R_used"] == 0.2 for row in fixed.history))
            self.assertTrue(all(row["C_used"] == 0.7 for row in fixed.history))
            self.assertEqual(fixed.calibrator.delta_k, 0.1)
            adaptive = main(
                mdp=Robot2DMotionMDP(),
                n_deployment_episodes=2, training_episodes=10, horizon=5,
                calibration_scores=[0.1] * 20,
                save_plots=False, verbose=False,
            )
            self.assertEqual(adaptive.history[0]["C_hat"], 0.1)
            coverage = 1 - np.mean([row["miscoverage"] for row in adaptive.history])
            change = adaptive.calibrator.delta_k - adaptive.calibrator.delta_1
            self.assertAlmostEqual(coverage, 0.9 + change / (adaptive.calibrator.eta * 2))

    def test_controller_continuation_preserves_experiment_settings(self):
        cases = {
            "nominal": dict(method="nominal"), "global": dict(method="global"),
            "fixed": dict(R=0.2, C=0.7),
        }
        with (
            tempfile.TemporaryDirectory(prefix="robot_online_continue_") as directory,
            patch("robot_2d.main_robot_2d_online.__file__", str(Path(directory) / "main.py")),
        ):
            for name, settings in cases.items():
                with self.subTest(setting=name):
                    controller = main(
                        mdp=Robot2DMotionMDP(),
                        n_deployment_episodes=1, training_episodes=10, horizon=5,
                        calibration_scores=[0.1] * 20, save_plots=False, verbose=False,
                        **settings,
                    )
                    previous_agent = controller.agent
                    previous_count = controller.transition_count
                    record = controller.run_episode()
                    self.assertEqual(record["episode"], 2)
                    self.assertEqual(record["transition_count"],
                                     previous_count + record["episode_length"])
                    if name == "nominal":
                        self.assertIs(controller.agent, previous_agent)
                        self.assertEqual(record["R_used"], 0)
                        self.assertEqual(record["C_used"], 0)
                    else:
                        self.assertIsNot(controller.agent, previous_agent)
                        expected_C = np.inf if name == "global" else 0.7
                        self.assertEqual(record["C_used"], expected_C)
                    if name == "fixed":
                        self.assertEqual(record["R_used"], 0.2)
                        self.assertEqual(controller.calibrator.n_updates, 0)
                    else:
                        self.assertEqual(controller.calibrator.n_updates, 2)

    def test_episode_maps_noise_and_saved_truth_match_the_actual_rollouts(self):
        with (
            tempfile.TemporaryDirectory(prefix="robot_shifting_noise_") as directory,
            patch("robot_2d.main_robot_2d_online.__file__", str(Path(directory) / "main.py")),
        ):
            mdp = Robot2DMotionMDP()
            randomize_obstacles(mdp, np.random.default_rng(7))
            controller = main(
                mdp=mdp, n_deployment_episodes=3, training_episodes=10, horizon=20,
                calibration_scores=[0.6] * 20, shift_period=10,
                save_plots=True, verbose=False,
            )
            for row in controller.history:
                self.assertNotIn("obstacles", row)
                self.assertLessEqual(row["score"], row["C_star_k"])
                np.testing.assert_array_equal(row["mismatches"], row["deviations"] > 0)
                np.testing.assert_allclose(
                    row["positions"][1:], row["nominal_positions"] + row["deviation_vectors"],
                )
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

    def test_exit_counts_one_transition_and_calibrates_the_raw_disturbance(self):
        mdp = Robot2DMotionMDP(
            x_max=2, y_max=2, grid_shape=(2, 2), noise_radius=10,
            noise_sampler=lambda position, rng: np.array([10.0, 0.0]),
        )
        controller = ConformallyRobustController(
            mdp, horizon=5, training_episodes=100, calibration_scores=[0.1] * 20,
            R_hat_init=0, seed=7,
        )
        record = controller.run_episode()
        self.assertTrue(record["workspace_exit"])
        self.assertEqual(record["transition_count"], 1)
        self.assertEqual(record["episode_mismatches"], 1)
        self.assertEqual(record["R_hat_next"], 1)
        self.assertEqual(record["score"], 10)
        self.assertEqual(record["miscoverage"], 1)
        np.testing.assert_array_equal(controller.Q[mdp.exit_state], 0)


if __name__ == "__main__":
    unittest.main()
