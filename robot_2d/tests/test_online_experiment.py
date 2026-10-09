"""Check continuous deployment, uncertainty updates, and online controller settings."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from conformally_robust_controller import ConformallyRobustController
from robot_2d.main_robot_2d_online import evaluate_policy, main
from robot_2d.plot_online_comparison import mismatch_probabilities
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP


class ContinuousDeploymentTests(unittest.TestCase):
    def test_noise_is_continuous_and_positions_are_retained_before_discretization(self):
        sampler = lambda position, rng: np.array([0.2, 0.1])
        mdp = Robot2DMotionMDP(
            noise_sampler=sampler, max_episode_steps=2, episode_mode="fixed_horizon",
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

    def test_exit_retains_sampled_deviation_and_completes_the_fixed_horizon(self):
        samples = []

        def sampler(position, rng):
            samples.append(position.copy())
            return np.array([-1.0, 0.0])

        mdp = Robot2DMotionMDP(
            noise_sampler=sampler, max_episode_steps=4, episode_mode="fixed_horizon",
        )
        row = mdp.deploy_policy(np.ones(mdp.n_state, dtype=int), 0.95)
        np.testing.assert_allclose(row["positions"][1:], [(-0.5, 0.5)] * 4)
        np.testing.assert_allclose(row["deviation_vectors"], [(-1, 0)] * 4)
        np.testing.assert_array_equal(row["states"][1:], mdp.failure_state)
        np.testing.assert_array_equal(row["source_states"], mdp.start_state)
        np.testing.assert_allclose(row["source_positions"], [(0.5, 0.5)] * 4)
        np.testing.assert_allclose(row["quantized_positions"], [(-0.5, 0.5)] * 4)
        self.assertAlmostEqual(row["score"], 1.0)
        self.assertEqual(row["episode_mismatches"], 4)
        self.assertEqual(row["boundary_violations"], 4)
        self.assertEqual(row["workspace_exit_count"], 4)
        self.assertEqual(row["workspace_exit_step"], 1)
        self.assertTrue(row["workspace_exit"])
        self.assertTrue(row["timeout"])
        self.assertEqual(len(samples), 4)
        self.assertEqual(len(row["actions"]), 4)
        np.testing.assert_allclose(mdp.position, (0.5, 0.5))

    def test_obstacle_landing_is_recorded_before_an_uncounted_restart(self):
        mdp = Robot2DMotionMDP(
            obstacles=[(1, 0)], noise_sampler=lambda position, rng: np.array([1.0, 0.0]),
            max_episode_steps=3, episode_mode="fixed_horizon",
        )
        seen = []

        def policy(state):
            seen.append(state)
            return 1

        row = mdp.deploy_policy(policy, 0.95)
        self.assertEqual(seen, [mdp.start_state] * 3)
        np.testing.assert_allclose(row["positions"][1:], [(1.5, 0.5)] * 3)
        np.testing.assert_allclose(row["source_positions"], [(0.5, 0.5)] * 3)
        self.assertEqual(row["collision_count"], 3)
        self.assertEqual(row["episode_mismatches"], 3)
        self.assertEqual(len(row["actions"]), mdp.elapsed_steps)

    def test_goal_samples_noise_and_exit_restarts_at_the_pre_action_position(self):
        deviations = iter(([0.6, 0.0], [0.2, 0.0], [-1.0, 0.0]))
        mdp = Robot2DMotionMDP(
            noise_sampler=lambda position, rng: np.array(next(deviations)),
            max_episode_steps=3, episode_mode="fixed_horizon",
        )
        row = mdp.deploy_policy(
            np.zeros(mdp.n_state, dtype=int), 0.95, start_position=mdp.coordinates(mdp.goal_state),
        )
        np.testing.assert_array_equal(row["source_states"], mdp.goal_state)
        np.testing.assert_allclose(row["nominal_positions"], [(7.5, 7.5)] * 3)
        np.testing.assert_allclose(row["positions"][1:], [(8.1, 7.5), (7.7, 7.5), (6.5, 7.5)])
        np.testing.assert_allclose(row["source_positions"], [(7.5, 7.5), (7.5, 7.5), (7.7, 7.5)])
        np.testing.assert_array_equal(row["mismatches"], [1, 0, 1])
        np.testing.assert_allclose(row["deviations"], [0.6, 0.2, 1.0])

    def test_reference_is_uniform_including_goal_obstacles_and_boundaries(self):
        mdp = Robot2DMotionMDP(noise_probability=0.3, noise_radius=1.2)
        probabilities = mismatch_probabilities(mdp)
        cases = [((2, 2), 0), ((0, 2), 1), ((0, 0), 1), ((7, 7), 0), ((2, 1), 0)]
        for cell, action in cases:
            self.assertAlmostEqual(probabilities[mdp.cell_to_state(cell), action], 0.2336854404)
        np.testing.assert_array_equal(probabilities[mdp.failure_state], 0)

    def test_workspace_exit_is_reported_separately_in_evaluation(self):
        mdp = Robot2DMotionMDP(
            noise_sampler=lambda position, rng: np.array([-1.0, 0.0]),
            max_episode_steps=4, episode_mode="fixed_horizon",
        )
        result = evaluate_policy(mdp, np.ones(mdp.n_state, dtype=int), 0.95, 5,
                                 np.random.default_rng(7))
        self.assertEqual(result["workspace_exit_rate"], 1)
        self.assertEqual(result["success_rate"], 0)
        self.assertEqual(result["collision_rate"], 0)
        self.assertEqual(result["timeout_rate"], 1)

    def test_goal_and_exit_rates_can_overlap_after_a_restart(self):
        deviations = iter(([-2.0, 0.0], [0.0, 0.0], [0.0, 0.0]))
        mdp = Robot2DMotionMDP(
            x_max=2, y_max=2, grid_shape=(2, 2), noise_radius=2,
            noise_sampler=lambda position, rng: np.array(next(deviations)),
            max_episode_steps=3, episode_mode="fixed_horizon",
        )
        policy = np.array([0, 2, 0, 0, 0])
        result = evaluate_policy(mdp, policy, 0.95, 1, np.random.default_rng(7))
        self.assertEqual(result["success_rate"], 1)
        self.assertEqual(result["workspace_exit_rate"], 1)
        self.assertEqual(result["timeout_rate"], 0)


class OnlineSettingsTests(unittest.TestCase):
    def test_baseline_settings_and_evaluation_do_not_change_estimator_counts(self):
        with tempfile.TemporaryDirectory(prefix="robot_online_settings_") as directory:
            for method in ("proposed", "nominal", "global"):
                with self.subTest(method=method):
                    controller = main(
                        method=method, n_episodes=2, training_episodes=10, horizon=5,
                        calibration_scores=[0.1] * 20, evaluation_episodes=8,
                        output_dir=Path(directory) / method, save_plots=False, verbose=False,
                    )
                    first, last = controller.history
                    self.assertEqual(last["transition_count"], 10)
                    self.assertAlmostEqual(last["R_next"], last["mismatch_count"] / 10)
                    self.assertEqual(controller.calibrator.n_updates, 2)
                    self.assertTrue(all(r["training"]["zero_initialized"]
                                        for r in controller.history))
                    if method == "nominal":
                        self.assertEqual(last["R_used"], 0)
                        self.assertTrue(last["training_reused"])
                    else:
                        self.assertAlmostEqual(last["R_used"], first["R_next"])
                        self.assertFalse(last["training_reused"])
                    if method == "global":
                        self.assertTrue(np.isposinf(last["planning_radius"]))
                        self.assertEqual(controller.agent.C, np.inf)
                    evaluation = controller.results["evaluation"]
                    self.assertAlmostEqual(
                        evaluation["success_rate"] + evaluation["timeout_rate"], 1,
                    )

    def test_fixed_parameters_and_ACP_update_order(self):
        with tempfile.TemporaryDirectory(prefix="robot_online_fixed_") as directory:
            fixed = main(
                n_episodes=2, training_episodes=10, horizon=5, R=0.2, C=0.7,
                calibration_scores=[0.1] * 20, save_plots=False, verbose=False,
                output_dir=Path(directory) / "fixed",
            )
            self.assertTrue(all(row["R_used"] == 0.2 for row in fixed.history))
            self.assertTrue(all(row["planning_radius"] == 0.7 for row in fixed.history))
            self.assertEqual(fixed.calibrator.delta, 0.1)
            adaptive = main(
                n_episodes=2, training_episodes=10, horizon=5, calibration_scores=[0.1] * 20,
                save_plots=False, verbose=False, output_dir=Path(directory) / "adaptive",
            )
            self.assertEqual(adaptive.history[0]["conformal_radius"], 0.1)
            coverage = 1 - np.mean([row["miscoverage"] for row in adaptive.history])
            change = adaptive.calibrator.delta - adaptive.calibrator.initial_delta
            self.assertAlmostEqual(coverage, 0.9 + change / (adaptive.calibrator.eta * 2))

    def test_controller_continuation_preserves_experiment_settings(self):
        cases = {
            "nominal": dict(method="nominal"), "global": dict(method="global"),
            "fixed": dict(R=0.2, C=0.7),
        }
        with tempfile.TemporaryDirectory(prefix="robot_online_continue_") as directory:
            for name, settings in cases.items():
                with self.subTest(setting=name):
                    controller = main(
                        n_episodes=1, training_episodes=10, horizon=5,
                        calibration_scores=[0.1] * 20, save_plots=False, verbose=False,
                        output_dir=Path(directory) / name, **settings,
                    )
                    previous_agent = controller.agent
                    record = controller.run_episode()
                    self.assertEqual(record["episode"], 2)
                    self.assertEqual(record["transition_count"], 10)
                    if name == "nominal":
                        self.assertIs(controller.agent, previous_agent)
                        self.assertEqual(record["R_used"], 0)
                        self.assertEqual(record["planning_radius"], 0)
                    else:
                        self.assertIsNot(controller.agent, previous_agent)
                        expected_C = np.inf if name == "global" else 0.7
                        self.assertEqual(record["planning_radius"], expected_C)
                    if name == "fixed":
                        self.assertEqual(record["R_used"], 0.2)
                        self.assertEqual(controller.calibrator.n_updates, 0)
                    else:
                        self.assertEqual(controller.calibrator.n_updates, 2)

    def test_exit_counts_all_H_transitions_and_calibrates_the_raw_disturbance(self):
        mdp = Robot2DMotionMDP(
            x_max=2, y_max=2, grid_shape=(2, 2), noise_radius=10,
            noise_sampler=lambda position, rng: np.array([10.0, 0.0]),
        )
        controller = ConformallyRobustController(
            mdp, horizon=5, training_episodes=100, calibration_scores=[0.1] * 20,
            initial_R=0, seed=7,
        )
        record = controller.run_episode()
        self.assertTrue(record["workspace_exit"])
        self.assertEqual(record["transition_count"], 5)
        self.assertEqual(record["episode_mismatches"], 5)
        self.assertEqual(record["R_next"], 1)
        self.assertEqual(record["score"], 10)
        self.assertEqual(record["miscoverage"], 1)
        np.testing.assert_array_equal(controller.Q[mdp.failure_state], 0)


if __name__ == "__main__":
    unittest.main()
