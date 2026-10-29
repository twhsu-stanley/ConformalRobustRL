"""General ACP bookkeeping and conformally robust controller lifecycle checks."""

import unittest
from unittest.mock import patch

import numpy as np

from acp import ACP
from conformally_robust_controller import ConformallyRobustController
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP


class ACPTests(unittest.TestCase):
    def test_order_statistic_uses_N_plus_one_without_interpolation(self):
        calibration = ACP([0, 1, 2, 3], delta=0.5)
        self.assertEqual(calibration.quantile(), 2)

    def test_quantile_precedes_append_and_update_is_unclipped(self):
        calibration = ACP([0, 1, 2, 3], delta=0.5, eta=0.1)
        prediction = calibration.quantile()
        update = calibration.update(9, prediction)
        self.assertEqual(update["miscoverage"], 1)
        self.assertAlmostEqual(calibration.delta_k, 0.45)
        self.assertEqual(calibration.scores[-1], 9)

    def test_quantile_tails_return_nonnegative_planning_radii(self):
        calibration = ACP([0], delta=0.1, eta=0.2)
        self.assertEqual(calibration.quantile(), np.inf)
        calibration.delta_k = 1.1
        self.assertEqual(calibration.quantile(), 0)
        self.assertEqual(calibration.update(0)["miscoverage"], 0)
        self.assertAlmostEqual(calibration.delta_k, 1.12)
        self.assertEqual(calibration.update(0.1)["miscoverage"], 1)
        self.assertAlmostEqual(calibration.delta_k, 0.94)

    def test_target_initial_and_adaptive_delta_are_distinct(self):
        calibration = ACP([0.1] * 10, delta=0.1, delta_1=0.4, eta=0.2)
        update = calibration.update(0.2)
        self.assertEqual(calibration.delta, 0.1)
        self.assertEqual(calibration.delta_1, 0.4)
        self.assertEqual(update["delta_k"], 0.4)
        self.assertAlmostEqual(update["delta_next"], 0.22)
        self.assertAlmostEqual(calibration.delta_k, 0.22)

    def test_deque_window_discards_old_scores(self):
        calibration = ACP([0, 1, 2], window_size=2)
        self.assertEqual(list(calibration.scores), [1, 2])
        calibration.update(3)
        self.assertEqual(list(calibration.scores), [2, 3])

    def test_telescoping_ACP_identity(self):
        calibration = ACP([0.2] * 10, delta=0.2, eta=0.1)
        misses = 0
        for score in (0.1, 0.8, 0.3, 0, 0.9, 0.2):
            misses += calibration.update(score)["miscoverage"]
        expected = calibration.delta_1 + calibration.eta * (6 * 0.2 - misses)
        self.assertAlmostEqual(calibration.delta_k, expected)


class ControllerTests(unittest.TestCase):
    def controller(self, **kwargs):
        mdp = Robot2DMotionMDP(
            x_max=2, y_max=2, grid_shape=(2, 2), movement_reward=0,
            noise_sampler=lambda position, rng: np.array((0.2, 0)),
        )
        options = dict(
            horizon=4, training_episodes=2000, training_horizon=10, R_hat_init=0.2,
            calibration_scores=[0.1] * 10, delta=0.2, eta=0.1, seed=5,
        )
        options.update(kwargs)
        return ConformallyRobustController(mdp, **options)

    def test_episode_statistics_use_real_deployment_observations(self):
        controller = self.controller()
        record = controller.run_episode()
        self.assertEqual(controller.transition_count, record["episode_length"])
        self.assertEqual(controller.mismatch_count, np.sum(record["mismatches"]))
        self.assertAlmostEqual(record["score"], max(record["deviations"]))
        self.assertAlmostEqual(record["R_hat_next"], np.mean(record["mismatches"]))
        self.assertAlmostEqual(record["C_hat"], 0.1)
        self.assertEqual(record["miscoverage"], int(record["score"] > 0.1))
        self.assertAlmostEqual(record["delta_next"], 0.2 + 0.1 * (0.2 - record["miscoverage"]))
        self.assertEqual(record["positions"].shape, (record["episode_length"] + 1, 2))
        self.assertLessEqual(record["episode_length"], controller.horizon)
        self.assertTrue(record["training"]["zero_initialized"])
        self.assertIsNot(controller.training_env, controller.deployment_env)
        self.assertTrue(controller.training_env.is_nominal)

    def test_each_episode_uses_fresh_agent_and_cumulative_R(self):
        controller = self.controller()
        first = controller.run_episode()
        first_agent = controller.agent
        first_agent.Q.fill(999)
        second = controller.run_episode()
        self.assertIsNot(first_agent, controller.agent)
        self.assertEqual(controller.agent.evaluation_return[0], 0)
        self.assertAlmostEqual(second["R_used"], first["R_hat_next"])
        count = first["episode_length"] + second["episode_length"]
        self.assertEqual(second["transition_count"], count)
        self.assertEqual(second["mismatch_count"],
                         first["episode_mismatches"] + second["episode_mismatches"])
        self.assertAlmostEqual(second["R_hat_next"], second["mismatch_count"] / count)

    def test_variable_episodes_weight_R_by_transitions_and_score_every_episode(self):
        deviations = iter(([0.2, 0], [0.1, 0], [-2.0, 0], [0, 0], [0, 0.6]))
        mdp = Robot2DMotionMDP(
            x_max=2, y_max=2, grid_shape=(2, 2), noise_radius=2,
            noise_sampler=lambda position, rng: np.array(next(deviations)),
        )
        controller = ConformallyRobustController(
            mdp, horizon=6, training_episodes=1, calibration_scores=[0.1] * 10,
            delta=0.2, eta=0.1, R_hat_init=0.2,
        )
        policy = np.array([0, 2, 0, 0])
        with patch("conformally_robust_controller.Tabular_Agent.robust_q_learning",
                   return_value=(np.zeros((mdp.n_state, 4)), policy)):
            first, second, third = controller.run(3)
        self.assertEqual([row["episode_length"] for row in controller.history], [2, 1, 2])
        self.assertEqual([row["transition_count"] for row in controller.history], [2, 3, 5])
        self.assertAlmostEqual(first["score"], 0.2)
        self.assertEqual(second["score"], 2.0)
        self.assertEqual(second["miscoverage"], 1)
        self.assertAlmostEqual(second["delta_next"], second["delta_k"] - 0.08)
        self.assertEqual(third["score"], 0.6)
        self.assertEqual(first["R_hat_next"], 0)
        self.assertAlmostEqual(second["R_hat_next"], 1 / 3)
        self.assertEqual(third["R_hat_next"], 0.4)
        self.assertEqual(second["R_used"], first["R_hat_next"])
        self.assertEqual(third["R_used"], second["R_hat_next"])
        self.assertEqual(controller.acp.n_updates, 3)
        self.assertAlmostEqual(controller.acp.delta_k, 0.06)

    def test_pilot_scores_do_not_enter_R_counts(self):
        controller = self.controller(calibration_scores=None, pilot_episodes=3)
        controller.initialize_calibration()
        self.assertEqual(len(controller.pilot_scores), 3)
        self.assertEqual(controller.transition_count, 0)
        self.assertEqual(controller.mismatch_count, 0)
        record = controller.run_episode()
        self.assertEqual(controller.transition_count, record["episode_length"])

    def test_zero_noise_stops_at_goal_with_zero_radius_prediction(self):
        mdp = Robot2DMotionMDP(x_max=2, y_max=2, grid_shape=(2, 2), movement_reward=0)
        controller = ConformallyRobustController(
            mdp, horizon=6, training_episodes=200, calibration_scores=[0] * 5,
            R_hat_init=0, delta=0.2, eta=0.1, seed=2,
        )
        controller.acp.delta_k = 1.1
        record = controller.run_episode()
        self.assertTrue(record["goal_reached"])
        self.assertEqual(record["episode_length"], 2)
        self.assertFalse(np.any(np.all(record["states"][:-1] == mdp.goal_state, axis=1)))
        self.assertEqual(record["transition_count"], 2)
        self.assertEqual(record["R_hat_next"], 0)
        self.assertEqual(record["score"], 0)
        self.assertEqual(record["C_hat"], 0)
        self.assertEqual(record["C_used"], 0)
        self.assertEqual(record["miscoverage"], 0)

    def test_fixed_radius_is_validated_and_passed_directly_to_training(self):
        for C in (0.0, 1.0, np.inf):
            with self.subTest(C=C):
                controller = self.controller(C=C, training_episodes=2000)
                record = controller.run_episode()
                self.assertNotIn("C_hat", record)
                self.assertEqual(record["C_used"], C)
                self.assertEqual(controller.agent.C, C)
                self.assertIsNone(controller.acp)
        for C in (np.nan, -0.1):
            with self.subTest(C=C):
                controller = self.controller(C=0.0, training_episodes=2000)
                controller.fixed_C = C
                with self.assertRaises(ValueError):
                    controller.run_episode()

    def test_R_and_C_independently_control_updates_and_policy_reuse(self):
        for R, C in ((None, None), (0.3, None), (None, 0.7), (0.3, 0.7)):
            with self.subTest(R=R, C=C):
                controller = self.controller(R=R, C=C, training_episodes=2000)
                first = controller.run_episode()
                first_agent = controller.agent
                second = controller.run_episode()
                self.assertFalse(hasattr(controller, "method"))
                self.assertEqual(second["training_reused"], R is not None and C is not None)
                if R is not None and C is not None:
                    self.assertIs(controller.agent, first_agent)
                else:
                    self.assertIsNot(controller.agent, first_agent)
                if R is None:
                    count = first["episode_length"] + second["episode_length"]
                    self.assertEqual(second["transition_count"], count)
                    mismatches = first["episode_mismatches"] + second["episode_mismatches"]
                    self.assertEqual(second["mismatch_count"], mismatches)
                    self.assertEqual(second["R_used"], first["R_hat_next"])
                    self.assertAlmostEqual(controller.R_hat, mismatches / count)
                else:
                    self.assertEqual(second["R_used"], R)
                    self.assertIsNone(controller.R_hat)
                    self.assertEqual(controller.transition_count, 0)
                    self.assertEqual(controller.mismatch_count, 0)
                    self.assertNotIn("R_hat_next", second)
                    self.assertNotIn("transition_count", second)
                    self.assertNotIn("mismatch_count", second)
                if C is None:
                    self.assertEqual(controller.acp.n_updates, 2)
                    self.assertEqual(second["C_used"], second["C_hat"])
                    self.assertEqual(controller.acp.scores[-1], second["score"])
                else:
                    self.assertEqual(second["C_used"], C)
                    self.assertIsNone(controller.acp)
                    self.assertFalse(controller.pilot_scores)
                    self.assertNotIn("C_hat", second)
                    self.assertNotIn("miscoverage", second)

    def test_seeded_controllers_repeat_the_same_run(self):
        mdp = Robot2DMotionMDP(
            x_max=2, y_max=2, grid_shape=(2, 2), movement_reward=0,
            noise_probability=1, noise_radius=0.3,
        )
        options = dict(horizon=6, training_episodes=2000, calibration_scores=[0.1] * 10, seed=5)
        first = ConformallyRobustController(mdp, **options)
        second = ConformallyRobustController(mdp, **options)
        for left, right in zip(first.run(3), second.run(3)):
            np.testing.assert_array_equal(left["states"], right["states"])
            np.testing.assert_allclose(left["positions"], right["positions"])
            np.testing.assert_allclose(left["deviation_vectors"], right["deviation_vectors"])
        np.testing.assert_allclose(first.Q, second.Q)
        self.assertFalse(np.array_equal(
            first.history[0]["deviation_vectors"], first.history[1]["deviation_vectors"],
        ))

    def test_initializer_gets_planning_inputs_without_being_saved_in_the_agent(self):
        calls = []

        def initializer(mdp, R_used, C_used):
            calls.append((mdp, R_used, C_used))
            return np.full((len(mdp.states), mdp.action_space.n), 0.5)

        adaptive = self.controller(q_initializer=initializer, training_episodes=2000)
        for record in adaptive.run(2):
            self.assertFalse(record["training"]["zero_initialized"])
        self.assertEqual(len(calls), 2)
        for call, record in zip(calls, adaptive.history):
            self.assertIs(call[0], adaptive.training_env)
            self.assertEqual(call[1:], (record["R_used"], record["C_used"]))
        self.assertFalse(hasattr(adaptive.agent, "q_initializer"))
        calls.clear()
        fixed = self.controller(q_initializer=initializer, R=0.3, C=np.inf, training_episodes=2000)
        fixed.run(2)
        self.assertEqual(len(calls), 1)
        self.assertTrue(fixed.history[1]["training_reused"])

    def test_convergence_options_are_forwarded_to_training(self):
        controller = self.controller(training_tolerance=1e-5, training_check_every=7)
        record = controller.run_episode()
        diagnostics = record["training"]
        self.assertTrue(diagnostics["converged"])
        self.assertEqual(diagnostics["tolerance"], 1e-5)
        self.assertEqual(diagnostics["episodes"] % 7, 0)
        self.assertLess(diagnostics["episodes"], controller.training_episodes)

    def test_capped_training_does_not_deploy_an_unconverged_policy(self):
        controller = self.controller(training_episodes=1, R=0, C=0)
        with patch.object(controller.deployment_env, "deploy_policy") as deploy:
            with self.assertRaisesRegex(RuntimeError, "Q-learning did not converge"):
                controller.run_episode()
            failed_agent = controller.agent
            with self.assertRaisesRegex(RuntimeError, "Q-learning did not converge"):
                controller.run_episode()
            self.assertIsNot(controller.agent, failed_agent)
        deploy.assert_not_called()
        self.assertFalse(controller.agent.training_diagnostics["converged"])
        self.assertFalse(controller.history)
        self.assertEqual(controller.transition_count, 0)

    def test_invalid_convergence_settings_fail_before_deployment(self):
        for settings in ({"training_tolerance": 0}, {"training_check_every": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.controller(**settings)

if __name__ == "__main__":
    unittest.main()
