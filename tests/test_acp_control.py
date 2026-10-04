"""Coverage bookkeeping, controller lifecycle, and runner integration checks."""

from contextlib import redirect_stdout
import io
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from acp import ACP
from conformally_robust_controller import ConformallyRobustController
from main_robot_2d_crc import main as online_main
from robot_2d_motion_mdp import Robot2DMotionMDP


class ACPTests(unittest.TestCase):
    def test_order_statistic_uses_N_plus_one_without_interpolation(self):
        calibration = ACP([0, 1, 2, 3], target_failure=0.5)
        self.assertEqual(calibration.quantile(), 2)

    def test_quantile_precedes_append_and_update_is_unclipped(self):
        calibration = ACP([0, 1, 2, 3], target_failure=0.5, eta=0.1)
        prediction = calibration.quantile()
        update = calibration.update(9, prediction)
        self.assertEqual(update["miscoverage"], 1)
        self.assertAlmostEqual(calibration.delta, 0.45)
        self.assertEqual(calibration.scores[-1], 9)

    def test_extended_quantile_tails_and_zero_radius_fallback(self):
        calibration = ACP([0], target_failure=0.1, eta=0.2)
        self.assertEqual(calibration.quantile(), np.inf)
        calibration.delta = 1.1
        self.assertEqual(calibration.quantile(), -np.inf)
        self.assertEqual(calibration.planning_radius(calibration.quantile()), 0)
        self.assertEqual(calibration.update(0)["miscoverage"], 1)
        self.assertAlmostEqual(calibration.delta, 0.92)

    def test_deque_window_discards_old_scores(self):
        calibration = ACP([0, 1, 2], window_size=2)
        self.assertEqual(list(calibration.scores), [1, 2])
        calibration.update(3)
        self.assertEqual(list(calibration.scores), [2, 3])

    def test_telescoping_ACP_identity(self):
        calibration = ACP([0.2] * 10, target_failure=0.2, eta=0.1)
        misses = 0
        for score in (0.1, 0.8, 0.3, 0, 0.9, 0.2):
            misses += calibration.update(score)["miscoverage"]
        expected = calibration.initial_delta + calibration.eta * (6 * 0.2 - misses)
        self.assertAlmostEqual(calibration.delta, expected)


class ControllerTests(unittest.TestCase):
    def controller(self, **kwargs):
        mdp = Robot2DMotionMDP(
            x_max=2, y_max=2, grid_shape=(2, 2), movement_reward=0,
            noise_sampler=lambda position, rng: np.array((0.2, 0)),
        )
        options = dict(
            horizon=4, training_episodes=100, training_horizon=10, initial_R=0.2,
            calibration_scores=[0.1] * 10, target_failure=0.2, eta=0.1, seed=5,
        )
        options.update(kwargs)
        return ConformallyRobustController(mdp, **options)

    def test_episode_statistics_use_real_deployment_observations(self):
        controller = self.controller()
        record = controller.run_episode()
        self.assertEqual(controller.transition_count, 4)
        self.assertEqual(controller.mismatch_count, np.sum(record["mismatches"]))
        self.assertAlmostEqual(record["score"], max(record["deviations"]))
        self.assertAlmostEqual(record["R_next"], np.sum(record["mismatches"]) / 4)
        self.assertAlmostEqual(record["conformal_radius"], 0.1)
        self.assertEqual(record["miscoverage"], int(record["score"] > 0.1))
        self.assertAlmostEqual(record["delta_after"], 0.2 + 0.1 * (0.2 - record["miscoverage"]))
        self.assertEqual(record["positions"].shape, (5, 2))
        self.assertEqual(len(record["actions"]), 4)
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
        self.assertAlmostEqual(second["R_used"], first["R_next"])
        self.assertEqual(second["transition_count"], 8)
        self.assertEqual(second["mismatch_count"],
                         first["episode_mismatches"] + second["episode_mismatches"])
        self.assertAlmostEqual(second["R_next"], second["mismatch_count"] / 8)

    def test_pilot_scores_do_not_enter_R_counts(self):
        controller = self.controller(calibration_scores=None, pilot_episodes=3)
        controller.initialize_calibration()
        self.assertEqual(len(controller.pilot_scores), 3)
        self.assertEqual(controller.transition_count, 0)
        self.assertEqual(controller.mismatch_count, 0)
        controller.run_episode()
        self.assertEqual(controller.transition_count, 4)

    def test_zero_noise_counts_absorbing_steps_and_retains_empty_predictions(self):
        mdp = Robot2DMotionMDP(x_max=2, y_max=2, grid_shape=(2, 2), movement_reward=0)
        controller = ConformallyRobustController(
            mdp, horizon=6, training_episodes=200, calibration_scores=[0] * 5,
            initial_R=0, target_failure=0.2, eta=0.1, seed=2,
        )
        controller.calibrator.delta = 1.1
        record = controller.run_episode()
        self.assertTrue(record["goal_reached"])
        self.assertEqual(record["absorbing_transitions"], 4)
        self.assertEqual(record["transition_count"], 6)
        self.assertEqual(record["R_next"], 0)
        self.assertEqual(record["score"], 0)
        self.assertEqual(record["miscoverage"], 1)
        self.assertEqual(record["planned_miscoverage"], 0)

    def test_seeded_controllers_repeat_the_same_run(self):
        first, second = self.controller(), self.controller()
        left, right = first.run_episode(), second.run_episode()
        np.testing.assert_array_equal(left["states"], right["states"])
        np.testing.assert_allclose(left["positions"], right["positions"])
        np.testing.assert_allclose(first.Q, second.Q)


class RunnerIntegrationTests(unittest.TestCase):
    def test_runner_saves_single_and_multiple_episode_results_without_MDP_JSON(self):
        with tempfile.TemporaryDirectory(prefix="robot_motion_tests_") as directory:
            root = Path(directory)
            for n_episodes in (1, 2):
                with self.subTest(n_episodes=n_episodes):
                    output_dir = root / str(n_episodes)
                    with redirect_stdout(io.StringIO()):
                        controller = online_main(
                            training_episodes=10, horizon=5, n_episodes=n_episodes,
                            pilot_episodes=2, save_plots=False, output_dir=output_dir,
                        )
                    with (output_dir / "results.pkl").open("rb") as file:
                        results = pickle.load(file)
                    history = results["history"]
                    self.assertEqual(len(history), n_episodes)
                    self.assertEqual(history[-1]["transition_count"], 5 * n_episodes)
                    self.assertEqual(history[0]["R_used"], 0.1)
                    self.assertTrue(
                        all(record["training"]["zero_initialized"] for record in history)
                    )
                    self.assertEqual(results["final_R"], controller.R_estimate)
                    self.assertEqual(results["final_delta"], controller.calibrator.delta)
                    with np.load(output_dir / "final_policy.npz") as saved:
                        self.assertEqual(saved["Q"].shape, (64, 4))
                        self.assertEqual(saved["policy"].shape, (64,))
                        np.testing.assert_array_equal(saved["Q"], controller.Q)
            self.assertFalse(list(root.rglob("*.json")))


if __name__ == "__main__":
    unittest.main()
