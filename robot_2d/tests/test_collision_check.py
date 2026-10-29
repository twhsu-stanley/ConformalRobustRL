"""Check grid traversal and calibration when contact interrupts deployment."""

import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from conformally_robust_controller import ConformallyRobustController
from robot_2d.plot_online_comparison import plot_acp
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP
from robot_2d.utils import plot_conformal_history


class SegmentCollisionTests(unittest.TestCase):
    def test_crossings_from_both_directions_and_with_rectangular_cells(self):
        mdp = Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0, obstacles=[(2, 1)],
        )
        cases = (
            ((0.5, 1.5), (4.5, 1.5), (2.0, 1.5)),
            ((4.5, 1.5), (0.5, 1.5), (3.0, 1.5)),
            ((2.5, 0.5), (2.5, 4.5), (2.5, 1.0)),
            ((2.5, 4.5), (2.5, 0.5), (2.5, 2.0)),
            ((1.5, 0.5), (3.5, 2.5), (2.0, 1.0)),
        )
        for start, end, expected in cases:
            with self.subTest(start=start, end=end):
                state, position = mdp._collision_check_segment(start, end)
                self.assertEqual(state, (2, 1))
                np.testing.assert_allclose(position, expected)
        rectangular = Robot2DMotionMDP(
            x_max=16, y_max=8, grid_shape=(8, 8), start_cell=(0, 0), goal_cell=(7, 7), gamma=0.9,
            movement_reward=0.0, goal_reward=1.0, obstacles=[(2, 1)],
        )
        state, position = rectangular._collision_check_segment((1, 1.5), (9, 1.5))
        self.assertEqual(state, (2, 1))
        np.testing.assert_allclose(position, (4, 1.5))

    def test_corner_grazing_grid_lines_and_boundary_endpoints(self):
        mdp = Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0, obstacles=[(1, 0)],
        )
        for start, end in (
            ((0.5, 0.5), (1.5, 1.5)),
            ((0.5, 1.0), (2.5, 1.0)),
            ((0.5, 0.5), (1.0, 0.5)),
            ((1.0, 0.5), (0.5, 0.5)),
        ):
            with self.subTest(start=start, end=end):
                state, _ = mdp._collision_check_segment(start, end)
                self.assertEqual(state, (1, 0))
        self.assertIsNone(mdp._collision_check_segment((0.5, 0.5), (0.5, 0.5)))
        self.assertIsNone(mdp._collision_check_segment((0.5, 2.5), (0.5, 2.5)))
        self.assertIsNone(mdp._collision_check_segment((0.5, 0.5), (-3, 0.5)))
        self.assertIsNone(mdp._collision_check_segment((0.5, 2.5), (8, 2.5)))
        self.assertIsNone(mdp._collision_check_segment((0.5, 0.5), (1.5, 1.500001)))
        state, _ = mdp._collision_check_segment((0.5, 0.5), (1.5, 1.499999))
        self.assertEqual(state, (1, 0))

    def test_first_terminal_event_and_obstacle_priority_at_a_corner(self):
        mdp = Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(3, 1),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0, obstacles=[(5, 1)],
        )
        state, position = mdp._collision_check_segment((0.5, 1.5), (9, 1.5))
        self.assertEqual(state, mdp.goal_state)
        np.testing.assert_allclose(position, (3, 1.5))
        state, position = mdp._collision_check_segment((7.5, 1.5), (0.5, 1.5))
        self.assertEqual(state, (5, 1))
        np.testing.assert_allclose(position, (6, 1.5))
        mdp = Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(1, 1),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0, obstacles=[(1, 0)],
        )
        state, position = mdp._collision_check_segment((0.5, 0.5), (1.5, 1.5))
        self.assertEqual(state, (1, 0))
        np.testing.assert_allclose(position, (1, 1))

    def test_traversal_matches_independent_rectangle_intersections(self):
        mdp = Robot2DMotionMDP(
            x_max=16, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0, obstacles=[(2, 1), (3, 4), (5, 6)],
        )
        rng = np.random.default_rng(37)
        terminal_states = [tuple(s) for s in mdp.states[mdp.terminal_mask]]
        for _ in range(150):
            start = rng.uniform(0, mdp.bounds)
            end = rng.uniform(-mdp.cell_widths, mdp.bounds + mdp.cell_widths)
            direction = end - start
            intersections = []
            for state in terminal_states:
                lower = mdp.coordinates(state) - mdp.cell_widths / 2
                upper = lower + mdp.cell_widths
                first, last = (lower - start) / direction, (upper - start) / direction
                enter = max(0.0, np.minimum(first, last).max())
                leave = min(1.0, np.maximum(first, last).min())
                if enter <= leave:
                    intersections.append((enter, state))
            actual = mdp._collision_check_segment(start, end)
            if not intersections:
                self.assertIsNone(actual)
            else:
                parameter, expected = min(intersections)
                self.assertEqual(actual[0], expected)
                np.testing.assert_allclose(actual[1], start + parameter * direction, atol=1e-12)

    def test_very_long_segments_do_not_merge_distinct_grid_crossings(self):
        mdp = Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0, obstacles=[(1, 1)],
        )
        state, position = mdp._collision_check_segment((0.5, 0.5), (1e12, 2e12))
        self.assertEqual(state, (1, 1))
        np.testing.assert_allclose(position, (1, 1.5), atol=1e-11)

    def test_nominal_training_keeps_its_existing_endpoint_transitions(self):
        mdp = Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0, obstacles=[(1, 0)],
        )
        with patch.object(mdp, "_collision_check_segment", side_effect=AssertionError):
            state, _, terminated, _, _ = mdp.step(0)
        np.testing.assert_array_equal(state, (1, 0))
        np.testing.assert_allclose(mdp.position, (1.5, 0.5))
        self.assertTrue(terminated)


class ContactCalibrationTests(unittest.TestCase):
    def test_terminal_disturbance_is_scored_for_obstacle_goal_and_exit(self):
        for outcome in ("collision", "goal_reached", "workspace_exit"):
            with self.subTest(outcome=outcome):
                samples = iter(((0.1, 0), (10 if outcome == "workspace_exit" else 1.5, 0)))
                mdp = Robot2DMotionMDP(
                    x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0),
                    goal_cell=(3, 0) if outcome == 'goal_reached' else (7, 7), gamma=0.9,
                    movement_reward=0.0, goal_reward=1.0,
                    obstacles=[(3, 0)] if outcome == 'collision' else [], noise_radius=10,
                    noise_sampler=lambda position, rng: np.array(next(samples)),
                )
                record = mdp.deploy_policy(np.zeros(mdp.n_state, dtype=int))
                self.assertTrue(record[outcome])
                self.assertEqual(record["episode_length"], 2)
                self.assertEqual(record["score"], record["deviations"][-1])
                self.assertGreater(record["score"], 0.1)
                self.assertEqual(record["episode_mismatches"], 1)
                if outcome != "workspace_exit":
                    np.testing.assert_allclose(record["positions"][-1], (3, 0.5))

    def test_time_limit_retains_the_last_nonterminal_disturbance(self):
        samples = iter(((0.1, 0), (0.4, 0)))
        mdp = Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0, obstacles=[], max_episode_steps=2,
            noise_sampler=lambda position, rng: np.array(next(samples)),
        )
        record = mdp.deploy_policy(np.zeros(mdp.n_state, dtype=int))
        self.assertTrue(record["timeout"])
        self.assertEqual(record["score"], 0.4)

    def test_first_step_contact_keeps_endpoint_mismatch_and_updates_ACP(self):
        mdp = Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0, obstacles=[(1, 0)],
            noise_sampler=lambda position, rng: np.array((1.0, 0)),
        )
        controller = ConformallyRobustController(
            mdp, max_training_episodes=1, calibration_scores=[0.1] * 10, delta=0.2, seed=5,
        )
        policy = np.zeros(mdp.n_state, dtype=int)
        with patch("conformally_robust_controller.Tabular_Agent.robust_q_learning",
                   return_value=(np.zeros((mdp.n_state, 4)), policy)):
            controller.run(2)
        for record in controller.history:
            record["training"]["bellman_residual"] = 0.0
            np.testing.assert_array_equal(
                record["states"][-1], mdp.nominal_next_state(mdp.start_state, 0),
            )
            np.testing.assert_allclose(record["positions"][-1], (1, 0.5))
            np.testing.assert_allclose(
                record["nominal_positions"][-1] + record["deviation_vectors"][-1], (2.5, 0.5),
            )
            self.assertEqual(record["episode_mismatches"], 1)
            self.assertEqual(record["score"], 1.0)
            self.assertEqual(record["miscoverage"], 1)
            self.assertAlmostEqual(record["delta_next"], record["delta_k"] - 0.008)
        self.assertEqual(controller.transition_count, 2)
        self.assertEqual(controller.R_hat, 1)
        self.assertEqual(controller.acp.n_updates, 2)
        self.assertEqual(list(controller.acp.scores), [0.1] * 10 + [1.0, 1.0])
        run = {
            "mdp": controller.deployment_env, "history": controller.history,
            "initial_calibration_scores": controller.initial_calibration_scores,
            "pilot_scores": controller.pilot_scores,
            "delta": 0.2, "delta_1": 0.2, "calibration_window": 100,
        }
        for figure in (plot_conformal_history(controller.history, 0.2), plot_acp([run])):
            try:
                figure.canvas.draw()
                if len(figure.axes) == 6:
                    coverage = np.ma.asarray(figure.axes[3].lines[0].get_ydata()).filled(np.nan)
                    np.testing.assert_array_equal(coverage, [0, 0])
                else:
                    self.assertEqual(len(figure.axes), 2)
                    np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), [1, 1])
                    np.testing.assert_allclose(figure.axes[1].lines[0].get_ydata(), [0.2, 0.192])
            finally:
                plt.close(figure)

    def test_every_pilot_including_a_zero_score_enters_calibration(self):
        controller = ConformallyRobustController(Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0,
        ), pilot_episodes=3)
        records = [{"score": 0.2}, {"score": 0}, {"score": 0.4}]
        with patch.object(controller.deployment_env, "deploy_policy", side_effect=records):
            controller.initialize_calibration()
        self.assertEqual(controller.pilot_scores, [0.2, 0, 0.4])
        self.assertEqual(list(controller.acp.scores), [0.2, 0, 0.4])
        self.assertEqual(controller.transition_count, 0)


if __name__ == "__main__":
    unittest.main()
