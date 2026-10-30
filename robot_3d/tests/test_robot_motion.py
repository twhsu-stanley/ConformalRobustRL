"""Geometric, transition, collision, and solver checks for the 3-D robot MDP."""

import unittest
from unittest.mock import patch

import numpy as np
from gymnasium.utils.env_checker import check_env

from Tabular_Agent import Tabular_Agent
from robot_3d.robot_3d_motion_mdp import Robot3DMotionMDP
from robot_3d.utils import C_star_shift, sample_deviation


class RobotMotionTests(unittest.TestCase):
    def mdp(self, **settings):
        return Robot3DMotionMDP(
            x_max=3, y_max=3, z_max=3, grid_shape=(3, 3, 3),
            start_cell=(0, 0, 0), goal_cell=(2, 2, 2), gamma=0.9,
            movement_reward=0, goal_reward=1, **settings,
        )

    def test_gym_interface_and_full_workspace_indexing(self):
        mdp = Robot3DMotionMDP(
            x_max=10.0, y_max=10.0, z_max=10.0, grid_shape=(10, 10, 10),
            start_cell=(0, 0, 0), goal_cell=(9, 9, 9), gamma=0.9,
            movement_reward=0.0, goal_reward=1.0,
        )
        check_env(mdp, skip_render_check=True)
        self.assertEqual(mdp.obstacle_mask.shape, (10, 10, 10))
        self.assertEqual(mdp.action_space.n, 6)
        self.assertEqual(Tabular_Agent(mdp, lr_init=0.5).Q.shape, (1000, 6))
        for row, state in enumerate(mdp.states):
            self.assertEqual(mdp.state_index(state), row)
            np.testing.assert_array_equal(mdp.discretize(mdp.coordinates(state)), state)
        self.assertEqual(mdp.state_index((3, 4, 5)), 543)
        self.assertEqual(mdp.discretize((10.0, -0.1, 3.0)), (10, -1, 3))
        np.testing.assert_array_equal(mdp.coordinates((10, -1, 3)), (10.5, -0.5, 3.5))

    def test_nominal_commands_rewards_and_terminal_sources(self):
        mdp = self.mdp(obstacles=[(1, 1, 1)])
        for action, direction in enumerate(mdp.ACTION_DIRECTIONS):
            source = (1, 1, 0)
            expected = tuple(max(0, s + d) for s, d in zip(source, direction))
            self.assertEqual(mdp.nominal_next_state(source, action), expected)
        self.assertEqual(mdp.nominal_next_state((0, 0, 0), 5), (0, 0, 0))
        self.assertEqual(mdp.reward((2, 2, 1), 4), 1)
        self.assertEqual(mdp.reward((0, 0, 0), 0), 0)
        for state in (mdp.goal_state, (1, 1, 1), (3, 1, 1), (-1, 1, 1)):
            self.assertTrue(mdp.is_terminal(state))
            for action in range(6):
                self.assertEqual(mdp.nominal_next_state(state, action), state)
                self.assertEqual(mdp.reward(state, action), 0)
            self.assertEqual(mdp.values_at(state, np.ones(mdp.n_state)), 0)
            with self.assertRaises(ValueError):
                mdp.reset(options={"state": state})

    def test_neighborhoods_match_independent_distances_to_voxel_boxes(self):
        mdp = Robot3DMotionMDP(
            x_max=6, y_max=4, z_max=6, grid_shape=(3, 4, 2),
            start_cell=(0, 0, 0), goal_cell=(2, 3, 1), gamma=0.9,
            movement_reward=0, goal_reward=1,
        )
        cells = np.array([(x, y, z) for x in range(-5, 8) for y in range(-5, 9)
                          for z in range(-5, 7)])
        lower, upper = cells * mdp.cell_widths, (cells + 1) * mdp.cell_widths
        for nominal in ((0, 0, 0), (1, 2, 1), (3, -1, 2), mdp.goal_state):
            center = mdp.coordinates(nominal)
            distances = np.linalg.norm(np.clip(center, lower, upper) - center, axis=1)
            for radius in (0, 0.5, 0.6, 1.1, 2.2):
                expected = set(map(tuple, cells[distances <= radius + 1e-12]))
                actual = set(map(tuple, mdp.localized_uncertainty_states(nominal, radius)))
                self.assertEqual(actual, expected)
        for state, threshold in (((0, 0, 0), 0.5), ((1, 2, 0), 1.5)):
            for radius in (threshold - 1e-8, threshold, threshold + 1e-8):
                outside = any(mdp.is_outside(s)
                              for s in mdp.localized_uncertainty_states(state, radius))
                self.assertEqual(outside, radius >= threshold)

    def test_terminal_successors_keep_the_full_uncertainty_neighborhood(self):
        mdp = self.mdp(obstacles=[(1, 1, 1)])
        values = np.full(mdp.n_state, -0.2)
        agent = Tabular_Agent(mdp, lr_init=0.5, R=0.25, C=0.6)
        for terminal, neighbor in (
            (mdp.goal_state, (1, 2, 2)), ((1, 1, 1), (0, 1, 1)), ((3, 1, 1), (2, 1, 1)),
        ):
            states = set(map(tuple, mdp.localized_uncertainty_states(terminal, 0.6)))
            self.assertIn(neighbor, states)
            self.assertEqual(mdp.values_at(terminal, values), 0)
            self.assertAlmostEqual(mdp.worst_case_value(terminal, 0.6, values), -0.2)
            self.assertAlmostEqual(agent.robust_td_target(1, terminal, values), 0.955)
            self.assertEqual(mdp.worst_case_value(terminal, 0, values), 0)
        self.assertEqual(mdp.worst_case_value((0, 0, 0), np.inf, np.ones(mdp.n_state)), 0)

    def test_collision_checks_match_independent_segment_box_intersections(self):
        mdp = self.mdp(obstacles=[(1, 1, 0), (1, 0, 2), (2, 1, 2)])
        terminals = [tuple(cell) for cell in mdp.states[mdp.terminal_mask]]
        rng = np.random.default_rng(37)
        for _ in range(200):
            start, end = rng.uniform(0, mdp.bounds), rng.uniform(-1, mdp.bounds + 1)
            direction = end - start
            candidates = []
            for state in terminals:
                lower = mdp.coordinates(state) - mdp.cell_widths / 2
                upper = lower + mdp.cell_widths
                left, right = (lower - start) / direction, (upper - start) / direction
                enter, leave = max(0, np.minimum(left, right).max()), min(1, np.maximum(left,
                                                                                     right).min())
                if enter <= leave:
                    candidates.append((enter, state))
            actual = mdp._collision_check_segment(start, end)
            if candidates:
                parameter, state = min(candidates)
                self.assertEqual(actual[0], state)
                np.testing.assert_allclose(actual[1], start + parameter * direction, atol=1e-12)
            else:
                self.assertIsNone(actual)

    def test_corner_edge_and_stationary_contacts(self):
        mdp = self.mdp(obstacles=[(1, 0, 0), (1, 1, 1)])
        for start, end in (
            ((0.5, 0.5, 0.5), (1.5, 1.5, 1.5)),
            ((0.5, 1, 1), (2.5, 1, 1)),
            ((1, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ):
            contact = mdp._collision_check_segment(start, end)
            self.assertEqual(contact[0], (1, 0, 0))
        self.assertIsNone(mdp._collision_check_segment((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.assertIsNone(mdp._collision_check_segment((0.5, 0.5, 0.5), (-10, 0.5, 0.5)))
        state, point = mdp._collision_check_segment((0.5, 0.5, 0.5), (1e12, 2e12, 2e12))
        self.assertEqual(state, (1, 1, 1))
        np.testing.assert_allclose(point, (1, 1.5, 1.5), atol=1e-11)

    def test_uncertain_contact_retains_raw_mismatch_and_terminal_score(self):
        mdp = self.mdp(
            obstacles=[(2, 0, 0)], noise_radius=2,
            noise_sampler=lambda position, rng: np.array((2, 0, 0)),
        )
        record = mdp.deploy_policy(lambda state: 0)
        np.testing.assert_array_equal(record["states"][-1], (2, 0, 0))
        np.testing.assert_allclose(record["positions"][-1], (2, 0.5, 0.5))
        raw = record["nominal_positions"][-1] + record["deviation_vectors"][-1]
        np.testing.assert_allclose(raw, (3.5, 0.5, 0.5))
        self.assertTrue(record["collision"])
        self.assertEqual(record["episode_length"], 1)
        self.assertEqual(record["score"], 2)
        self.assertEqual(record["episode_mismatches"], 1)

    def test_exits_and_time_limits_keep_all_observations(self):
        mdp = self.mdp(noise_radius=4, noise_sampler=lambda position, rng: np.array((-4, 0, 0)))
        record = mdp.deploy_policy(lambda state: 1)
        np.testing.assert_array_equal(record["states"][-1], (-4, 0, 0))
        np.testing.assert_allclose(record["positions"][-1], (-3.5, 0.5, 0.5))
        self.assertEqual(record["score"], 4)
        self.assertTrue(record["workspace_exit"])
        mdp = self.mdp(max_episode_steps=3)
        record = mdp.deploy_policy(lambda state: 1)
        self.assertTrue(record["timeout"])
        self.assertEqual(record["episode_length"], 3)
        self.assertEqual(record["score"], 0)

    def test_ball_sampler_and_shifting_axis_sampler(self):
        mdp = self.mdp(noise_probability=1, noise_radius=1.2, seed=7)
        draws = np.array([mdp._sample_deviation(np.zeros(3)) for _ in range(2000)])
        radii = np.linalg.norm(draws, axis=1)
        self.assertLessEqual(radii.max(), 1.2)
        self.assertAlmostEqual(np.mean((radii / 1.2)**3), 0.5, delta=0.04)
        np.testing.assert_allclose(draws.mean(axis=0), 0, atol=0.04)
        rng = np.random.default_rng(7)
        for episode in (0, 49, 99):
            radius = C_star_shift(episode)
            draws = np.array([sample_deviation(np.zeros(3), rng, radius=radius, R_star=0.4)
                              for _ in range(2000)])
            lengths = np.linalg.norm(draws, axis=1)
            self.assertAlmostEqual(np.mean(lengths > 0.5), 0.4, delta=0.04)
            self.assertLessEqual(lengths.max(), radius)
            self.assertTrue(np.all((lengths == 0) | (lengths > 0.5)))
            self.assertEqual(set(np.argmax(np.abs(draws[lengths > 0]), axis=1)), {0, 1, 2})

    def test_q_learning_matches_value_iteration_and_clones_preserve_the_task(self):
        mdp = Robot3DMotionMDP(
            x_max=2, y_max=2, z_max=2, grid_shape=(2, 2, 2),
            start_cell=(0, 0, 0), goal_cell=(1, 1, 1), gamma=0.61,
            movement_reward=0, goal_reward=1,
        )
        for R, C in ((0, 0), (0.25, 0.6), (0.5, np.inf)):
            agent = Tabular_Agent(mdp.clone(nominal=True), lr_init=0.5, R=R, C=C, seed=7)
            expected, _ = agent.value_iteration(tolerance=1e-10)
            actual, policy = agent.robust_q_learning(max_training_episodes=5000)
            np.testing.assert_allclose(actual, expected, atol=1e-5)
            np.testing.assert_array_equal(actual[mdp.terminal_mask], 0)
            np.testing.assert_array_equal(agent.visit_counts[mdp.terminal_mask], 0)
            self.assertEqual(agent.training_diagnostics["transitions"], agent.visit_counts.sum())
            self.assertTrue(agent.training_diagnostics["converged"])
            clone = mdp.clone(seed=7)
            self.assertEqual(clone.gamma, 0.61)
            record = clone.deploy_policy(policy)
            self.assertAlmostEqual(record["discounted_return"], 0.61**2)
        with patch.object(mdp, "step", side_effect=AssertionError("No training needed")):
            agent = Tabular_Agent(mdp, lr_init=0.5)
            initial, _ = agent.value_iteration()
            agent.robust_q_learning(1, initial_Q=initial)
            self.assertEqual(agent.training_diagnostics["transitions"], 0)


if __name__ == "__main__":
    unittest.main()
