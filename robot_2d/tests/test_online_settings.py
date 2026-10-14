"""Check the randomized maps and state-independent mismatch law of the online experiment."""

import unittest
from unittest.mock import Mock

import numpy as np

from robot_2d.online_settings import randomize_obstacles, sample_deviation, support_radius
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP


class EpisodeSettingsTests(unittest.TestCase):
    def test_trial_maps_reproduce_the_seed_and_change_for_another_seed(self):
        maps = []
        for seed in (87, 87, 88):
            mdp = Robot2DMotionMDP(x_max=10, y_max=10, grid_shape=(10, 10))
            randomize_obstacles(mdp, np.random.default_rng(seed))
            maps.append(mdp.obstacle_mask.copy())
        np.testing.assert_array_equal(maps[0], maps[1])
        self.assertFalse(np.array_equal(maps[0], maps[2]))

    def test_support_cycles_gradually_within_requested_bounds(self):
        radii = np.array([support_radius(episode) for episode in range(201)])
        self.assertGreater(radii.min(), 0.5)
        self.assertLessEqual(radii.max(), 2.0)
        self.assertAlmostEqual(radii.min(), 0.5, delta=0.001)
        self.assertAlmostEqual(radii.max(), 2.0, delta=0.001)
        self.assertLess(np.max(np.abs(np.diff(radii))), 0.05)
        np.testing.assert_allclose(radii[:101], radii[100:])
        with self.assertRaises(ValueError):
            support_radius(0, 0)

    def test_every_nonzero_component_mismatches_at_all_nominal_targets(self):
        mdp = Robot2DMotionMDP(x_max=10, y_max=10, grid_shape=(10, 10))
        rng = Mock()
        rng.random.return_value = 0.0
        rng.uniform.side_effect = lambda low, high: high
        for radius in (support_radius(0), 1.0, 2.0):
            for state in range(mdp.n_grid_states):
                position = mdp.coordinates(state)
                for direction in range(4):
                    rng.integers.return_value = direction
                    deviation = sample_deviation(position, rng, radius=radius, true_R=0.2)
                    successor = mdp.discretize(position + deviation)
                    self.assertNotEqual(successor, state)
                    self.assertAlmostEqual(np.linalg.norm(deviation), radius)
        rng.random.return_value = 0.8
        deviation = sample_deviation(mdp.bounds - 0.5, rng, radius=2, true_R=0.2)
        np.testing.assert_array_equal(deviation, [0, 0])

    def test_mismatch_frequency_stays_fixed_as_support_changes(self):
        mdp = Robot2DMotionMDP(x_max=10, y_max=10, grid_shape=(10, 10))
        for radius, cell in ((support_radius(0), (0, 0)), (1.0, (5, 5)), (2.0, (9, 9))):
            rng = np.random.default_rng(5)
            position = mdp.coordinates(mdp.cell_to_state(cell))
            deviations = np.array([
                sample_deviation(position, rng, radius=radius, true_R=0.2)
                for _ in range(10000)
            ])
            norms = np.linalg.norm(deviations, axis=1)
            self.assertLessEqual(norms.max(), radius)
            self.assertGreater(norms[norms > 0].min(), 0.5)
            mismatches = [mdp.discretize(position + delta) != mdp.cell_to_state(cell)
                          for delta in deviations]
            self.assertAlmostEqual(np.mean(mismatches), 0.2, delta=0.015)

    def test_random_maps_have_ten_protected_obstacles_and_a_path(self):
        mdp = Robot2DMotionMDP(x_max=10, y_max=10, grid_shape=(10, 10))
        rng = np.random.default_rng(7)
        previous = set()
        for _ in range(50):
            obstacles = randomize_obstacles(mdp, rng)
            self.assertEqual(len(set(obstacles)), 10)
            self.assertEqual(np.count_nonzero(mdp.obstacle_mask), 10)
            self.assertNotEqual(set(obstacles), previous)
            for cell in obstacles:
                for protected in (mdp.start_cell, mdp.goal_cell):
                    self.assertGreaterEqual(np.max(np.abs(np.array(cell) - protected)), 2)
            mdp._validate_path()
            previous = set(obstacles)

    def test_map_updates_rebuild_rewards_and_terminal_states(self):
        mdp = Robot2DMotionMDP(x_max=4, y_max=4, grid_shape=(4, 4))
        source = mdp.cell_to_state((0, 0))
        target = mdp.cell_to_state((1, 0))
        self.assertEqual(mdp.reward(source, 0), 0.01)
        mdp.set_obstacles([(1, 0)])
        self.assertTrue(mdp.terminal_mask[target])
        self.assertEqual(mdp.reward(source, 0), 0)
        self.assertEqual(mdp.nominal_next_state(source, 0), target)
        mdp.set_obstacles([])
        self.assertFalse(mdp.terminal_mask[target])
        self.assertEqual(mdp.reward(source, 0), 0.01)
        mdp.set_obstacles([(1, 0)])
        with self.assertRaises(ValueError):
            mdp.set_obstacles([(1, 0), (0, 1)])
        self.assertTrue(mdp.terminal_mask[target])
        self.assertEqual(np.count_nonzero(mdp.obstacle_mask), 1)


if __name__ == "__main__":
    unittest.main()
