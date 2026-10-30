"""Check that a restricted 3-D task reproduces the existing 2-D implementation."""

import unittest

import numpy as np

from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP
from robot_3d.robot_3d_motion_mdp import Robot3DMotionMDP
from Tabular_Agent import Tabular_Agent


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.mdp_2d = Robot2DMotionMDP(
            x_max=4, y_max=3, grid_shape=(4, 3), start_cell=(0, 0), goal_cell=(3, 2),
            gamma=0.83, movement_reward=0.01, goal_reward=1, obstacles=[(1, 1)],
        )
        # The upper layer is blocked; its distant faces do not affect local XY neighborhoods.
        self.mdp_3d = Robot3DMotionMDP(
            x_max=4, y_max=3, z_max=40, grid_shape=(4, 3, 2),
            start_cell=(0, 0, 0), goal_cell=(3, 2, 0), gamma=0.83,
            movement_reward=0.01, goal_reward=1,
            obstacles=[(1, 1, 0)] + [(x, y, 1) for y in range(3) for x in range(4)],
        )

    def test_xy_transitions_rewards_terminal_values_and_uncertainty_sets_match_2d(self):
        two, three = self.mdp_2d, self.mdp_3d
        values_2d = np.random.default_rng(7).uniform(-1, 1, two.n_state)
        values_3d = np.concatenate((values_2d, np.full(two.n_state, -100)))
        states = [tuple(state) for state in two.states] + [(-1, 0), (4, 2), (1, -1)]
        for state in states:
            state_3d = (*state, 0)
            self.assertEqual(two.is_terminal(state), three.is_terminal(state_3d))
            self.assertEqual(two.values_at(state, values_2d), three.values_at(state_3d, values_3d))
            np.testing.assert_array_equal(two.coordinates(state), three.coordinates(state_3d)[:2])
            for action in range(4):
                self.assertEqual((*two.nominal_next_state(state, action), 0),
                                 three.nominal_next_state(state_3d, action))
                self.assertEqual(two.reward(state, action), three.reward(state_3d, action))
            for C in (0, 0.49, 0.5, np.sqrt(0.5), 1.2, 2):
                expected = {(*cell, 0) for cell in two.localized_uncertainty_states(state, C)}
                actual = set(map(tuple, three.localized_uncertainty_states(state_3d, C)))
                self.assertEqual(actual, expected)
                self.assertEqual(two.worst_case_value(state, C, values_2d),
                                 three.worst_case_value(state_3d, C, values_3d))
            self.assertEqual(two.worst_case_value(state, np.inf, values_2d),
                             three.worst_case_value(state_3d, np.inf, values_3d))

    def test_robust_value_iteration_reduces_to_the_2d_solution(self):
        for R, C in ((0, 0), (0.35, 0), (0.35, 0.6), (0.5, 1.1), (0.5, np.inf)):
            with self.subTest(R=R, C=C):
                two = Tabular_Agent(self.mdp_2d, lr_init=0.5, R=R, C=C)
                three = Tabular_Agent(self.mdp_3d, lr_init=0.5, R=R, C=C)
                Q_2d, _ = two.value_iteration(tolerance=1e-11)
                Q_3d, _ = three.value_iteration(tolerance=1e-11)
                np.testing.assert_allclose(Q_3d[:self.mdp_2d.n_state, :4], Q_2d, atol=1e-10)
                np.testing.assert_array_equal(Q_3d[self.mdp_3d.terminal_mask], 0)

    def test_deployment_xy_contacts_exits_scores_and_returns_match_2d(self):
        cases = (
            ((1.5, 0.5), (0, 1), 0),
            ((0.5, 0.5), (-2, 0), 1),
            ((2.5, 2.5), (0, 0), 0),
        )
        for position, deviation, action in cases:
            with self.subTest(position=position, deviation=deviation):
                two, three = self.mdp_2d.clone(), self.mdp_3d.clone()
                two.noise_sampler = lambda nominal, rng: np.asarray(deviation)
                three.noise_sampler = lambda nominal, rng: np.array((*deviation, 0))
                two.noise_radius = three.noise_radius = 3
                two.is_nominal = three.is_nominal = False
                record_2d = two.deploy_policy(lambda state: action, start_position=position)
                record_3d = three.deploy_policy(lambda state: action,
                                              start_position=(*position, 10))
                for key in ("states", "positions", "nominal_positions", "deviation_vectors"):
                    np.testing.assert_array_equal(record_2d[key], record_3d[key][:, :2])
                for key in ("actions", "rewards", "deviations", "mismatches"):
                    np.testing.assert_array_equal(record_2d[key], record_3d[key])
                for key in ("episode_length", "score", "episode_mismatches", "discounted_return",
                            "goal_reached", "collision", "workspace_exit", "timeout"):
                    self.assertEqual(record_2d[key], record_3d[key])


if __name__ == "__main__":
    unittest.main()
