"""Behavioral checks for the robot model and localized robust Q-learning."""

import unittest

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from robot_2d_motion_mdp import Robot2DMotionMDP
from Tabular_Agent import Tabular_Agent


def reference_q(mdp, gamma, R, C):
    """Synchronous robust value iteration, independent of the TD training loop."""
    values = np.zeros(mdp.n_state)
    for _ in range(2000):
        Q = np.zeros((mdp.n_state, 4))
        for state in mdp.trainable_states:
            for action in range(4):
                nominal = mdp.nominal_next_state(state, action)
                candidates = mdp.localized_uncertainty_states(state, action, C)
                worst_value = np.min(values[candidates])
                continuation = (1 - R) * values[nominal] + R * worst_value
                Q[state, action] = mdp.reward(state, action) + gamma * continuation
        updated = np.max(Q, axis=1)
        if np.max(np.abs(updated - values)) < 1e-12:
            return Q
        values = updated
    raise AssertionError("Reference value iteration did not converge.")


class RobotMotionModelTests(unittest.TestCase):
    def test_gymnasium_interface(self):
        check_env(Robot2DMotionMDP(seed=3), skip_render_check=True)

    def test_coordinates_quantization_and_bounds(self):
        mdp = Robot2DMotionMDP(x_max=16, y_max=8)
        np.testing.assert_allclose(mdp.coordinates(0), (1, 0.5))
        np.testing.assert_allclose(mdp.coordinates(63), (15, 7.5))
        for state in range(64):
            self.assertEqual(mdp.discretize(mdp.coordinates(state)), state)
        self.assertEqual(mdp.discretize((16, 8)), 63)
        with self.assertRaises(ValueError):
            mdp.discretize((-0.01, 0))

    def test_direction_commands_snap_to_centers(self):
        mdp = Robot2DMotionMDP()
        state = mdp.cell_to_state((3, 2))
        expected_cells = ((4, 2), (2, 2), (3, 3), (3, 1))
        for action, expected in enumerate(expected_cells):
            self.assertEqual(mdp.nominal_next_state(state, action), mdp.cell_to_state(expected))
        mdp.reset(options={"position": (3.2, 2.7)})
        successor, _, _, _, info = mdp.step(0)
        np.testing.assert_allclose(info["position"], mdp.coordinates(successor))

    def test_collision_goal_and_boundary_rewards(self):
        mdp = Robot2DMotionMDP(obstacles=[(1, 0)])
        successor, reward, terminated, _, info = mdp.step(0)
        self.assertEqual(successor, 1)
        self.assertEqual(reward, 0)
        self.assertTrue(terminated and info["collision"])
        mdp.reset()
        successor, reward, terminated, _, info = mdp.step(1)
        self.assertEqual(successor, mdp.start_state)
        self.assertEqual(reward, 0)
        self.assertFalse(terminated)
        self.assertTrue(info["boundary_attempt"])
        mdp.reset(options={"state": mdp.cell_to_state((6, 7))})
        successor, reward, terminated, _, info = mdp.step(0)
        self.assertEqual(successor, mdp.goal_state)
        self.assertEqual(reward, 1)
        self.assertTrue(terminated and info["goal_reached"])

    def test_rewards_stay_state_action_based_under_noise(self):
        mdp = Robot2DMotionMDP(
            obstacles=[(1, 1)], noise_sampler=lambda position, rng: np.array((0.0, 1.0)),
        )
        expected_reward = mdp.reward(mdp.start_state, 0)
        _, reward, terminated, _, info = mdp.step(0)
        self.assertTrue(terminated and info["collision"])
        self.assertEqual(reward, expected_reward)

    def test_subcell_deviation_is_not_a_discrete_mismatch(self):
        mdp = Robot2DMotionMDP(noise_sampler=lambda position, rng: np.array((0.2, 0.0)))
        successor, _, _, _, info = mdp.step(0)
        self.assertEqual(successor, mdp.nominal_next_state(mdp.start_state, 0))
        self.assertFalse(info["mismatch"])
        self.assertAlmostEqual(info["deviation_norm"], 0.2)

    def test_projection_reports_effective_deviation_and_violation(self):
        mdp = Robot2DMotionMDP(
            noise_sampler=lambda position, rng: np.array((10.0, 0.0)), noise_radius=10,
        )
        _, _, _, _, info = mdp.step(1)
        np.testing.assert_allclose(info["position"], (8, 0.5))
        self.assertTrue(info["boundary_violation"])
        self.assertAlmostEqual(info["deviation_norm"], 7.5)

    def test_cell_intersection_includes_cells_with_distant_centers(self):
        mdp = Robot2DMotionMDP()
        center = mdp.cell_to_state((3, 3))
        self.assertEqual(set(mdp.localized_states_around(center, 0.6)), {19, 26, 27, 28, 35})
        self.assertEqual(mdp.localized_states_around(center, 0).tolist(), [center])
        self.assertEqual(len(mdp.localized_states_around(center, np.inf)), 64)
        mdp = Robot2DMotionMDP(x_max=4, y_max=2, grid_shape=(2, 2))
        self.assertEqual(mdp.localized_states_around(0, 0.6).tolist(), [0, 2])

    def test_fixed_horizon_executes_absorbing_transitions(self):
        mdp = Robot2DMotionMDP(
            grid_shape=(2, 2), x_max=2, y_max=2, max_episode_steps=4,
            episode_mode="fixed_horizon",
        )
        mdp.step(0)
        _, reward, terminated, truncated, _ = mdp.step(2)
        self.assertEqual(reward, 1)
        self.assertFalse(terminated or truncated)
        _, reward, _, _, info = mdp.step(1)
        self.assertEqual(reward, 0)
        self.assertTrue(info["absorbing_transition"])
        _, _, _, truncated, _ = mdp.step(1)
        self.assertTrue(truncated)
        with self.assertRaises(RuntimeError):
            mdp.step(0)

    def test_clone_is_independent_and_nominal(self):
        mdp = Robot2DMotionMDP(noise_probability=0.7)
        clone = mdp.clone(nominal=True)
        self.assertTrue(clone.is_nominal)
        clone.step(0)
        self.assertEqual(mdp.s, mdp.start_state)
        np.testing.assert_array_equal(clone.nominal_transitions, mdp.nominal_transitions)
        with self.assertRaises(ValueError):
            mdp.clone(max_episode_steps=0)

    def test_invalid_maps_parameters_and_noise_are_rejected(self):
        with self.assertRaises(ValueError):
            Robot2DMotionMDP(obstacles=[(0, 0)])
        with self.assertRaises(ValueError):
            Robot2DMotionMDP(grid_shape=(2, 2), obstacles=[(1, 0), (0, 1)])
        with self.assertRaises(ValueError):
            Robot2DMotionMDP(noise_probability=1.5)
        mdp = Robot2DMotionMDP(noise_sampler=lambda position, rng: np.array((5, 0)))
        with self.assertRaises(ValueError):
            mdp.step(0)

    def test_seeded_noise_is_reproducible(self):
        left = Robot2DMotionMDP(noise_probability=1, noise_radius=0.3)
        right = left.clone()
        left.reset(seed=42)
        right.reset(seed=42)
        for action in (0, 0, 0, 0):
            left_step = left.step(action)
            right_step = right.step(action)
            np.testing.assert_allclose(left_step[4]["position"], right_step[4]["position"])


class RobustQLearningTests(unittest.TestCase):
    def small_mdp(self):
        return Robot2DMotionMDP(x_max=2, y_max=2, grid_shape=(2, 2), movement_reward=0)

    def test_robust_target_and_limiting_cases(self):
        mdp = self.small_mdp()
        agent = Tabular_Agent(mdp, 0.9, 0.5, R=0.25, C=0.6)
        values = np.array((0.4, 0.8, 0.2, 0.0))
        self.assertAlmostEqual(agent.robust_td_target(0.1, 1, values), 0.1 + 0.9 * 0.75 * 0.8)
        self.assertAlmostEqual(agent.robust_td_target(0.1, 1, values, R=0), 0.82)
        self.assertAlmostEqual(agent.robust_td_target(0.1, 1, values, C=0), 0.82)
        self.assertAlmostEqual(agent.robust_td_target(0.1, 1, values, R=1, C=np.inf), 0.1)

    def test_training_matches_reference_and_resets_initialization(self):
        mdp = self.small_mdp()
        agent = Tabular_Agent(mdp, 0.9, 0.5, R=0.2, C=0.6, seed=2)
        agent.Q.fill(99)
        Q, policy = agent.Robust_Q_learning(2000, record_every=100)
        oracle = reference_q(mdp, 0.9, 0.2, 0.6)
        np.testing.assert_allclose(Q, oracle, atol=1e-6)
        self.assertAlmostEqual(np.max(Q[0]), 0.72, places=6)
        self.assertEqual(agent.evaluation_return[0], 0)
        self.assertEqual(agent.training_diagnostics["unvisited_pairs"], 0)
        self.assertLess(agent.training_diagnostics["bellman_residual"], 1e-6)
        self.assertEqual(policy.shape, (4,))
        np.testing.assert_array_equal(Q[mdp.terminal_mask], 0)
        Q.fill(-10)
        self.assertTrue(np.all(agent.Q >= 0))

    def test_R_zero_and_C_zero_recover_nominal_learning(self):
        mdp = self.small_mdp()
        standard = Tabular_Agent(mdp.clone(nominal=True), 0.9, 0.5, R=0, seed=4)
        radius_zero = Tabular_Agent(mdp.clone(nominal=True), 0.9, 0.5, R=0.8, C=0, seed=4)
        left, _ = standard.Robust_Q_learning(300)
        right, _ = radius_zero.Robust_Q_learning(300)
        np.testing.assert_allclose(left, right, atol=1e-12)

    def test_training_rejects_uncertain_simulator(self):
        mdp = Robot2DMotionMDP(noise_probability=0.1)
        with self.assertRaises(ValueError):
            Tabular_Agent(mdp, 0.9, 0.5).Robust_Q_learning(1)

    def test_exploring_starts_cover_every_nonterminal_pair_in_one_sweep(self):
        mdp = Robot2DMotionMDP(max_episode_steps=2)
        n_pairs = len(mdp.trainable_states) * 4
        agent = Tabular_Agent(mdp, 0.95, 0.5, seed=9)
        agent.Robust_Q_learning(n_pairs, record_every=100)
        self.assertTrue(np.all(agent.visit_counts[~mdp.terminal_mask] >= 1))

    def test_frozenlake_fixed_parameter_api_remains_usable(self):
        env = gym.make("FrozenLake-v1", is_slippery=False, max_episode_steps=20)
        agent = Tabular_Agent(env, 0.95, 0.5, R=0.1, C=1, seed=1)
        Q, policy = agent.Robust_Q_learning(5)
        self.assertEqual(Q.shape, (16, 4))
        self.assertEqual(policy.shape, (16,))
        env.close()


if __name__ == "__main__":
    unittest.main()
