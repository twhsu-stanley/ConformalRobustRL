"""General robust tabular Q-learning checks using small nominal MDP fixtures."""

import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP
from Tabular_Agent import Tabular_Agent
from utils_tabular import calc_evaluation_return_mean_std


def reference_q(mdp, R, C):
    """Synchronous robust value iteration, independent of the TD training loop."""
    values = np.zeros(mdp.n_state)
    for _ in range(2000):
        Q = np.zeros((mdp.n_state, 4))
        for state in mdp.trainable_states:
            row = mdp.state_index(state)
            for action in range(4):
                nominal = mdp.nominal_next_state(state, action)
                worst_value = mdp.worst_case_value(nominal, C, values)
                continuation = (1 - R) * mdp.values_at(nominal, values) + R * worst_value
                Q[row, action] = mdp.reward(state, action) + mdp.gamma * continuation
        updated = np.max(Q, axis=1)
        if np.max(np.abs(updated - values)) < 1e-12:
            return Q
        values = updated
    raise AssertionError("Reference value iteration did not converge.")


class RobustQLearningTests(unittest.TestCase):
    def small_mdp(self):
        return Robot2DMotionMDP(
            x_max=2, y_max=2, grid_shape=(2, 2), start_cell=(0, 0), goal_cell=(1, 1), gamma=0.9,
            movement_reward=0, goal_reward=1.0,
        )

    def test_solvers_and_deployment_use_the_MDP_discount(self):
        mdp = Robot2DMotionMDP(
            x_max=2, y_max=2, grid_shape=(2, 2), start_cell=(0, 0), goal_cell=(1, 1), gamma=0.37,
            movement_reward=0.0, goal_reward=1.0,
        )
        agent = Tabular_Agent(mdp, lr_init=0.5, seed=2)
        expected, policy = agent.value_iteration()
        self.assertAlmostEqual(expected[0].max(), 0.37)
        learned, _ = agent.robust_q_learning(2000)
        np.testing.assert_allclose(learned, expected, atol=1e-5)
        for nominal in (False, True):
            clone = mdp.clone(nominal=nominal)
            self.assertEqual(clone.gamma, 0.37)
            record = clone.deploy_policy(policy)
            self.assertEqual(record["episode_length"], 2)
            self.assertAlmostEqual(record["discounted_return"], 0.37)

    def test_robust_target_and_limiting_cases(self):
        mdp = self.small_mdp()
        agent = Tabular_Agent(mdp, 0.5, R=0.25, C=0.6)
        values = np.array((0.4, 0.8, 0.2, 0.0))
        self.assertAlmostEqual(agent.robust_td_target(0.1, (1, 0), values), 0.1 + 0.9 * 0.75 * 0.8)
        self.assertAlmostEqual(agent.robust_td_target(0.1, (1, 0), values, R=0), 0.82)
        self.assertAlmostEqual(agent.robust_td_target(0.1, (1, 0), values, C=0), 0.82)
        self.assertAlmostEqual(agent.robust_td_target(0.1, (1, 0), values, R=1, C=np.inf), 0.1)

    def test_training_matches_reference_and_resets_initialization(self):
        mdp = self.small_mdp()
        agent = Tabular_Agent(mdp, 0.5, R=0.2, C=0.6, seed=2)
        agent.Q.fill(99)
        Q, policy = agent.robust_q_learning(
            max_training_episodes=2000, record_every=100, tolerance=1e-8,
        )
        oracle = reference_q(mdp, 0.2, 0.6)
        np.testing.assert_allclose(Q, oracle, atol=1e-6)
        self.assertAlmostEqual(np.max(Q[0]), 0.72, places=6)
        self.assertEqual(agent.evaluation_return[0], 0)
        self.assertEqual(agent.training_diagnostics["unvisited_pairs"], 0)
        self.assertLess(agent.training_diagnostics["bellman_residual"], 1e-6)
        self.assertEqual(policy.shape, (mdp.n_state,))
        np.testing.assert_array_equal(Q[mdp.terminal_mask], 0)
        Q.fill(-10)
        self.assertTrue(np.all(agent.Q >= 0))

    def test_R_zero_and_C_zero_recover_nominal_learning(self):
        mdp = self.small_mdp()
        standard = Tabular_Agent(mdp.clone(nominal=True), 0.5, R=0, seed=4)
        radius_zero = Tabular_Agent(mdp.clone(nominal=True), 0.5, R=0.8, C=0, seed=4)
        left, _ = standard.robust_q_learning(300)
        right, _ = radius_zero.robust_q_learning(300)
        np.testing.assert_allclose(left, right, atol=1e-12)

    def test_training_rejects_uncertain_simulator(self):
        mdp = Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0, noise_probability=0.1,
        )
        with self.assertRaises(ValueError):
            Tabular_Agent(mdp, 0.5).robust_q_learning(1)

    def test_exploring_starts_cover_every_nonterminal_pair_in_one_sweep(self):
        mdp = Robot2DMotionMDP(
            x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0, max_episode_steps=2,
        )
        n_pairs = len(mdp.trainable_states) * 4
        agent = Tabular_Agent(mdp, 0.5, seed=9)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            agent.robust_q_learning(n_pairs, record_every=100)
        self.assertTrue(np.all(agent.visit_counts[~mdp.terminal_mask] >= 1))

    def test_agent_uses_the_MDP_uncertainty_set(self):
        mdp = self.small_mdp()
        agent = Tabular_Agent(mdp, 0.5, R=0.25, C=0.6)
        values = np.array((0.4, 0.8, 0.2, 0))
        with patch.object(
            mdp, "localized_uncertainty_states", return_value=np.array([(0, 1)]),
        ) as states:
            target = agent.robust_td_target(0.1, (1, 0), values)
        states.assert_called_once_with((1, 0), 0.6)
        self.assertAlmostEqual(target, 0.1 + 0.9 * (0.75 * 0.8 + 0.25 * 0.2))

    def test_custom_initialization_keeps_terminal_values_zero(self):
        class WarmStartAgent(Tabular_Agent):
            def initialize_q(self):
                self.Q.fill(2.0)

        mdp = Robot2DMotionMDP(
            x_max=2, y_max=2, grid_shape=(2, 2), start_cell=(0, 0), goal_cell=(1, 1), gamma=0.9,
            movement_reward=0, goal_reward=1.0, max_episode_steps=1,
        )
        agent = WarmStartAgent(mdp, 0.5, epsilon_init=0, epsilon_lb=0, seed=1)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            agent.robust_q_learning(1, exploring_starts=False)
        np.testing.assert_array_equal(agent.Q[mdp.terminal_mask], 0)
        np.testing.assert_array_equal(agent.visit_counts[mdp.terminal_mask], 0)
        np.testing.assert_allclose(agent.Q[agent.visit_counts > 0], 1.9)
        self.assertEqual(agent.evaluation_return[0], 2.0)
        self.assertFalse(agent.training_diagnostics["zero_initialized"])
        self.assertEqual(mdp.values_at((-1, 0), agent.Q.max(axis=1)), 0)

    def test_environment_must_supply_the_MDP_definitions(self):
        env = gym.make("FrozenLake-v1", is_slippery=False, max_episode_steps=20)
        try:
            with self.assertRaisesRegex(ValueError, "The MDP must define reward"):
                Tabular_Agent(env, 0.5, R=0.1, C=1, seed=1)
        finally:
            env.close()

    def test_supplied_initial_table_is_copied_and_terminal_rows_stay_zero(self):
        mdp = Robot2DMotionMDP(
            x_max=2, y_max=2, grid_shape=(2, 2), start_cell=(0, 0), goal_cell=(1, 1), gamma=0.9,
            movement_reward=0, goal_reward=1.0, max_episode_steps=1,
        )
        initial_Q = np.full((mdp.n_state, 4), 2.0)
        agent = Tabular_Agent(mdp, 0.5, epsilon_init=0, epsilon_lb=0, seed=1)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            agent.robust_q_learning(1, initial_Q=initial_Q, exploring_starts=False)
        np.testing.assert_array_equal(initial_Q, 2.0)
        self.assertEqual(agent.evaluation_return[0], 2.0)
        np.testing.assert_array_equal(agent.Q[mdp.terminal_mask], 0)
        self.assertFalse(agent.training_diagnostics["zero_initialized"])
        for invalid in (np.zeros((2, 4)), np.full(agent.Q.shape, np.nan)):
            with self.assertRaisesRegex(ValueError, "initial_Q"):
                agent.robust_q_learning(1, initial_Q=invalid)

    def test_value_iteration_matches_reference_and_rejects_unconverged_labels(self):
        mdp = self.small_mdp()
        agent = Tabular_Agent(mdp, 0.5)
        for R, C in ((0, 0), (0.25, 0.6), (0.9, np.inf)):
            with self.subTest(R=R, C=C):
                Q, policy = agent.value_iteration(R=R, C=C, tolerance=1e-10)
                np.testing.assert_allclose(Q, reference_q(mdp, R, C), atol=1e-9)
                np.testing.assert_array_equal(Q[mdp.terminal_mask], 0)
                np.testing.assert_array_equal(policy, Q.argmax(axis=1))
                self.assertLessEqual(agent.value_iteration_diagnostics["bellman_residual"], 1e-10)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            agent.value_iteration(R=0, C=0, max_iterations=1)
        mdp.gamma = 0.0
        one_step = Tabular_Agent(mdp, 0.5)
        one_step.value_iteration(max_iterations=1)
        self.assertEqual(one_step.value_iteration_diagnostics["bellman_residual"], 0)

    def test_q_learning_stops_when_the_full_residual_meets_tolerance(self):
        mdp = self.small_mdp()
        agent = Tabular_Agent(mdp, 0.5, R=0.25, C=0.6, seed=2)
        Q, _ = agent.robust_q_learning(2000, tolerance=1e-6, check_every=25, record_every=37)
        diagnostics = agent.training_diagnostics
        self.assertTrue(diagnostics["converged"])
        self.assertLess(diagnostics["episodes"], 2000)
        self.assertEqual(diagnostics["episodes"] % 25, 0)
        self.assertLessEqual(agent.robust_bellman_residual(), diagnostics["tolerance"])
        self.assertEqual(diagnostics["transitions"], agent.visit_counts.sum())
        self.assertEqual(agent.evaluation_return[-1], mdp.values_at(mdp.start_state, Q.max(axis=1)))
        np.testing.assert_allclose(Q, reference_q(mdp, 0.25, 0.6), atol=1e-5)

    def test_converged_initialization_needs_no_transitions(self):
        mdp = self.small_mdp()
        expected = reference_q(mdp, 0.25, 0.6)
        initial_Q = expected.copy()
        initial_Q[mdp.terminal_mask] = 99
        agent = Tabular_Agent(mdp, 0.5, R=0.25, C=0.6)
        with patch.object(mdp, "step", side_effect=AssertionError("No TD updates needed")):
            Q, _ = agent.robust_q_learning(1, initial_Q=initial_Q)
        np.testing.assert_allclose(Q, expected)
        np.testing.assert_array_equal(initial_Q[mdp.terminal_mask], 99)
        np.testing.assert_array_equal(agent.visit_counts, 0)
        self.assertEqual(agent.training_diagnostics["episodes"], 0)
        self.assertEqual(agent.training_diagnostics["transitions"], 0)
        self.assertTrue(agent.training_diagnostics["converged"])
        self.assertEqual(len(agent.evaluation_return), 1)

    def test_small_updates_and_unvisited_pairs_cannot_falsely_converge(self):
        for exploring_starts in (False, True):
            with self.subTest(exploring_starts=exploring_starts):
                mdp = self.small_mdp()
                mdp.max_episode_steps = 1
                agent = Tabular_Agent(mdp, 1e-10, seed=2)
                cap = len(mdp.trainable_states) * mdp.action_space.n
                with self.assertRaisesRegex(RuntimeError, "Bellman residual"):
                    agent.robust_q_learning(cap, exploring_starts=exploring_starts, check_every=1)
                self.assertFalse(agent.training_diagnostics["converged"])
                self.assertEqual(agent.training_diagnostics["episodes"], cap)
                self.assertGreater(agent.training_diagnostics["bellman_residual"], 0.9)
                self.assertLess(np.max(np.abs(agent.Q)), 1e-6)

    def test_nonfinite_values_cannot_pass_the_residual_check(self):
        agent = Tabular_Agent(self.small_mdp(), 0.5)
        for value in (np.nan, np.inf, -np.inf):
            agent.Q.fill(value)
            self.assertEqual(agent.robust_bellman_residual(), np.inf)

    def test_failed_value_iteration_replaces_previous_success_diagnostics(self):
        agent = Tabular_Agent(self.small_mdp(), 0.5)
        agent.value_iteration()
        self.assertTrue(agent.value_iteration_diagnostics["converged"])
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            agent.value_iteration(max_iterations=1)
        self.assertFalse(agent.value_iteration_diagnostics["converged"])
        self.assertEqual(agent.value_iteration_diagnostics["iterations"], 1)

    def test_invalid_convergence_settings_are_rejected(self):
        agent = Tabular_Agent(self.small_mdp(), 0.5)
        for settings in ({"tolerance": 0}, {"tolerance": np.nan}, {"check_every": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                agent.robust_q_learning(10, **settings)


class LearningCurveTests(unittest.TestCase):
    def test_stopped_curves_retain_their_final_value_and_longer_runs_are_kept(self):
        curves = [[0, 1], [0, 0.5, 0.8, 1.2], [2]]
        mean, std, samples = calc_evaluation_return_mean_std(curves)
        expected = np.array([[0, 1, 1, 1], [0, 0.5, 0.8, 1.2], [2, 2, 2, 2]])
        np.testing.assert_array_equal(samples, expected)
        np.testing.assert_allclose(mean, expected.mean(axis=0))
        np.testing.assert_allclose(std, expected.std(axis=0))
        self.assertEqual(curves, [[0, 1], [0, 0.5, 0.8, 1.2], [2]])

    def test_empty_learning_curves_are_rejected(self):
        for curves in ([], [[]], [[0], []]):
            with self.subTest(curves=curves), self.assertRaises(ValueError):
                calc_evaluation_return_mean_std(curves)

if __name__ == "__main__":
    unittest.main()

