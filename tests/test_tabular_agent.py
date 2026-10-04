"""General robust tabular Q-learning checks using small nominal MDP fixtures."""

import unittest

import gymnasium as gym
import numpy as np

from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP
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

