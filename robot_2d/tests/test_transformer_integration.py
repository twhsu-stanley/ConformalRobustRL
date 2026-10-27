"""Check map descriptions and optional initialization at both 2-D entry points."""

from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from q_transformer.data import format_prompt
from robot_2d.generate_robot_2d_q_data import randomize_start_goal, training_maps
from robot_2d.main_robot_2d import main as train_fixed
from robot_2d.main_robot_2d_online import main as run_online
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP


class TransformerIntegrationTests(unittest.TestCase):
    def test_context_describes_grid_and_endpoints_but_excludes_unknown_disturbances(self):
        mdp = Robot2DMotionMDP()
        description = mdp.q_transformer_context()
        rows = description.split("\n\n", 1)[1].splitlines()[:mdp.ny]
        grid = np.array([[int(cell) for cell in row.split()] for row in rows])
        np.testing.assert_array_equal(grid, mdp.obstacle_mask)
        self.assertIn(f"Start at {mdp.start_state} and move to {mdp.goal_state}", description)
        mdp.noise_probability, mdp.noise_radius = 0.9, 10.0
        self.assertEqual(mdp.q_transformer_context(), description)
        mdp.start_state = (1, 0)
        moved_start = mdp.q_transformer_context()
        self.assertNotEqual(moved_start, description)
        mdp.set_obstacles([])
        updated_description = mdp.q_transformer_context()
        self.assertNotEqual(updated_description, moved_start)

    def test_training_maps_keep_template_unchanged_and_randomize_endpoints(self):
        template = Robot2DMotionMDP(
            x_max=10, y_max=10, grid_shape=(10, 10), start_cell=(1, 1), goal_cell=(7, 6),
        )
        maps = list(training_maps(template, 3, n_obstacles=(10, 30), seed=7))
        self.assertEqual(np.count_nonzero(template.obstacle_mask), 0)
        self.assertEqual(template.start_state, (1, 1))
        self.assertEqual(template.goal_state, (7, 6))
        self.assertGreater(len({(mdp.start_state, mdp.goal_state) for mdp in maps}), 1)
        for mdp in maps:
            np.testing.assert_array_equal(mdp.state_coordinates, template.state_coordinates)
            self.assertTrue(mdp.is_nominal)
            self.assertTrue(10 <= np.count_nonzero(mdp.obstacle_mask) <= 30)
            distance = np.linalg.norm(mdp.coordinates(mdp.start_state) -
                                      mdp.coordinates(mdp.goal_state))
            self.assertGreaterEqual(distance, 7.5)
            self.assertEqual(mdp.s, mdp.start_state)
            np.testing.assert_array_equal(mdp.position, mdp.coordinates(mdp.start_state))
            self.assertFalse(mdp.is_terminal(mdp.start_state))
            self.assertTrue(mdp.is_terminal(mdp.goal_state))
            description = mdp.q_transformer_context()
            self.assertIn(f"Start at {mdp.start_state} and move to {mdp.goal_state}", description)
            for iy, ix in np.argwhere(mdp.obstacle_mask):
                for sx, sy in (mdp.start_state, mdp.goal_state):
                    self.assertGreaterEqual(max(abs(ix - sx), abs(iy - sy)), 2)
            for action, (dx, dy) in enumerate(mdp.ACTION_DIRECTIONS):
                source = (mdp.goal_state[0] - dx, mdp.goal_state[1] - dy)
                if not mdp.is_terminal(source):
                    self.assertEqual(mdp.reward(source, action), 1.0)

    def test_prompt_keeps_original_instruction_style_and_numeric_grid_orientation(self):
        mdp = Robot2DMotionMDP(
            x_max=3, y_max=2, grid_shape=(3, 2), obstacles=[(1, 0)], goal_cell=(2, 1),
        )
        prompt = format_prompt(mdp.q_transformer_context(), 0.3, np.inf)
        self.assertTrue(prompt.startswith("### Instruction:\n"))
        self.assertIn("size 3x2", prompt)
        self.assertIn("0 = free space, 1 = obstacle.\n\n0 1 0\n0 0 0", prompt)
        self.assertIn("Start at (0, 0) and move to (2, 1)", prompt)
        self.assertIn("R = 0.3 and C = infinity", prompt)
        self.assertIn("Output the Q Table only", prompt)
        self.assertTrue(prompt.endswith("### Response:\n"))

    def test_random_endpoints_use_physical_distance_and_repeat_with_the_seed(self):
        mdp = Robot2DMotionMDP(x_max=6, y_max=3, grid_shape=(3, 3), obstacles=[])
        first_rng, second_rng = np.random.default_rng(7), np.random.default_rng(7)
        for _ in range(20):
            start, goal = randomize_start_goal(mdp, first_rng, start_goal_seperation=4.25)
            self.assertEqual((start, goal), randomize_start_goal(mdp, second_rng, 4.25))
            self.assertGreaterEqual(np.linalg.norm(mdp.coordinates(start) -
                                                  mdp.coordinates(goal)), 4.25)
            self.assertLess(np.linalg.norm(np.asarray(start) - goal), 4.25)
        start, goal = randomize_start_goal(mdp, first_rng, start_goal_seperation=0)
        self.assertNotEqual(start, goal)
        for invalid in (-1, np.nan, np.inf, 5):
            with self.subTest(separation=invalid):
                with self.assertRaisesRegex(ValueError, "start_goal_seperation"):
                    randomize_start_goal(mdp, first_rng, invalid)

    def test_online_and_fixed_runs_accept_initializers_without_serializing_them(self):
        calls = []

        def initializer(mdp, R_used, C_used):
            calls.append((R_used, C_used))
            return np.full((mdp.n_state, mdp.action_space.n), 0.5)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with patch("robot_2d.main_robot_2d_online.__file__", str(directory / "main.py")):
                controller = run_online(
                    Robot2DMotionMDP(), n_deployment_episodes=2, training_episodes=3,
                    method="nominal", horizon=2, save_plots=False, verbose=False,
                    q_initializer=initializer,
                )
            self.assertEqual(calls, [(0.0, 0.0)])
            self.assertFalse(controller.history[0]["training"]["zero_initialized"])
            path = next((directory / "saved_results/online").glob("results_*.pkl"))
            with path.open("rb") as file:
                saved = pickle.load(file)
            self.assertNotIn("q_initializer", saved)
            calls.clear()
            train_fixed(
                training_episodes=2, n_trials=1, horizon=2, output_dir=directory,
                q_initializer=initializer,
            )
            self.assertEqual(calls, [(0.15, 1.2)])
            path = next(directory.glob("*_agents.pkl"))
            with path.open("rb") as file:
                agent = pickle.load(file)["agents"][0]
            self.assertFalse(hasattr(agent, "q_initializer"))
            self.assertFalse(agent.training_diagnostics["zero_initialized"])


if __name__ == "__main__":
    unittest.main()
