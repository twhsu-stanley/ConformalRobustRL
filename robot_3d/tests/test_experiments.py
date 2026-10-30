"""Check reproducible 3-D datasets, controller runs, saved files, and figures."""

import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from q_transformer.data import format_prompt
from robot_3d.generate_robot_3d_q_data import main as generate_data
from robot_3d.generate_robot_3d_q_data import randomize_start_goal, training_maps
from robot_3d.main_robot_3d import main as train_fixed
from robot_3d.main_robot_3d_online import METHOD_SETTINGS, main as run_online
from robot_3d.plot_fixedRC_eval_returns import main as plot_returns
from robot_3d.plot_fixedRC_value_policy import main as plot_policy
from robot_3d.plot_online_comparison import plot_comparison
from robot_3d.robot_3d_motion_mdp import Robot3DMotionMDP
from robot_3d.utils import plot_robot_map, plot_robot_motion


def small_runner_mdp(**settings):
    """Use a small, fully configured task for runner storage tests."""
    return Robot3DMotionMDP(
        x_max=2, y_max=2, z_max=2, grid_shape=(2, 2, 2),
        start_cell=(0, 0, 0), goal_cell=(1, 1, 1), gamma=settings["gamma"],
        movement_reward=settings["movement_reward"], goal_reward=settings["goal_reward"],
        max_episode_steps=settings["max_episode_steps"],
    )


class DatasetTests(unittest.TestCase):
    def test_random_maps_have_ten_unit_endpoint_separation_and_complete_coordinate_prompts(self):
        template = Robot3DMotionMDP(
            x_max=10, y_max=10, z_max=10, grid_shape=(10, 10, 10),
            start_cell=(0, 0, 0), goal_cell=(9, 9, 9), gamma=0.9,
            movement_reward=0, goal_reward=1,
        )
        first = training_maps(template, 3, seed=7)
        second = training_maps(template, 3, seed=7)
        for mdp, repeated in zip(first, second):
            self.assertEqual(mdp.start_state, repeated.start_state)
            self.assertEqual(mdp.goal_state, repeated.goal_state)
            np.testing.assert_array_equal(mdp.obstacle_mask, repeated.obstacle_mask)
            distance = np.linalg.norm(mdp.coordinates(mdp.start_state) -
                                      mdp.coordinates(mdp.goal_state))
            self.assertGreaterEqual(distance, 10.0)
            self.assertTrue(20 <= mdp.obstacle_mask.sum() <= 30)
            context = mdp.q_transformer_context()
            header = "Obstacle cells are at coordinates (x, y, z):\n"
            rows = context.split(header)[1].split("\nAll other cells")[0]
            cells = [tuple(map(int, line.split())) for line in rows.splitlines()]
            expected = [tuple(map(int, cell[::-1])) for cell in np.argwhere(mdp.obstacle_mask)]
            self.assertEqual(cells, expected)
            for cell in cells:
                for endpoint in (mdp.start_state, mdp.goal_state):
                    self.assertGreaterEqual(max(abs(a - b) for a, b in zip(cell, endpoint)), 2)
            prompt = format_prompt(context, 0.2, 1.75)
            self.assertIn("Predict the Q-table for R = 0.2 and C = 1.75.", prompt)
            self.assertTrue(mdp.is_nominal)
            self.assertEqual(mdp.gamma, 0.9)
        self.assertFalse(template.obstacle_mask.any())
        self.assertEqual(template.start_state, (0, 0, 0))
        for invalid in (-1, np.inf, 100):
            with self.assertRaises(ValueError):
                randomize_start_goal(template, np.random.default_rng(7), invalid)

    def test_generator_saves_numeric_labels_from_the_full_mdp(self):
        mdp = Robot3DMotionMDP(
            x_max=2, y_max=2, z_max=2, grid_shape=(2, 2, 2),
            start_cell=(0, 0, 0), goal_cell=(1, 1, 1), gamma=0.73,
            movement_reward=0, goal_reward=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            script = str(Path(directory) / "main.py")
            with patch("robot_3d.generate_robot_3d_q_data.__file__", script):
                path = generate_data(
                    mdp, n_maps=4, n_obstacles=(0, 0), start_goal_seperation=1,
                    R_values=(0, 0.3), C_values=(0, 0.6), seed=7,
                )
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["Q"].shape, (16, 8, 6))
                self.assertTrue(np.all(data["bellman_residual"] <= 1e-8))
                for index, Q, R in zip(data["map_index"], data["Q"], data["R"]):
                    text = str(data["descriptions"][index])
                    endpoints = text.split("Start at ")[1].split(".\n")[0]
                    start_text, goal_text = endpoints.split(" and move to the goal state at ")
                    start, goal = ast.literal_eval(start_text), ast.literal_eval(goal_text)
                    goal_row = goal[0] + 2 * (goal[1] + 2 * goal[2])
                    np.testing.assert_array_equal(Q[goal_row], 0)
                    if R == 0:
                        distance = sum(abs(a - b) for a, b in zip(start, goal))
                        row = start[0] + 2 * (start[1] + 2 * start[2])
                        self.assertAlmostEqual(Q[row].max(), 0.73**(distance - 1))
            self.assertFalse(list(Path(directory).rglob("model.pt")))


class ExperimentTests(unittest.TestCase):
    def test_plotting_preserves_mdp_bounds_and_future_clones(self):
        mdp = Robot3DMotionMDP(
            x_max=10, y_max=10, z_max=10, grid_shape=(10, 10, 10),
            start_cell=(0, 0, 0), goal_cell=(9, 9, 9), gamma=0.9,
            movement_reward=0, goal_reward=1,
        )
        figure = plt.figure()
        axis = figure.add_subplot(projection="3d")
        plot_robot_map(mdp, axis)
        plot_robot_map(mdp, axis)
        np.testing.assert_array_equal(mdp.bounds, (10, 10, 10))
        np.testing.assert_array_equal(mdp.clone().bounds, (10, 10, 10))
        plt.close(figure)
        figure = plot_robot_motion(mdp, np.zeros((1000, 6)), np.zeros(1000, dtype=int))
        for axis in figure.axes[:2]:
            self.assertEqual(axis.get_xlim(), (0, 10))
            self.assertEqual(axis.get_ylim(), (0, 10))
            self.assertEqual(axis.get_zlim(), (0, 10))
        np.testing.assert_array_equal(mdp.bounds, (10, 10, 10))
        plt.close(figure)

    def test_online_methods_share_the_support_schedule_and_generate_3d_figures(self):
        mdp = Robot3DMotionMDP(
            x_max=2, y_max=2, z_max=2, grid_shape=(2, 2, 2),
            start_cell=(0, 0, 0), goal_cell=(1, 1, 1), gamma=0.9,
            movement_reward=0, goal_reward=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / "main.py")
            schedules = []
            with patch("robot_3d.main_robot_3d_online.__file__", filename):
                for method, (R, C, _) in METHOD_SETTINGS.items():
                    controller = run_online(
                        mdp, method=method, n_deployment_episodes=3, max_training_episodes=5000,
                        horizon=6, pilot_episodes=2, seed=7, shift_period=10,
                        verbose=False, save_plots=False,
                    )
                    schedules.append([row["C_star_k"] for row in controller.history])
                    name = f"results_{method}_R{R}_C{C}_trial_1.pkl"
                    with (Path(directory) / "saved_results/online" / name).open("rb") as file:
                        saved = pickle.load(file)
                    self.assertEqual(saved["Q"].shape, (8, 6))
                    for record in saved["history"]:
                        self.assertEqual(record["positions"].shape[1], 3)
                        self.assertEqual(record["states"].shape[1], 3)
                        self.assertEqual(record["score"], max(record["deviations"]))
                        self.assertTrue(record["training"]["converged"])
                    if method == "proposed":
                        self.assertEqual(controller.acp.n_updates, 3)
                    else:
                        self.assertIsNone(controller.acp)
                        self.assertTrue(controller.history[1]["training_reused"])
            self.assertTrue(all(schedule == schedules[0] for schedule in schedules))
            with patch("robot_3d.plot_online_comparison.__file__", filename):
                plot_comparison()
            output = Path(directory) / "saved_results/online"
            for name in ("policy_comparison.png", "r_estimation.png", "acp_calibration.png"):
                self.assertGreater((output / name).stat().st_size, 0)
            self.assertEqual(len(list(output.glob("trajectories_*.png"))), 4)
            self.assertFalse(plt.get_fignums())

    def test_fixed_runs_save_step_counts_curves_policies_and_slice_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("robot_3d.main_robot_3d.Robot3DMotionMDP", side_effect=small_runner_mdp):
                for R, C in ((0, 1), (0.1, 1), (0.2, 1), (0.2, 0), (0.2, 2)):
                    with redirect_stdout(io.StringIO()):
                        curves = train_fixed(
                            R=float(R), C=float(C), n_trials=1, max_training_episodes=5000,
                            horizon=10, output_dir=directory,
                        )
                    name = f"robot_3d_R{float(R)}_C{float(C)}_agents.pkl"
                    with (Path(directory) / name).open("rb") as file:
                        saved = pickle.load(file)
                    agent = saved["agents"][0]
                    self.assertIs(agent.env, saved["mdp"])
                    self.assertEqual(len(curves[0]), agent.training_diagnostics["transitions"] + 1)
                    self.assertTrue(agent.training_diagnostics["converged"])
            figures = plot_returns(data_dir=directory, show=False)
            figures.append(plot_policy(data_dir=directory))
            for figure in figures:
                plt.close(figure)
            self.assertEqual(len(list(Path(directory).glob("*.png"))), 4)

    def test_entry_points_import_outside_the_repository_without_running_experiments(self):
        scripts = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            for name in ("main_robot_3d", "main_robot_3d_online", "generate_robot_3d_q_data",
                         "train_robot_3d_transformer", "plot_fixedRC_eval_returns",
                         "plot_fixedRC_value_policy", "plot_online_comparison"):
                code = f"import runpy; runpy.run_path({str(scripts / (name + '.py'))!r})"
                result = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=directory,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(list(Path(directory).iterdir()))

    def test_zero_initialization_needs_no_transformer_imports(self):
        code = """
import sys
class BlockTransformerImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'transformers'):
            raise AssertionError('Transformer dependency imported for zero initialization')
sys.meta_path.insert(0, BlockTransformerImports())
from conformally_robust_controller import ConformallyRobustController
from robot_3d.robot_3d_motion_mdp import Robot3DMotionMDP
mdp = Robot3DMotionMDP(
    x_max=2, y_max=2, z_max=2, grid_shape=(2, 2, 2),
    start_cell=(0, 0, 0), goal_cell=(1, 1, 1), gamma=0.9,
    movement_reward=0, goal_reward=1,
)
controller = ConformallyRobustController(mdp, R=0, C=0, max_training_episodes=5000)
assert controller.run_episode()['training']['zero_initialized']
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
