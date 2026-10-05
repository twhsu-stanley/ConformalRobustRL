"""Behavioral checks for the 2-D robot MDP and robot experiment runners."""

from contextlib import ExitStack, redirect_stdout
import io
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.utils.env_checker import check_env

from robot_2d.main_robot_2d import main as single_main
from robot_2d.main_robot_2d_online import main as online_main
from robot_2d.plot_eval_returns import main as plot_saved_returns
from robot_2d.plot_value_policy import main as plot_saved_policy
from robot_2d.utils import plot_robot_2d_tabular
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP
from Tabular_Agent import Tabular_Agent


class ScriptImportTests(unittest.TestCase):
    def test_direct_script_imports_work_outside_the_repository(self):
        script_directory = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="robot_script_imports_") as directory:
            for name in (
                "main_robot_2d.py", "main_robot_2d_online.py", "plot_eval_returns.py",
                "plot_value_policy.py",
            ):
                with self.subTest(script=name):
                    script = script_directory / name
                    code = f"import runpy; runpy.run_path({str(script)!r})"
                    result = subprocess.run(
                        [sys.executable, "-I", "-B", "-c", code], cwd=directory,
                        capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)


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


class SingleRunnerTests(unittest.TestCase):
    def test_runner_saves_curves_and_reloadable_MDP_agents_for_every_trial(self):
        with tempfile.TemporaryDirectory(prefix="robot_motion_single_") as directory:
            root = Path(directory)
            saved_dir = root / "saved_results"
            previous_directory = Path.cwd()
            try:
                os.chdir(root)
                self.assertFalse(saved_dir.exists())
                for R, C, n_trials in ((0.15, 1.2, 1), (0.2, 1.3, 2)):
                    with self.subTest(R=R, C=C, n_trials=n_trials):
                        with redirect_stdout(io.StringIO()) as output:
                            curves = single_main(
                                R=R, C=C, training_episodes=20, n_trials=n_trials, horizon=5,
                                output_dir=saved_dir,
                            )
                        filename = saved_dir / f"robot_2d_R{R}_C{C}.pkl"
                        with filename.open("rb") as file:
                            saved_curves = pickle.load(file)
                        self.assertEqual(saved_curves, curves)
                        self.assertEqual(len(saved_curves), n_trials)
                        self.assertIn(str(filename), output.getvalue())
                        self.assertTrue(all(curve[0] == 0 for curve in saved_curves))
                        self.assertTrue(all(len(curve) > 1 for curve in saved_curves))
                        filename = saved_dir / f"robot_2d_R{R}_C{C}_agents.pkl"
                        with filename.open("rb") as file:
                            saved = pickle.load(file)
                        self.assertEqual(len(saved["agents"]), n_trials)
                        for agent, curve in zip(saved["agents"], saved_curves):
                            self.assertIs(agent.env, saved["mdp"])
                            self.assertEqual(agent.R, R)
                            self.assertEqual(agent.C, C)
                            start_value = np.max(agent.Q[saved["mdp"].start_state])
                            self.assertEqual(start_value, curve[-1])
                            self.assertEqual(len(curve), agent.visit_counts.sum() + 1)
                        with redirect_stdout(io.StringIO()):
                            fig = plot_saved_policy(
                                R=R, C=C, trial=n_trials - 1, data_dir=saved_dir, show=False,
                            )
                        plot_name = f"robot_2d_R{R}_C{C}_trial{n_trials}_value_policy.png"
                        self.assertTrue((saved_dir / plot_name).is_file())
                        plt.close(fig)
                self.assertEqual(len(list(root.rglob("*.pkl"))), 4)
                self.assertFalse(list(root.glob("*.pkl")))
                self.assertFalse(list(root.rglob("*.npz")))
                self.assertFalse(list(root.rglob("*.json")))
            finally:
                os.chdir(previous_directory)


class ResultsDirectoryTests(unittest.TestCase):
    def test_default_results_paths_follow_script_locations_from_an_unrelated_directory(self):
        with tempfile.TemporaryDirectory(prefix="robot_results_paths_") as directory:
            root = Path(directory)
            script_directory = root / "robot_2d"
            saved_dir = script_directory / "saved_results"
            working_directory = root / "unrelated"
            working_directory.mkdir()
            previous_directory = Path.cwd()
            try:
                os.chdir(working_directory)
                with ExitStack() as stack:
                    for name in (
                        "main_robot_2d", "main_robot_2d_online", "plot_eval_returns",
                        "plot_value_policy",
                    ):
                        filename = str(script_directory / f"{name}.py")
                        stack.enter_context(patch(f"robot_2d.{name}.__file__", filename))
                    with redirect_stdout(io.StringIO()):
                        for R, C in ((0.0, 1.0), (0.1, 1.0), (0.2, 1.0), (0.2, 0.0), (0.2, 1.5)):
                            single_main(R=R, C=C, training_episodes=10, horizon=5)
                        figures = plot_saved_returns(show=False)
                        figures.append(plot_saved_policy(show=False))
                        for figure in figures:
                            plt.close(figure)
                        online_main(
                            n_episodes=1, training_episodes=10, horizon=5, pilot_episodes=2,
                        )
                files = [path for path in root.rglob("*") if path.is_file()]
                self.assertEqual(len(files), 17)
                self.assertTrue(all(path.is_relative_to(saved_dir) for path in files))
                self.assertFalse(list(working_directory.iterdir()))
                self.assertTrue(all(path.stat().st_size > 0 for path in files))
                self.assertTrue((saved_dir / "online" / "results.pkl").is_file())
            finally:
                os.chdir(previous_directory)


class ValuePolicyPlotTests(unittest.TestCase):
    def test_plot_uses_physical_coordinates_values_and_nonterminal_policy_arrows(self):
        from matplotlib.patches import FancyArrow
        mdp = Robot2DMotionMDP(x_max=4, y_max=2, grid_shape=(2, 2))
        agent = Tabular_Agent(mdp, gamma=0.95, lr_init=0.5, R=0.2, C=0.6)
        agent.Q = np.array(((0.1, 0.4, 0.2, 0.3), (0.7, 0.3, 0.2, 0.1),
                            (0.2, 0.3, 0.9, 0.4), (0, 0, 0, 0)))
        fig = plot_robot_2d_tabular(mdp, agent, show=False)
        try:
            values_axis, policy_axis = fig.axes[:2]
            np.testing.assert_allclose(values_axis.images[0].get_array(), ((0.4, 0.7), (0.9, 0)))
            self.assertEqual(tuple(values_axis.images[0].get_extent()), (0, 4, 0, 2))
            self.assertEqual(policy_axis.get_xlim(), (0, 4))
            self.assertEqual(policy_axis.get_ylim(), (0, 2))
            arrows = [patch for patch in policy_axis.patches if isinstance(patch, FancyArrow)]
            self.assertEqual(len(arrows), 3)
        finally:
            plt.close(fig)


class OnlineRunnerTests(unittest.TestCase):
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

