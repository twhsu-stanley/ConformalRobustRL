"""Check saved-data trajectory plots across deployment episodes, trials, and methods."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

from robot_2d.main_robot_2d_online import main
from robot_2d.plot_online_comparison import plot_comparison, plot_trajectories
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP


class OnlineTrajectoryPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_all_trajectories_preserve_continuous_positions_and_terminal_landings(self):
        mdp = Robot2DMotionMDP(x_max=4, y_max=2, grid_shape=(2, 2), obstacles=[(1, 0)])
        paths = [np.array([(1, 0.5), (1.1, 1.2), (3.2, 1.7)]),
                 np.array([(1, 0.5), (3.1, 0.6)]), np.array([(1, 0.5), (-1.2, 0.7)])]
        history = [{"positions": paths[k % len(paths)]} for k in range(100)]
        run = {"mdp": mdp, "history": history, "method": "proposed", "trial": 4}
        fig = plot_trajectories(run)
        axis = fig.axes[0]
        self.assertEqual(len(axis.lines), len(history))
        for line, row in zip(axis.lines, history):
            np.testing.assert_array_equal(line.get_xydata(), row["positions"])
        self.assertEqual(len(axis.patches), mdp.n_grid_states)
        obstacle = axis.patches[mdp.cell_to_state((1, 0))]
        self.assertEqual(obstacle.get_xy(), (2, 0))
        self.assertEqual((obstacle.get_width(), obstacle.get_height()), (2, 1))
        self.assertEqual(obstacle.get_facecolor(), to_rgba("0.25"))
        self.assertEqual({text.get_text() for text in axis.texts}, {"S", "G"})
        positions = np.concatenate(paths)
        self.assertLess(axis.get_xlim()[0], positions[:, 0].min())
        self.assertGreater(axis.get_xlim()[1], max(mdp.bounds[0], positions[:, 0].max()))
        self.assertLess(axis.get_ylim()[0], min(0, positions[:, 1].min()))
        self.assertGreater(axis.get_ylim()[1], max(mdp.bounds[1], positions[:, 1].max()))
        self.assertFalse(np.array_equal(axis.lines[0].get_color(), axis.lines[-1].get_color()))
        self.assertEqual(fig.axes[1].get_ylabel(), "Deployment episode")

    def test_comparison_saves_one_trajectory_figure_per_trial_and_method_without_deployment(self):
        with (
            tempfile.TemporaryDirectory(prefix="robot_online_plots_") as directory,
            patch("robot_2d.main_robot_2d_online.__file__", str(Path(directory) / "main.py")),
            patch("robot_2d.plot_online_comparison.__file__", str(Path(directory) / "plot.py")),
        ):
            output_dir = Path(directory) / "saved_results/online"
            expected = set()
            for trial in (1, 2):
                mdp = Robot2DMotionMDP(
                    x_max=3, y_max=3, grid_shape=(3, 3), obstacles=[(1, trial - 1)],
                )
                for method in ("proposed", "nominal", "R_contamination_0.1", "R_contamination_0.3"):
                    main(mdp, method=method, trial=trial, seed=trial, n_deployment_episodes=3,
                         training_episodes=10, horizon=5, calibration_scores=[0.8] * 20,
                         save_plots=False, verbose=False)
                    R, C = {"proposed": (None, None), "nominal": (0.0, 0.0),
                            "R_contamination_0.1": (0.1, np.inf),
                            "R_contamination_0.3": (0.3, np.inf)}[method]
                    for extension in ("png", "pdf"):
                        expected.add(f"trajectories_{method}_R{R}_C{C}_trial_{trial}.{extension}")
            saved_results = {path: path.read_bytes() for path in output_dir.glob("*.pkl")}
            with patch.object(Robot2DMotionMDP, "deploy_policy", side_effect=AssertionError):
                plot_comparison()
            for name in ("r_estimation", "acp_calibration", "policy_comparison"):
                expected.update(f"{name}.{extension}" for extension in ("png", "pdf"))
            figures = {path.name for path in output_dir.iterdir() if path.suffix != ".pkl"}
            self.assertEqual(figures, expected)
            for name in expected:
                self.assertGreater((output_dir / name).stat().st_size, 0)
            for path, contents in saved_results.items():
                self.assertEqual(path.read_bytes(), contents)
            self.assertFalse(plt.get_fignums())


if __name__ == "__main__":
    unittest.main()
