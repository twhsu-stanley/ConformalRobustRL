"""Check dataset reconstruction, greedy tie-breaking, and display-only plotting."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrow
import numpy as np

from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP
from robot_2d.utils_2d import plot_robot_motion
from robot_2d.visualize_robot_2d_q_data import _mdp_from_description, _nominal_policy_trajectory
from Tabular_Agent import Tabular_Agent
from tests.script_runner import script_entrypoint


main = script_entrypoint("robot_2d.visualize_robot_2d_q_data", "fig", positional=('index',))


class DatasetVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.mdp = Robot2DMotionMDP(
            x_max=4, y_max=3, grid_shape=(4, 3), start_cell=(0, 0), goal_cell=(3, 2),
            gamma=0.9, movement_reward=0.0, goal_reward=1.0, obstacles=[(1, 0), (1, 1), (3, 1)],
        )

    def test_map_orientation_and_greedy_ties_preserve_a_goal_reaching_policy(self):
        recovered = _mdp_from_description(self.mdp.q_transformer_context())
        np.testing.assert_array_equal(recovered.obstacle_mask, self.mdp.obstacle_mask)
        self.assertEqual(recovered.start_state, self.mdp.start_state)
        self.assertEqual(recovered.goal_state, self.mdp.goal_state)
        Q = np.zeros((self.mdp.n_state, 4))
        policy, trajectory = _nominal_policy_trajectory(recovered, Q)
        np.testing.assert_array_equal(trajectory[0], self.mdp.coordinates(self.mdp.start_state))
        np.testing.assert_array_equal(trajectory[-1], self.mdp.coordinates(self.mdp.goal_state))
        np.testing.assert_array_equal(np.linalg.norm(np.diff(trajectory, axis=0), axis=1), 1)
        self.assertEqual(len(np.unique(trajectory, axis=0)), len(trajectory))
        for current, successor in zip(trajectory[:-1], trajectory[1:]):
            state = recovered.discretize(current)
            action = policy[recovered.state_index(state)]
            self.assertEqual(recovered.nominal_next_state(state, action),
                             recovered.discretize(successor))
        Q[self.mdp.state_index(self.mdp.start_state), 1] = 1
        with self.assertRaisesRegex(ValueError, "no goal-reaching nominal greedy policy"):
            _nominal_policy_trajectory(recovered, Q)

    def test_one_based_indices_plot_values_arrows_and_nominal_path_without_saving(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data_dir = directory / "saved_results/q_transformer"
            data_dir.mkdir(parents=True)
            first, _ = Tabular_Agent(self.mdp, 0.5).value_iteration(R=0, C=0)
            last, _ = Tabular_Agent(self.mdp, 0.5).value_iteration(R=0.5, C=np.inf)
            filename = data_dir / "q_data_2d.npz"
            np.savez_compressed(
                filename, descriptions=[self.mdp.q_transformer_context()], map_index=[0, 0],
                R=[0, 0.5], C=[0, np.inf], Q=[first, last],
            )
            with (
                patch("robot_2d.visualize_robot_2d_q_data.__file__", str(directory / "view.py")),
                patch("robot_2d.visualize_robot_2d_q_data.plot_robot_motion",
                      wraps=plot_robot_motion) as plot,
                patch.object(Figure, "savefig", side_effect=AssertionError("No figures saved")),
            ):
                for index, expected in ((1, first), (2, last)):
                    fig = main(index, show=False)
                    self.addCleanup(plt.close, fig)
                    fig.canvas.draw()
                    values, policy = fig.axes[:2]
                    np.testing.assert_array_equal(values.images[0].get_array(),
                                                  expected.max(axis=1).reshape(3, 4))
                    self.assertEqual(len(values.texts), self.mdp.n_state + 2)
                    arrows = [item for item in policy.patches if isinstance(item, FancyArrow)]
                    self.assertEqual(len(arrows), len(self.mdp.trainable_states))
                    trajectory = policy.lines[0].get_xydata()
                    np.testing.assert_array_equal(trajectory[0], (0.5, 0.5))
                    np.testing.assert_array_equal(trajectory[-1], (3.5, 2.5))
                self.assertEqual(plot.call_count, 2)
                for invalid in (0, -1, 3, 1.5):
                    with self.assertRaisesRegex(ValueError, "integer from 1 to 2"):
                        main(invalid, show=False)
            self.assertEqual([path.name for path in data_dir.iterdir()], ["q_data_2d.npz"])


if __name__ == "__main__":
    unittest.main()
