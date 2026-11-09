"""Compare zero and transformer initialization on one freshly sampled 2-D map."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grid_q_data import training_maps
from q_transformer.evaluate import compare_initializations
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP


if __name__ == "__main__":
    R = 0.2
    C = 0.75
    lr_init = 0.5
    step_start_decay_lr = 10000000
    epsilon_init = 1.0
    epsilon_lb = 0.01
    epsilon_decay_rate = 0.999
    algorithm = "robust_value_iteration"  # Or "robust_q_learning".
    tolerance = 1e-6
    seed = None  # Set an integer to repeat the map and learning RNG seeds.
    n_obstacles = (25, 40)
    max_iterations = 10000
    max_training_episodes = 200000
    check_every = 100
    checkpoint_path = Path(__file__).resolve().parent / "saved_results/q_transformer/checkpoint"

    template = Robot2DMotionMDP(
        x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(1, 1), goal_cell=(8, 8),
        gamma=0.9, movement_reward=0.0, goal_reward=1.0,
    )
    mdp = next(training_maps(template, 1, n_obstacles=n_obstacles, seed=seed))
    compare_initializations(
        mdp, checkpoint_path, R=R, C=C, lr_init=lr_init, step_start_decay_lr=step_start_decay_lr,
        epsilon_init=epsilon_init, epsilon_lb=epsilon_lb, epsilon_decay_rate=epsilon_decay_rate,
        algorithm=algorithm, tolerance=tolerance,
        max_iterations=max_iterations, max_training_episodes=max_training_episodes,
        check_every=check_every, seed=seed,
    )
