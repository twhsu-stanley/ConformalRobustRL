"""Compare zero and transformer initialization on one freshly sampled 3-D map."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grid_q_data import training_maps
from q_transformer.evaluate import compare_initializations
from robot_3d.robot_3d_motion_mdp import Robot3DMotionMDP


if __name__ == "__main__":
    R = 0.2
    C = 0.75
    lr_init = 0.9
    step_start_decay_lr = 1000000
    epsilon_init = 0.1
    epsilon_lb = 0.01
    epsilon_decay_rate = 0.999
    algorithm = "robust_q_learning"  # Or "robust_value_iteration".
    tolerance = 1e-3
    seed = 67 # 91
    n_obstacles = (250, 400)
    max_iterations = 10000
    max_training_episodes = 100000
    check_every = 100
    checkpoint_path = Path(__file__).resolve().parent / "saved_results/q_transformer/checkpoint"

    template = Robot3DMotionMDP(
        x_max=10.0, y_max=10.0, z_max=10.0, grid_shape=(10, 10, 10),
        start_cell=(1, 1, 1), goal_cell=(8, 8, 8), gamma=0.9,
        movement_reward=0.0, goal_reward=1.0,
    )
    mdp = next(training_maps(template, 1, n_obstacles=n_obstacles, seed=seed))
    compare_initializations(
        mdp, checkpoint_path, R=R, C=C, lr_init=lr_init, step_start_decay_lr=step_start_decay_lr,
        epsilon_init=epsilon_init, epsilon_lb=epsilon_lb, epsilon_decay_rate=epsilon_decay_rate,
        algorithm=algorithm, tolerance=tolerance,
        max_iterations=max_iterations, max_training_episodes=max_training_episodes,
        check_every=check_every, seed=seed,
    )
