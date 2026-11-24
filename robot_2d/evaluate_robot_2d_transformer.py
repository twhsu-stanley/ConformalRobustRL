"""Compare Q prediction errors and convergence steps on sampled 2-D map/R/C cases."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q_transformer.evaluate import evaluate_initializations
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP


if __name__ == "__main__":
    n_cases = 100
    lr_init = 0.5
    step_start_decay_lr = 10000
    epsilon_init = 0.1
    epsilon_lb = 0.01
    epsilon_decay_rate = 0.999
    algorithm = "robust_q_learning" # "robust_value_iteration" #
    tolerance = 1e-4
    reference_tolerance = 1e-10
    seed = 168  # Set None for new random cases on each run.
    n_obstacles = (25, 40)
    max_iterations = 10000
    max_training_episodes = 200000
    check_every = 100
    accelerated = True  # Use the existing Numba backend for Q-learning.
    root = Path(__file__).resolve().parent / "saved_results"
    dataset_path = root / "q_transformer/q_data_2d.npz"
    checkpoint_path = root / "q_transformer/checkpoint"
    output_dir = root / "q_transformer/evaluation"

    template = Robot2DMotionMDP(
        x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(1, 1), goal_cell=(8, 8),
        gamma=0.9, movement_reward=0.0, goal_reward=1.0,
    )
    results = evaluate_initializations(
        template, dataset_path, checkpoint_path, n_cases=n_cases, n_obstacles=n_obstacles,
        lr_init=lr_init, step_start_decay_lr=step_start_decay_lr,
        epsilon_init=epsilon_init, epsilon_lb=epsilon_lb, epsilon_decay_rate=epsilon_decay_rate,
        algorithm=algorithm, tolerance=tolerance, reference_tolerance=reference_tolerance,
        max_iterations=max_iterations, max_training_episodes=max_training_episodes,
        check_every=check_every, seed=seed, accelerated=accelerated, output_dir=output_dir,
    )
