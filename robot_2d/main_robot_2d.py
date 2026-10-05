"""Run Algorithm 1 at fixed R,C and save initial-state value learning curves."""

import pickle
import sys
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP
from Tabular_Agent import Tabular_Agent


def main(
    *, R=0.15, C=1.2, gamma=0.95, training_episodes=5000, n_trials=15,
    horizon=100, seed=7, output_dir=None,
):
    if not isinstance(n_trials, (int, np.integer)) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer.")
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "saved_results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mdp = Robot2DMotionMDP(
        x_max=8.0, y_max=8.0, grid_shape=(8, 8),
        obstacles=[(2, 1), (5, 1), (1, 3), (4, 3), (6, 4), (2, 5), (5, 6)],
        start_cell=(0, 0), goal_cell=(7, 7), movement_reward=0.01, goal_reward=1.0,
        noise_probability=0.0, noise_radius=0.0, noise_sampler=None,
        max_episode_steps=horizon, episode_mode="episodic", seed=seed,
    )
    evaluation_return = []
    agents = []
    for trial in range(n_trials):
        agent = Tabular_Agent(
            mdp, gamma=gamma, lr_init=0.5, step_start_decay_lr=100000,
            epsilon_init=1.0, epsilon_lb=0.1, epsilon_decay_rate=0.995,
            R=R, C=C, seed=seed + trial,
        )
        agent.robust_q_learning(training_episodes, record_every=1)
        # Match the FrozenLake learning curves: V(start), recorded at every transition.
        evaluation_return.append(agent.evaluation_return.copy())
        agents.append(agent)
        print(
            f"Trial {trial + 1}: R={R}, C={C}, "
            f"V(start)={evaluation_return[-1][-1]:.4f}, "
            f"Bellman residual={agent.training_diagnostics['bellman_residual']:.3g}",
        )

    filename = output_dir / f"robot_2d_R{R}_C{C}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(evaluation_return, f)
    print(f"Saved evaluation return to {Path(filename).resolve()}")
    
    filename = output_dir / f"robot_2d_R{R}_C{C}_agents.pkl"
    with open(filename, "wb") as f:
        pickle.dump({"mdp": mdp, "agents": agents}, f)
    print(f"Saved MDP and trained agents to {Path(filename).resolve()}")
    
    return evaluation_return


if __name__ == "__main__":
    # Fix C = 1.0, compare different R values
    main(R=0.0, C=1.0)

    #main(R=0.1, C=1.0)

    #main(R=0.2, C=1.0)

    # Fix R = 0.2, compare different C values
    #main(R=0.2, C=0.0)
    
    #main(R=0.2, C=1.0)
    
    #main(R=0.2, C=1.5)
