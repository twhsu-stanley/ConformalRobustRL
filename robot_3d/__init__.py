"""Three-dimensional robot motion planning with the shared RL and transformer code.

From the repository root:
    python -m robot_3d.generate_robot_3d_q_data
    python -m robot_3d.train_robot_3d_transformer
    python -m robot_3d.main_robot_3d
    python -m robot_3d.plot_fixedRC_eval_returns
    python -m robot_3d.plot_fixedRC_value_policy
    python -m robot_3d.main_robot_3d_online
    python -m robot_3d.plot_online_comparison

Configure runs in the scripts' main blocks. To enable warm starts after training,
set checkpoint_path to Path(__file__).resolve().parent /
"saved_results/q_transformer/checkpoint". Leave it None for zero initialization.

The 3-D runners allow up to 200,000 Q-learning episodes, check the full Bellman
residual every 1,000 episodes at tolerance 1e-6, and use a 10,000,000-transition
constant-learning-rate warmup before the shared per-pair decay. These larger
budgets account for the 6,000 state-action pairs. Converged runs stop early.
"""
