"""Compare solver updates from zero initialization and a transformer warm start."""

import numpy as np

from Tabular_Agent import Tabular_Agent


def compare_initializations(
    mdp, checkpoint_path, *, R, C, lr_init, step_start_decay_lr, epsilon_init, epsilon_lb,
    epsilon_decay_rate, seed, algorithm="robust_value_iteration", tolerance=1e-6,
    max_iterations=10000, max_training_episodes=200000, check_every=100,
):
    """Use one map, one Bellman tolerance, and matching Q-learning RNG seeds.

    Q-learning counts actual training transitions. Value iteration counts one
    update per nonterminal state-action pair per sweep. Residual checks and
    transformer inference are excluded from these learning-update counters.
    """
    if algorithm not in ("robust_value_iteration", "q_learning"):
        raise ValueError("algorithm must be 'robust_value_iteration' or 'q_learning'.")
    from q_transformer.model import QTransformerPredictor

    initial_Q = QTransformerPredictor(checkpoint_path)(mdp, R, C)
    if algorithm == "q_learning" and seed is None:
        seed = int(np.random.default_rng().integers(2**32))
    steps = {}
    print(f"Algorithm={algorithm}, R={R}, C={C}, Bellman tolerance={tolerance:g}")
    if algorithm == "robust_value_iteration":
        print("Global time steps count individual nonterminal state-action updates.")
    else:
        print(f"Global time steps count training transitions; check_every={check_every} episodes.")
    for name, Q in (("Zero initialization", None), ("Transformer warm start", initial_Q)):
        agent = Tabular_Agent(env=mdp, R=R, C=C)
        if algorithm == "robust_value_iteration":
            agent.robust_value_iteration(
                initial_Q=Q, tolerance=tolerance, max_iterations=max_iterations,
            )
        else:
            agent.robust_q_learning(
                max_training_episodes, initial_Q=Q, tolerance=tolerance, check_every=check_every,
                record_every=max_training_episodes * mdp.unwrapped.max_episode_steps,
                lr_init=lr_init, step_start_decay_lr=step_start_decay_lr,
                epsilon_init=epsilon_init, epsilon_lb=epsilon_lb,
                epsilon_decay_rate=epsilon_decay_rate, seed=seed,
            )
        diagnostics = agent.diagnostics
        count = diagnostics["global_steps"]
        steps[name] = count
        print(f"{name}: global time steps={count:,}; "
              f"Bellman residual={diagnostics['bellman_residual']:.6g}", flush=True)
    return steps
