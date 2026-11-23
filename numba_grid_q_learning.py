"""Numba backend for robust Q-learning on deterministic grid robot MDPs.

Tabular_Agent owns Q, the environment, and the training results. This module runs
the same training steps using NumPy arrays that Numba can compile into machine
code. Keeping it separate lets the ordinary agent run without importing Numba.
"""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _minimum_next_value(values, successor, offsets, neighbors, global_neighborhood):
    """Find the smallest continuation value allowed by the uncertainty neighborhood."""
    if global_neighborhood:
        return min(0.0, values.min())
    result = np.inf
    for index in range(offsets[successor], offsets[successor + 1]):
        result = min(result, values[neighbors[index]])
    return result


@njit(cache=True, nogil=True)
def _bellman_residual(
    Q, values, rows, successors, rewards, offsets, neighbors, R, gamma, global_set,
):
    """Check the largest Bellman error across all nonterminal state-action pairs."""
    result = 0.0
    for row in rows:
        for action in range(Q.shape[1]):
            successor = successors[row, action]
            continuation = values[successor]
            if R > 0:
                worst = _minimum_next_value(values, successor, offsets, neighbors, global_set)
                continuation = (1 - R) * continuation + R * worst
            target = rewards[row, action] + gamma * continuation
            result = max(result, abs(target - Q[row, action]))
    return result


@njit(cache=True, nogil=True)
def _train_episodes(
    Q, values, visits, decay_visits, successors, rewards, terminal, offsets, neighbors,
    pairs, first_episode, count, start_row, horizon, rng, epsilon, epsilon_lb,
    epsilon_decay_rate, lr_init, decay_start, global_steps, record_every, R, gamma, global_set,
):
    """Run a batch of nominal episodes, updating Q and visit counts in place."""
    # Record the start-state value every record_every training transitions.
    curve = np.empty(count * horizon // record_every + 2)
    n_records = 0
    worst_global = min(0.0, values.min())
    for episode in range(first_episode, first_episode + count):
        row, first_action = start_row, -1
        if len(pairs):
            row, first_action = pairs[episode % len(pairs)]
        for _ in range(horizon):
            action = first_action
            if action < 0:
                if rng.random() < epsilon:
                    action = rng.integers(0, Q.shape[1])
                else:
                    # Break greedy-action ties uniformly, matching the ordinary solver.
                    maximum = Q[row].max()
                    n_ties = 0
                    for candidate in range(Q.shape[1]):
                        n_ties += int(Q[row, candidate] == maximum)
                    selected = rng.integers(0, n_ties)
                    for candidate in range(Q.shape[1]):
                        if Q[row, candidate] == maximum:
                            if selected == 0:
                                action = candidate
                                break
                            selected -= 1
            first_action = -1
            successor = successors[row, action]
            visits[row, action] += 1
            learning_rate = lr_init
            if global_steps >= decay_start:
                decay_visits[row, action] += 1
                learning_rate /= decay_visits[row, action]
            continuation = values[successor]
            if R > 0:
                worst = worst_global if global_set else _minimum_next_value(
                    values, successor, offsets, neighbors, False,
                )
                continuation = (1 - R) * continuation + R * worst
            target = rewards[row, action] + gamma * continuation
            previous_value = values[row]
            Q[row, action] += learning_rate * (target - Q[row, action])
            values[row] = Q[row].max()
            if global_set:
                # Reuse the global minimum until an update can change it.
                if values[row] < worst_global:
                    worst_global = values[row]
                elif previous_value == worst_global and previous_value < 0:
                    worst_global = min(0.0, values.min())
            global_steps += 1
            if global_steps % record_every == 0:
                curve[n_records] = values[start_row]
                n_records += 1
            row = successor
            if terminal[row]:
                break
        epsilon = max(epsilon_lb, epsilon * epsilon_decay_rate)
    return global_steps, epsilon, curve[:n_records]


def _build_training_tables(base, C):
    """Convert grid coordinates and neighborhoods into arrays used by the compiled loop.

    Workspace states use the same row numbers as Q. Row base.n_state is an extra
    value lookup with value zero for every outside terminal state.

    The neighbors for state s occupy neighbors[offsets[s]:offsets[s + 1]]. This
    packs all neighbor lists into one array, even when their lengths differ.
    """
    shape = np.asarray(base.grid_shape)
    weights = np.cumprod((1,) + tuple(base.grid_shape[:-1]))
    cells = base.nominal_transitions
    inside = np.all((cells >= 0) & (cells < shape), axis=-1)
    successors = np.where(inside, (cells * weights).sum(axis=-1), base.n_state)
    global_set = C >= np.linalg.norm(base.bounds)
    # On a uniform grid, one pattern of relative cells works at every successor.
    stencil = None if global_set else base.localized_uncertainty_states((0,) * len(shape), C)
    key = None if global_set else tuple(map(tuple, stencil))
    cache = base.__dict__.setdefault("_compiled_grid_cache", {})
    # Nearby radii often include exactly the same cells; reuse those tables.
    if key not in cache:
        if global_set:
            offsets = np.r_[np.zeros(base.n_state + 1, dtype=np.int64), 1]
            neighbors = np.array([base.n_state], dtype=np.int64)
        else:
            cells = base.states[:, None, :] + stencil[None, :, :]
            inside = np.all((cells >= 0) & (cells < shape), axis=-1)
            neighbors = np.where(inside, (cells * weights).sum(axis=-1), base.n_state)
            offsets = np.r_[np.arange(base.n_state + 1) * len(stencil), neighbors.size + 1]
            neighbors = np.r_[neighbors.ravel(), base.n_state]
        cache[key] = offsets, neighbors
    offsets, neighbors = cache[key]
    return successors.astype(np.int64), offsets, neighbors, global_set


def run_robust_q_learning(
    agent, max_training_episodes, *, lr_init, step_start_decay_lr, epsilon_init, epsilon_lb,
    epsilon_decay_rate, seed, exploring_starts, record_every, initial_Q, tolerance, check_every,
    verbose,
):
    """Train the supplied Tabular_Agent using its MDP, R, C, and initial Q-table.

    The agent's public method validates the solver settings before calling here.
    Results are written back to that same agent. Deployment continues to use the
    original environment's step() method.
    """
    from grid_robot_mdp import GridRobotMotionMDP

    base = agent.env.unwrapped
    if not isinstance(base, GridRobotMotionMDP) or agent.env is not base:
        raise ValueError("Compiled training requires an unwrapped nominal grid robot MDP.")
    agent.initialize_q(initial_Q)
    successors, offsets, neighbors, global_set = _build_training_tables(base, agent.C)
    rows = np.flatnonzero(~base.terminal_mask)
    terminal = np.r_[base.terminal_mask, True]
    values = np.r_[agent.Q.max(axis=1), 0.0]
    visits = np.zeros_like(agent.Q, dtype=np.int64)
    decay_visits = np.zeros_like(visits)
    start_row = base.state_index(base.start_state)
    agent.evaluation_return = [float(values[start_row])]
    zero_initialized = bool(np.all(agent.Q == 0.0))
    pairs = np.array([(row, action) for row in rows for action in range(agent.n_action)])
    if not exploring_starts:
        pairs = np.empty((0, 2), dtype=np.int64)
    rng = np.random.default_rng(seed)
    epsilon, global_steps, episodes = epsilon_init, 0, 0
    residual = _bellman_residual(
        agent.Q, values, rows, successors, base.rewards, offsets, neighbors,
        agent.R, base.gamma, global_set,
    )
    while residual > tolerance and episodes < max_training_episodes:
        # End each batch at a residual check or a shuffle of the exploring starts.
        count = min(check_every - episodes % check_every, max_training_episodes - episodes)
        if exploring_starts:
            if episodes % len(pairs) == 0:
                rng.shuffle(pairs)
            count = min(count, len(pairs) - episodes % len(pairs))
        global_steps, epsilon, curve = _train_episodes(
            agent.Q, values, visits, decay_visits, successors, base.rewards, terminal,
            offsets, neighbors, pairs, episodes, count, start_row, base.max_episode_steps,
            rng, epsilon, epsilon_lb, epsilon_decay_rate, lr_init, step_start_decay_lr,
            global_steps, record_every, agent.R, base.gamma, global_set,
        )
        agent.evaluation_return.extend(curve.tolist())
        episodes += count
        if episodes % check_every == 0 or episodes == max_training_episodes:
            residual = _bellman_residual(
                agent.Q, values, rows, successors, base.rewards, offsets, neighbors,
                agent.R, base.gamma, global_set,
            )
            if verbose:
                print(f"Episode {episodes}: Bellman residual={residual:.3g}.", flush=True)
    if global_steps % record_every != 0:
        agent.evaluation_return.append(float(values[start_row]))
    agent.visit_counts = visits
    active_visits = visits[~base.terminal_mask]
    agent._record_diagnostics(
        "robust_q_learning", global_steps, residual, tolerance, episodes=episodes,
        minimum_visits=int(active_visits.min()), unvisited_pairs=int((active_visits == 0).sum()),
        initial_state_value=float(values[start_row]), zero_initialized=zero_initialized,
        backend="numba_grid",
    )
    if not agent.diagnostics["converged"]:
        raise RuntimeError(
            f"Q-learning did not converge in {max_training_episodes} episodes: "
            f"Bellman residual {residual:.3g} exceeds tolerance {tolerance:.3g}."
        )
    return agent.Q.copy(), np.argmax(agent.Q, axis=1)
