"""Plotting utilities for Robot2DMotionMDP experiments."""

import matplotlib.pyplot as plt
import numpy as np


def C_star_shift(episode, period=100):
    """Slowly cycle within [0.5, 2.0], staying strictly above the half-cell threshold."""
    if not isinstance(period, (int, np.integer)) or period < 2:
        raise ValueError("The distribution-shift period must be an integer of at least 2.")
    phase = (episode % period + 0.5) / period
    return float(0.5 + 1.0 * np.cos(np.pi * phase)**2)


def sample_deviation(nominal_position, rng, *, radius, R_star):
    """Mix zero with axis displacements that always leave the nominal unit cell.

    The law ignores nominal_position, so it is the same at every state and action.
    Every nonzero draw has norm strictly above 0.5 and at most radius. Its support
    reaches radius, and its mismatch probability is exactly R_star, even at exits.
    """
    if rng.random() >= R_star:
        return np.zeros(2)
    direction = int(rng.integers(4))
    magnitude = rng.uniform((0.5 + radius) / 2, radius)
    deviation = np.zeros(2)
    deviation[direction // 2] = magnitude if direction % 2 == 0 else -magnitude
    return deviation


def randomize_obstacles(mdp, rng, n_obstacles=10):
    """Place ten obstacles at least two cell indices from start and goal, with a path."""
    candidates = [
        (ix, iy) for iy in range(mdp.ny) for ix in range(mdp.nx)
        if all(max(abs(ix - sx), abs(iy - sy)) >= 2
               for sx, sy in (mdp.start_cell, mdp.goal_cell))
    ]
    if len(candidates) < n_obstacles:
        raise ValueError(f"The grid has too few eligible cells for {n_obstacles} obstacles.")
    previous = set(map(tuple, np.argwhere(mdp.obstacle_mask)[:, ::-1]))
    for _ in range(1000):
        indices = rng.choice(len(candidates), size=n_obstacles, replace=False)
        obstacles = [candidates[index] for index in indices]
        if set(obstacles) == previous:
            continue
        try:
            mdp.set_obstacles(obstacles)
        except ValueError:
            continue
        return obstacles
    raise ValueError(f"Could not sample a {n_obstacles}-obstacle map with a start-to-goal path.")


def plot_robot_map(mdp, axis):
    """Draw the physical grid, obstacles, start, and goal on an existing axis."""
    for state in range(mdp.n_grid_states):
        ix, iy = mdp.state_to_cell(state)
        color = "0.25" if mdp.obstacle_mask[iy, ix] else "none"
        if state == mdp.start_state:
            color = "lightskyblue"
        elif state == mdp.goal_state:
            color = "lightgreen"
        rectangle = plt.Rectangle(
            (ix * mdp.cell_widths[0], iy * mdp.cell_widths[1]), *mdp.cell_widths,
            facecolor=color, edgecolor="0.6", linewidth=0.7,
        )
        axis.add_patch(rectangle)
    axis.set(xlim=(0, mdp.bounds[0]), ylim=(0, mdp.bounds[1]), xlabel="x", ylabel="y",
             aspect="equal")
    axis.set_xticks(np.linspace(0, mdp.bounds[0], mdp.nx + 1))
    axis.set_yticks(np.linspace(0, mdp.bounds[1], mdp.ny + 1))
    for state, label in ((mdp.start_state, "S"), (mdp.goal_state, "G")):
        x, y = mdp.coordinates(state)
        axis.text(x, y, label, ha="center", va="center", weight="bold")


def plot_robot_motion(mdp, Q, policy, trajectory=None, title="Robot motion"):
    """Plot values, direction commands, and an optional continuous trajectory."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    extent = (0, mdp.bounds[0], 0, mdp.bounds[1])
    values = np.max(Q[:mdp.n_grid_states], axis=1).reshape(mdp.ny, mdp.nx)
    image = axes[0].imshow(values, origin="lower", extent=extent, cmap="YlGnBu")
    fig.colorbar(image, ax=axes[0], label="Learned robust value")
    for axis in axes:
        plot_robot_map(mdp, axis)
    for state in mdp.trainable_states:
        x, y = mdp.coordinates(state)
        value_y = y - 0.2 * mdp.cell_widths[1] if state == mdp.start_state else y
        axes[0].text(x, value_y, f"{np.max(Q[state]):.2f}", ha="center", va="center", fontsize=7)
        direction = np.asarray(mdp.ACTION_DIRECTIONS[int(policy[state])])
        dx, dy = direction * mdp.cell_widths * 0.28
        axes[1].arrow(
            x, y, dx, dy, head_width=min(mdp.cell_widths) * 0.12,
            length_includes_head=True, color="tab:blue",
        )
    if trajectory is not None:
        trajectory = np.asarray(trajectory)
        axes[1].plot(trajectory[:, 0], trajectory[:, 1], "o-", color="tab:orange", markersize=3)
        exits = np.array([mdp.discretize(x) == mdp.exit_state for x in trajectory])
        if np.any(exits):
            axes[1].plot(*trajectory[exits].T, "rx", markersize=9, label="Workspace exit")
            lo = np.minimum(np.min(trajectory, axis=0), 0) - 0.2 * mdp.cell_widths
            hi = np.maximum(np.max(trajectory, axis=0), mdp.bounds) + 0.2 * mdp.cell_widths
            axes[1].set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
            axes[1].legend()
    axes[0].set_title("Value function")
    axes[1].set_title("Policy and deployment trajectory")
    fig.suptitle(title)
    return fig


def plot_robot_2d_tabular(mdp, agent, *, show=True):
    """Plot V(s)=max_a Q(s,a) and the greedy policy on the physical robot grid."""
    if agent.Q.shape != (mdp.n_state, mdp.action_space.n):
        raise ValueError("The agent Q-table must match the MDP state and action spaces.")
    title = f"Robust Q-Learning for 2-D Robot Motion: R={agent.R}, C={agent.C}"
    fig = plot_robot_motion(mdp, agent.Q, np.argmax(agent.Q, axis=1), title=title)
    fig.axes[1].set_title("Greedy policy from learned Q")
    if show:
        plt.show()
    return fig


def plot_conformal_history(history, delta):
    """Plot deployment uncertainty, showing estimation and ACP only when enabled."""
    if not history:
        raise ValueError("At least one deployment episode is required for plotting.")
    has_acp = "C_hat" in history[0]
    fig, axes = plt.subplots(3 if has_acp else 2, 2, figsize=(12, 11 if has_acp else 7),
                             constrained_layout=True)
    episodes = np.arange(1, len(history) + 1)
    axes = axes.ravel()
    axes[0].plot(episodes, [item["R_used"] for item in history], label="R used for training")
    if "R_hat_next" in history[0]:
        axes[0].plot(episodes, [item["R_hat_next"] for item in history], label="Updated R estimate")
    axes[0].set(title="Planning R", ylabel="Probability", ylim=(0, 1))
    scores = np.asarray([item["score"] for item in history], dtype=float)
    radii = np.asarray([item["C_used"] for item in history])
    finite_radii = np.where(np.isfinite(radii), radii, np.nan)
    ceiling = max(1.0, np.nanmax(scores, initial=0),
                  np.max(radii[np.isfinite(radii)], initial=0)) * 1.1
    axes[1].plot(episodes, scores, label="Maximum nonterminal deviation")
    axes[1].plot(episodes, finite_radii, label="Conformal radius" if has_acp else "Fixed radius")
    if "C_star_k" in history[0]:
        support_bounds = [item["C_star_k"] for item in history]
        axes[1].plot(episodes, support_bounds, "k:", label="True C*")
        ceiling = max(ceiling, max(support_bounds) * 1.1)
    if np.any(np.isposinf(radii)):
        axes[1].scatter(
            episodes[np.isposinf(radii)], np.full(np.isposinf(radii).sum(), ceiling),
            marker="^", label="Unbounded prediction",
        )
    axes[1].set(title="Continuous uncertainty", ylabel="Distance")
    if has_acp:
        axes[2].plot(episodes, [item["delta_k"] for item in history], label="Adaptive delta")
        axes[2].axhline(delta, color="black", linestyle="--", label="Target failure")
        axes[2].set(title="ACP update", ylabel="Failure parameter")
        miscoverage = np.asarray([item["miscoverage"] for item in history], dtype=float)
        miscoverage = np.ma.masked_invalid(miscoverage)
        counts = (~np.ma.getmaskarray(miscoverage)).cumsum()
        coverage = 1 - np.ma.divide(miscoverage.filled(0).cumsum(), counts)
        axes[3].plot(episodes, coverage, label="Conformal coverage")
        axes[3].axhline(1 - delta, color="black", linestyle="--", label="Coverage target")
        axes[3].set(title="Cumulative empirical coverage", ylabel="Fraction", ylim=(0, 1.05))
    returns_axis, residual_axis = axes[-2:]
    returns = [item["discounted_return"] for item in history]
    returns_axis.plot(episodes, returns, label="Discounted r(s,a)")
    for key, marker, color in (
        ("goal_reached", "o", "green"), ("collision", "x", "red"),
        ("workspace_exit", "x", "purple"),
    ):
        indices = np.flatnonzero([item[key] for item in history])
        returns_axis.scatter(episodes[indices], np.asarray(returns)[indices], marker=marker,
                             color=color, label=key.replace("_", " "))
    returns_axis.set(title="Deployment outcomes", ylabel="Return")
    residuals = [max(item["training"]["bellman_residual"], 1e-15) for item in history]
    residual_axis.semilogy(episodes, residuals, label="Robust Bellman residual")
    residual_axis.set(title="Finite training accuracy", ylabel="Residual")
    for axis in axes:
        axis.set_xlabel("Deployment episode")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    return fig
