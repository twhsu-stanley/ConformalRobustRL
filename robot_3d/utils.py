"""Map generation, disturbances, and plots for the 3-D robot example."""


import matplotlib.pyplot as plt
import numpy as np


def sample_deviation(nominal_position, rng, *, radius, R_star):
    """Mix zero with six axis displacements that always leave the nominal unit voxel."""
    if rng.random() >= R_star:
        return np.zeros(3)
    direction = int(rng.integers(6))
    magnitude = rng.uniform((0.5 + radius) / 2, radius)
    deviation = np.zeros(3)
    deviation[direction // 2] = magnitude if direction % 2 == 0 else -magnitude
    return deviation


def randomize_obstacles(mdp, rng, n_obstacles=20):
    """Sample obstacle voxels with endpoint clearance and a nominal path to the goal."""
    if not isinstance(n_obstacles, (int, np.integer)) or n_obstacles < 0:
        raise ValueError("n_obstacles must be a nonnegative integer.")
    candidates = [
        tuple(cell) for cell in mdp.states
        if all(np.max(np.abs(cell - endpoint)) >= 2
               for endpoint in (mdp.start_state, mdp.goal_state))
    ]
    if len(candidates) < n_obstacles:
        raise ValueError(f"The grid has too few eligible cells for {n_obstacles} obstacles.")
    for _ in range(1000):
        indices = rng.choice(len(candidates), size=n_obstacles, replace=False)
        obstacles = [candidates[index] for index in indices]
        try:
            mdp.set_obstacles(obstacles)
        except ValueError:
            continue
        return obstacles
    raise ValueError(f"Could not sample a {n_obstacles}-obstacle map with a start-to-goal path.")


def plot_robot_map(mdp, axis):
    """Draw occupied voxels and endpoints in physical coordinates on a 3-D axis."""
    edges = [np.linspace(0, bound, size + 1) for bound, size in zip(mdp.bounds, mdp.grid_shape)]
    mesh = np.meshgrid(*edges, indexing="ij")
    axis.voxels(*mesh, mdp.obstacle_mask.transpose(2, 1, 0), facecolors="0.35", alpha=0.35)
    for state, label, color in (
        (mdp.start_state, "S", "tab:blue"), (mdp.goal_state, "G", "tab:green"),
    ):
        position = mdp.coordinates(state)
        axis.scatter(*position, color=color, s=55)
        axis.text(*position, label, fontsize=12, weight="bold")
    axis.set(xlim=(0, mdp.bounds[0]), ylim=(0, mdp.bounds[1]), zlim=(0, mdp.bounds[2]),
             xlabel="x", ylabel="y", zlabel="z")
    axis.set_box_aspect(mdp.bounds.copy())


def plot_robot_motion(mdp, Q, policy, trajectory=None, title="Robot motion"):
    """Plot workspace values, six-direction policy arrows, and an optional trajectory."""
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    axes = [fig.add_subplot(1, 2, index, projection="3d") for index in (1, 2)]
    for axis in axes:
        plot_robot_map(mdp, axis)
    mask = ~mdp.terminal_mask
    positions = mdp.state_coordinates[mask]
    image = axes[0].scatter(*positions.T, c=Q.max(axis=1)[mask], cmap="YlGnBu", alpha=0.6)
    fig.colorbar(image, ax=axes[0], shrink=0.65, label="Learned robust value")
    directions = np.asarray(mdp.ACTION_DIRECTIONS)[np.asarray(policy)[mask]]
    arrows = directions * mdp.cell_widths * 0.45
    axes[1].quiver(*positions.T, *arrows.T, color="tab:blue", arrow_length_ratio=0.3, alpha=0.4)
    if trajectory is not None:
        trajectory = np.asarray(trajectory)
        axes[1].plot(*trajectory.T, "o-", color="tab:orange", markersize=3)
        lower = np.minimum(trajectory.min(axis=0), 0) - 0.2 * mdp.cell_widths
        upper = np.maximum(trajectory.max(axis=0), mdp.bounds) + 0.2 * mdp.cell_widths
        axes[1].set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
                    zlim=(lower[2], upper[2]))
    axes[0].set_title("Value function")
    axes[1].set_title("Policy and deployment trajectory")
    fig.suptitle(title)
    return fig


def plot_robot_3d_tabular(mdp, agent, *, z_layers=None, show=True):
    """Plot value/policy slices; triangles mark +z/-z actions in the policy panels."""
    if agent.Q.shape != (mdp.n_state, mdp.action_space.n):
        raise ValueError("The agent Q-table must match the MDP state and action spaces.")
    if z_layers is None:
        z_layers = sorted({mdp.start_state[2], mdp.nz // 2, mdp.goal_state[2]})
    if not z_layers or any(int(z) != z or not 0 <= z < mdp.nz for z in z_layers):
        raise ValueError("z_layers must contain valid voxel-layer indices.")
    fig, axes = plt.subplots(2, len(z_layers), figsize=(5 * len(z_layers), 9), squeeze=False,
                             constrained_layout=True)
    values = agent.Q.max(axis=1).reshape(mdp.nz, mdp.ny, mdp.nx)
    policy = agent.Q.argmax(axis=1).reshape(mdp.nz, mdp.ny, mdp.nx)
    extent = (0, mdp.bounds[0], 0, mdp.bounds[1])
    for column, z in enumerate(z_layers):
        z = int(z)
        image = axes[0, column].imshow(values[z], origin="lower", extent=extent, cmap="YlGnBu")
        fig.colorbar(image, ax=axes[0, column], shrink=0.7, label="V(s)")
        for axis in axes[:, column]:
            axis.set(xlim=extent[:2], ylim=extent[2:], aspect="equal", xlabel="x", ylabel="y")
            for iy, ix in np.argwhere(mdp.obstacle_mask[z]):
                axis.add_patch(plt.Rectangle(
                    (ix * mdp.cell_widths[0], iy * mdp.cell_widths[1]), *mdp.cell_widths[:2],
                    facecolor="0.3", edgecolor="black",
                ))
            for state, label in ((mdp.start_state, "S"), (mdp.goal_state, "G")):
                if state[2] == z:
                    axis.text(*mdp.coordinates(state)[:2], label, ha="center", weight="bold")
        for iy in range(mdp.ny):
            for ix in range(mdp.nx):
                if mdp.is_terminal((ix, iy, z)):
                    continue
                action = int(policy[z, iy, ix])
                x, y = mdp.coordinates((ix, iy, z))[:2]
                if action >= 4:
                    axes[1, column].plot(x, y, "^" if action == 4 else "v", color="tab:purple")
                else:
                    dx, dy = np.asarray(mdp.ACTION_DIRECTIONS[action][:2]) * mdp.cell_widths[:2]
                    axes[1, column].arrow(x, y, dx * 0.3, dy * 0.3, head_width=0.12,
                                          length_includes_head=True, color="tab:blue")
        axes[0, column].set_title(f"Values, z index {z}")
        axes[1, column].set_title(f"Policy, z index {z} (+z: ▲, -z: ▼)")
    fig.suptitle(f"Robust Q-learning: R={agent.R}, C={agent.C}")
    if show:
        plt.show()
    return fig


def C_star_shift(episode, period=100):
    """Slowly cycle within [0.5, 1.5], staying strictly above the half-cell threshold."""
    if not isinstance(period, (int, np.integer)) or period < 2:
        raise ValueError("The distribution-shift period must be an integer of at least 2.")
    phase = (episode % period + 0.5) / period
    return float(0.5 + 1.0 * np.cos(np.pi * phase)**2)


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
    axes[1].plot(episodes, scores, label="Maximum deviation")
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
        coverage = 1 - np.cumsum(miscoverage) / episodes
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
