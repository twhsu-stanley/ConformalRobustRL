"""Disturbances and plots for the 2-D robot example."""

import matplotlib.pyplot as plt
import numpy as np

from utils import set_plot_style


def sample_deviation_2d(nominal_position, rng, *, radius, R_star, peak_angles, min_radius=None):
    """Mix zero error with uniform slip lengths and five von Mises directions.

    The nonzero length is uniform on [min_radius, radius], not over the shell's
    area. The default inner radius is the unit cell's half diagonal, sqrt(2)/2,
    so every active disturbance changes the discrete successor almost surely.
    Each step draws Dirichlet(1,...,1) weights and Uniform[0,8) concentrations.
    peak_angles is a writable floating array initialized as 2*pi*arange(5)/5.
    After sampling, its entries advance in place by independent Gaussian
    increments with standard deviation pi/12, wrapped modulo 2*pi, even if
    no slip occurs. Retain this array across pilot and deployment episodes;
    use a fresh array for each trial and method. Position does not affect the law.
    """
    if np.shape(nominal_position) != (2,):
        raise ValueError("The 2-D disturbance requires a 2-D position.")
    if not np.isfinite(R_star) or not 0 <= R_star <= 1:
        raise ValueError("R_star must be in [0, 1].")
    half_diagonal = np.sqrt(2) / 2
    min_radius = half_diagonal if min_radius is None else min_radius
    if not np.isfinite(min_radius) or min_radius < half_diagonal:
        raise ValueError("min_radius must be at least the unit cell's half diagonal.")
    if not np.isfinite(radius) or radius <= min_radius:
        raise ValueError("The outer radius must be finite and exceed min_radius.")
    if (not isinstance(peak_angles, np.ndarray) or peak_angles.shape != (5,)
            or peak_angles.dtype.kind != "f" or not peak_angles.flags.writeable
            or not np.all(np.isfinite(peak_angles))):
        raise ValueError("peak_angles must be a writable finite floating array of length five.")
    weights = rng.dirichlet(np.ones(5))
    concentrations = rng.uniform(0, 8, size=5)
    deviation = np.zeros(2)
    if rng.random() < R_star:
        length = rng.uniform(min_radius, radius)
        mode = rng.choice(5, p=weights)
        theta = rng.vonmises(peak_angles[mode], concentrations[mode]) % (2 * np.pi)
        deviation = length * np.array((np.cos(theta), np.sin(theta)))
    peak_angles[:] = (peak_angles + rng.normal(0, np.pi / 12, size=5)) % (2 * np.pi)
    return deviation


def plot_robot_map(mdp, axis):
    """Draw the physical grid, obstacles, start, and goal on an existing axis."""
    for cell in mdp.states:
        state = tuple(cell)
        ix, iy = state
        color = "0.25" if mdp.obstacle_mask[iy, ix] else "none"
        if state == mdp.start_state:
            color = "blue"
        elif state == mdp.goal_state:
            color = "red"
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
        axis.text(x, y, label, ha="center", va="center", weight="bold",
                  fontsize=1.5 * plt.rcParams["font.size"],
                  color="white" if state == mdp.start_state else "black")


def plot_robot_motion(mdp, Q, policy, trajectory=None, title="Robot motion", *, value_format=".2f"):
    """Plot values, direction commands, and an optional continuous trajectory."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    extent = (0, mdp.bounds[0], 0, mdp.bounds[1])
    values = np.max(Q, axis=1).reshape(mdp.ny, mdp.nx)
    image = axes[0].imshow(values, origin="lower", extent=extent, cmap="YlGnBu")
    fig.colorbar(image, ax=axes[0], label="Value V(s)")
    for axis in axes:
        plot_robot_map(mdp, axis)
    for state in mdp.states:
        state = tuple(state)
        row = mdp.state_index(state)
        x, y = mdp.coordinates(state)
        is_endpoint = state in (mdp.start_state, mdp.goal_state)
        value_y = y - 0.2 * mdp.cell_widths[1] if is_endpoint else y
        color = "white" if mdp.obstacle_mask[state[1], state[0]] else "black"
        axes[0].text(x, value_y, format(np.max(Q[row]), value_format), ha="center",
                     va="center", fontsize=7, color=color)
        if mdp.terminal_mask[row]:
            continue
        direction = np.asarray(mdp.ACTION_DIRECTIONS[int(policy[row])])
        dx, dy = direction * mdp.cell_widths * 0.28
        axes[1].arrow(
            x, y, dx, dy, head_width=min(mdp.cell_widths) * 0.12,
            length_includes_head=True, color="tab:blue",
        )
    if trajectory is not None:
        trajectory = np.asarray(trajectory)
        axes[1].plot(trajectory[:, 0], trajectory[:, 1], "o-", color="tab:orange", markersize=3)
        exits = np.array([mdp.is_outside(mdp.discretize(x)) for x in trajectory])
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


def plot_trajectories(runs, method_label):
    """Overlay all trials on their shared map, coloring each path by deployment episode."""
    if not runs or not any(run["history"] for run in runs):
        raise ValueError("At least one deployment episode is required for plotting.")
    set_plot_style()
    mdp = runs[0]["mdp"]
    episode_count = max(len(run["history"]) for run in runs)
    trajectory_count = sum(len(run["history"]) for run in runs)
    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    plot_robot_map(mdp, axis)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.5, vmax=episode_count + 0.5)
    for run in runs:
        for episode, row in enumerate(run["history"], 1):
            trajectory = np.asarray(row["positions"])
            axis.plot(trajectory[:, 0], trajectory[:, 1], color=cmap(norm(episode)),
                      linewidth=0.5, alpha=0.1)
    axis.set_title(
        f"{method_label}\n{len(runs)} trials $\\times$ {episode_count} episodes/trial"
        f" = {trajectory_count:,} total episodes", pad=12,
    )
    return fig
