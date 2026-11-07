"""Disturbances, map generation, and plots for the 2-D robot example."""

import math
import sys

import matplotlib.pyplot as plt
import numpy as np


# Existing result files store sample_deviation under robot_2d.utils.
sys.modules.setdefault("robot_2d.utils", sys.modules[__name__])


def mdp_from_description(description):
    """Reconstruct the unit-cell MDP used by the saved transformer dataset."""
    from q_transformer.data import parse_grid_description
    from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP

    shape, start, goal, obstacles = parse_grid_description(description)
    if len(shape) != 2:
        raise ValueError("The 2-D example requires a two-dimensional grid description.")
    mdp = Robot2DMotionMDP(
        x_max=float(shape[0]), y_max=float(shape[1]), grid_shape=shape,
        start_cell=start, goal_cell=goal, gamma=0.9, movement_reward=0.0, goal_reward=1.0,
        obstacles=obstacles,
    )
    if mdp.q_transformer_context() != description:
        raise ValueError("The reconstructed MDP does not reproduce its saved description.")
    return mdp


def C_star_shift_2d(episode, period=100):
    """Slowly cycle within [0.5, 1.5], staying strictly above the half-cell threshold."""
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
               for sx, sy in (mdp.start_state, mdp.goal_state))
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
    for cell in mdp.states:
        state = tuple(cell)
        ix, iy = state
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


def plot_robot_motion(mdp, Q, policy, trajectory=None, title="Robot motion", *, value_format=".2f"):
    """Plot values, direction commands, and an optional continuous trajectory."""
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


def plot_trajectories(run, method_label):
    """Overlay every recorded deployment episode on its trial's fixed obstacle map."""
    mdp, history = run["mdp"], run["history"]
    if not history:
        raise ValueError("At least one deployment episode is required for plotting.")
    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    plot_robot_map(mdp, axis)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.5, vmax=len(history) + 0.5)
    trajectories = [np.asarray(row["positions"]) for row in history]
    for episode, trajectory in enumerate(trajectories, 1):
        axis.plot(trajectory[:, 0], trajectory[:, 1], color=cmap(norm(episode)),
                  linewidth=1, alpha=0.5, marker=".", markersize=3)
    positions = np.concatenate(trajectories)
    lo = np.minimum(positions.min(axis=0), 0) - 0.2 * mdp.cell_widths
    hi = np.maximum(positions.max(axis=0), mdp.bounds) + 0.2 * mdp.cell_widths
    axis.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
    ticks = np.unique(np.linspace(1, len(history), min(6, len(history)), dtype=int))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis,
                 ticks=ticks, label="Deployment episode")
    axis.set_title(f"{method_label}, trial {run['trial']}\n"
                   f"{len(history)} deployment trajectories")
    return fig


def disk_rectangle_area(radius, xlo, xhi, ylo, yhi):
    """Disk/rectangle intersection, used only for the offline mismatch reference."""
    xlo, xhi = max(-radius, xlo), min(radius, xhi)
    ylo, yhi = max(-radius, ylo), min(radius, yhi)
    if xlo >= xhi or ylo >= yhi:
        return 0.0
    splits = [xlo, xhi]
    for height in (ylo, yhi):
        if abs(height) < radius:
            root = math.sqrt(radius**2 - height**2)
            splits.extend(x for x in (-root, root) if xlo < x < xhi)
    splits = sorted(set(splits))

    def primitive(x):
        root = math.sqrt(max(0.0, radius**2 - x**2))
        return 0.5 * (x * root + radius**2 * math.asin(x / radius))

    area = 0.0
    for left, right in zip(splits[:-1], splits[1:]):
        height = math.sqrt(max(0.0, radius**2 - ((left + right) / 2)**2))
        if min(yhi, height) <= max(ylo, -height):
            continue
        upper_const, upper_arc = (yhi, 0) if yhi < height else (0, 1)
        lower_const, lower_arc = (ylo, 0) if ylo > -height else (0, -1)
        area += (upper_const - lower_const) * (right - left)
        area += (upper_arc - lower_arc) * (primitive(right) - primitive(left))
    return max(0.0, area)


def mismatch_probabilities(mdp):
    """Common mismatch probability for every grid source, including the goal."""
    if mdp.noise_sampler is not None:
        raise ValueError("This offline reference assumes the default zero/uniform-disk noise.")
    probability = 0.0
    if mdp.noise_radius > 0:
        hx, hy = mdp.cell_widths / 2
        area = disk_rectangle_area(mdp.noise_radius, -hx, hx, -hy, hy)
        nominal_mass = area / (np.pi * mdp.noise_radius**2)
        probability = mdp.noise_probability * (1 - nominal_mass)
    result = np.full((mdp.n_state, mdp.action_space.n), probability)
    return np.clip(result, 0, 1)
