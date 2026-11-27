"""Disturbances and plots for the 3-D robot example."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtri

from utils import set_plot_style


def sample_deviation_3d(nominal_position, rng, *, R_star, cell_widths=(1., 1., 1.)):
    """Draw a Gaussian error using fresh uniform simplex weights at every step.

    Conditional on the weights, the axes are independent zero-mean Gaussians.
    Every conditional law leaves the nominal cell with probability R_star.
    There is no finite support radius; marginally the law is a Gaussian mixture.
    """
    if np.shape(nominal_position) != (3,):
        raise ValueError("The 3-D disturbance requires a 3-D position.")
    weights = rng.dirichlet(np.ones(3))
    variances = gaussian_directional_variance_3d(R_star, weights, cell_widths)
    return rng.normal(scale=np.sqrt(variances), size=3)


def gaussian_directional_variance_3d(R_star, weights, cell_widths=(1., 1., 1.)):
    """Choose diagonal variances with nominal-cell probability exactly 1 - R_star.

    Axis i stays in its cell with probability (1 - R_star)**weights[i]. The
    weights are positive and sum to one. Half cell widths set the physical scale.
    """
    if not np.isfinite(R_star) or not 0 < R_star < 1:
        raise ValueError("R_star must be in (0, 1) for a nondegenerate Gaussian.")
    weights = np.asarray(weights, dtype=float)
    widths = np.asarray(cell_widths, dtype=float)
    if weights.shape != (3,) or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("weights must contain three positive finite values.")
    if not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("weights must sum to one.")
    if widths.shape != (3,) or np.any(~np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("cell_widths must contain three positive finite values.")
    # Use the lower Gaussian tail to avoid cancellation when the inside mass is near one.
    tail = -np.expm1(weights * np.log1p(-R_star))
    standard_deviations = (widths / 2) / -ndtri(tail / 2)
    return standard_deviations**2


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
        axis.text(*position, label, fontsize=1.5 * plt.rcParams["font.size"], weight="bold")
    axis.set(xlim=(0, mdp.bounds[0]), ylim=(0, mdp.bounds[1]), zlim=(0, mdp.bounds[2]),
             xlabel="x", ylabel="y", zlabel="z")
    axis.set_box_aspect(mdp.bounds.copy())


def plot_robot_motion(mdp, Q, policy, trajectory=None, title="Robot motion"):
    """Plot workspace values, six-direction policy arrows, and an optional trajectory."""
    set_plot_style()
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
        axes[1].set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))
    axes[0].set_title("Value function")
    axes[1].set_title("Policy and deployment trajectory")
    fig.suptitle(title)
    return fig


def plot_robot_3d_tabular(mdp, agent, *, z_layers=None, show=True):
    """Plot value/policy slices; triangles mark +z/-z actions in the policy panels."""
    set_plot_style()
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
