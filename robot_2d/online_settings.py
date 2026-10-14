"""Episode settings for the robot experiment with a common true mismatch probability."""

import numpy as np


def C_star_shift(episode, period=100):
    """Slowly cycle within [0.5, 2.0], staying strictly above the half-cell threshold."""
    if not isinstance(period, (int, np.integer)) or period < 2:
        raise ValueError("The distribution-shift period must be an integer of at least 2.")
    phase = (episode % period + 0.5) / period
    return float(0.5 + 1.5 * np.sin(np.pi * phase)**2)


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
