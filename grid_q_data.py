"""Structured n-dimensional grid maps for robust Q-table datasets."""

from collections import deque
from itertools import product

import numpy as np


def randomize_start_goal(mdp, rng):
    """Fix the start at index one and sample an interior, opposite-half goal."""
    shape = np.asarray(mdp.grid_shape)
    candidates = mdp.states[np.all((mdp.states >= shape // 2) & (mdp.states < shape - 1), axis=1)]
    if not len(candidates):
        raise ValueError("The grid has no interior goal cells in its opposite half.")
    return (1,) * len(shape), tuple(map(int, candidates[rng.integers(len(candidates))]))


def _cell_distances(centers, cells, cell_widths):
    """Distances from cell centers to cell regions, matching the MDP uncertainty sets."""
    offsets = np.abs(np.asarray(centers)[:, None] - np.asarray(cells)[None]) * cell_widths
    return np.linalg.norm(np.maximum(offsets - cell_widths / 2, 0), axis=-1)


def _axis_path(waypoints, axis_order):
    path = [tuple(waypoints[0])]
    for target in waypoints[1:]:
        current = np.array(path[-1])
        for axis in axis_order:
            while current[axis] != target[axis]:
                current[axis] += int(np.sign(target[axis] - current[axis]))
                path.append(tuple(map(int, current)))
    return path


def _grid_path(allowed, starts, goals, rng):
    """Find an axial path between two sets of cells, randomizing shortest-path ties."""
    parents = {tuple(start): None for start in rng.permutation(starts)}
    queue, goals = deque(parents), set(map(tuple, goals))
    directions = np.concatenate((np.eye(allowed.ndim, dtype=int), -np.eye(allowed.ndim, dtype=int)))
    shape = np.asarray(allowed.shape)
    while queue:
        state = queue.popleft()
        if state in goals:
            path = []
            while state is not None:
                path.append(state)
                state = parents[state]
            return path[::-1]
        for direction in directions[rng.permutation(len(directions))]:
            neighbor = tuple(np.asarray(state) + direction)
            if neighbor in parents:
                continue
            if np.any(np.asarray(neighbor) < 0) or np.any(np.asarray(neighbor) >= shape):
                continue
            if not allowed[neighbor]:
                continue
            parents[neighbor] = state
            queue.append(neighbor)
    return None


def _wide_corridor(mdp, blocked, rng):
    """Slide 2-by-2 squares or 2-by-2-by-2 cubes, sharing full faces at every step."""
    dimension = len(mdp.grid_shape)
    shape = np.asarray(mdp.grid_shape) - 1
    offsets = np.array(list(product((0, 1), repeat=dimension)))
    anchors = np.indices(tuple(shape)).reshape(dimension, -1).T
    weights = np.cumprod((1,) + mdp.grid_shape[:-1])
    rows = ((anchors[:, None] + offsets[None]) * weights).sum(axis=-1)
    blocked_rows = np.zeros(mdp.n_state, dtype=bool)
    blocked_rows[[mdp.state_index(state) for state in blocked]] = True
    allowed = ~blocked_rows[rows].any(axis=1)
    starts = anchors[
        allowed & np.all((anchors <= mdp.start_state) & (mdp.start_state <= anchors + 1), axis=1)
    ]
    goals = anchors[
        allowed & np.all((anchors <= mdp.goal_state) & (mdp.goal_state <= anchors + 1), axis=1)
    ]
    if not len(starts) or not len(goals):
        return None
    path = _grid_path(allowed.reshape(tuple(shape)), starts, goals, rng)
    if path is None:
        return None
    cells = np.asarray(path)[:, None] + offsets[None]
    protected = np.zeros(mdp.n_state, dtype=bool)
    protected[(cells * weights).sum(axis=-1).ravel()] = True
    return protected


def _sample_obstacles(mdp, rng, n_obstacles):
    """Protect a wide route and a separate narrow route, then sample obstacles."""
    dimension = len(mdp.grid_shape)
    widths = mdp.cell_widths
    axis_order = rng.permutation(dimension)
    rail_axis = axis_order[0]
    start_rail, goal_rail = list(mdp.start_state), list(mdp.goal_state)
    start_rail[rail_axis] = goal_rail[rail_axis] = mdp.grid_shape[rail_axis] - 2
    narrow = _axis_path([mdp.start_state, start_rail, goal_rail, mdp.goal_state], axis_order)
    blocked = set(narrow[1:-1])
    protected = _wide_corridor(mdp, blocked, rng)
    if protected is None:
        return None
    endpoint_distances = _cell_distances([mdp.start_state, mdp.goal_state], mdp.states, widths)
    protected |= endpoint_distances.min(axis=0) <= 2 * float(np.min(widths)) + 1e-12
    protected[[mdp.state_index(state) for state in narrow]] = True

    # One free cell between obstacle walls forms a genuine narrow bottleneck.
    bottlenecks = []
    for index in range(1, len(narrow) - 1):
        tangent = np.asarray(narrow[index + 1]) - narrow[index - 1]
        if np.count_nonzero(tangent) != 1:
            continue
        walls = []
        for axis in np.flatnonzero(tangent == 0):
            for direction in (-1, 1):
                wall = list(narrow[index])
                wall[axis] += direction
                walls.append(tuple(wall))
        if all(not mdp.is_outside(wall) and not protected[mdp.state_index(wall)] for wall in walls):
            bottlenecks.append(walls)
    if not bottlenecks:
        return None
    forced = {mdp.state_index(wall) for wall in bottlenecks[rng.integers(len(bottlenecks))]}
    candidates = np.flatnonzero(~protected)
    low, high = max(n_obstacles[0], len(forced)), min(n_obstacles[1], len(candidates))
    if low > high:
        return None
    count = int(rng.integers(low, high + 1))
    remaining = np.setdiff1d(candidates, list(forced))
    rows = sorted(forced | set(map(int, rng.choice(remaining, count - len(forced), replace=False))))
    return [tuple(map(int, cell)) for cell in mdp.states[rows]]


def _all_free_states_connected(mdp, obstacles):
    """Reach all free cells from the start, treating the goal as an endpoint."""
    free = set(map(tuple, mdp.states)) - set(map(tuple, obstacles))
    if mdp.start_state not in free or mdp.goal_state not in free:
        return False
    queue, visited = deque([mdp.start_state]), {mdp.start_state}
    while queue:
        state = queue.popleft()
        for direction in mdp.ACTION_DIRECTIONS:
            neighbor = tuple(s + step for s, step in zip(state, direction))
            if neighbor not in free or neighbor in visited:
                continue
            visited.add(neighbor)
            if neighbor != mdp.goal_state:
                queue.append(neighbor)
    return len(visited) == len(free)


def _segment_obstacle_count(mdp, obstacles):
    """Count distinct obstacle cells touched by the complete start-goal segment."""
    cells = np.asarray(obstacles, dtype=int).reshape(-1, len(mdp.grid_shape))
    lower = np.unique(cells, axis=0) * mdp.cell_widths
    upper = lower + mdp.cell_widths
    start = mdp.coordinates(mdp.start_state)
    direction = mdp.coordinates(mdp.goal_state) - start
    near, far = np.full(lower.shape, -np.inf), np.full(lower.shape, np.inf)
    np.divide(lower - start, direction, out=near, where=direction != 0)
    np.divide(upper - start, direction, out=far, where=direction != 0)
    enter = np.maximum(np.minimum(near, far).max(axis=1), 0)
    leave = np.minimum(np.maximum(near, far).min(axis=1), 1)
    parallel_miss = np.any((direction == 0) & ((start < lower) | (start > upper)), axis=1)
    return int(np.count_nonzero((enter <= leave + 1e-12) & ~parallel_miss))


def training_maps(mdp, n_maps, *, n_obstacles, seed=7):
    """Yield distinct seeded maps with a wide route and a separate narrow route.

    The wide passage has two free cells across in 2-D or a 2-by-2 cross-section in
    3-D, including at turns. Endpoint buffers are separate from passage width.
    Additional routes are unrestricted, and map construction does not depend on C.
    Every nonterminal free cell can reach the start and goal without crossing an
    obstacle or using the terminal goal as an intermediate waypoint.
    The straight segment between the endpoint centers must hit at least two distinct obstacle cells.
    Requested obstacle counts are sampled within the capacity of the protected map.
    """
    if not isinstance(n_maps, (int, np.integer)) or n_maps < 2:
        raise ValueError("n_maps must be an integer of at least two.")
    counts = np.asarray(n_obstacles)
    if counts.shape != (2,) or not np.all(np.isfinite(counts)):
        raise ValueError("n_obstacles must contain two finite integer counts.")
    if np.any(counts != counts.astype(int)):
        raise ValueError("n_obstacles must contain two integer counts.")
    low, high = map(int, counts)
    if not 0 <= low <= high <= mdp.n_state - 2:
        raise ValueError("n_obstacles must be ordered nonnegative counts that fit in the grid.")
    rng, seen = np.random.default_rng(seed), set()
    for _ in range(n_maps):
        nominal_mdp = mdp.clone(nominal=True)
        for _ in range(1000):
            nominal_mdp.start_state, nominal_mdp.goal_state = randomize_start_goal(mdp, rng)
            obstacles = _sample_obstacles(nominal_mdp, rng, (low, high))
            if obstacles is None:
                continue
            key = nominal_mdp.goal_state, tuple(obstacles)
            if _segment_obstacle_count(nominal_mdp, obstacles) < 2:
                continue
            if key in seen or not _all_free_states_connected(nominal_mdp, obstacles):
                continue
            nominal_mdp.set_obstacles(obstacles)
            nominal_mdp.reset()
            seen.add(key)
            yield nominal_mdp
            break
        else:
            raise ValueError(
                "Could not construct a distinct map with the requested obstacle count, "
                "wide-corridor cross-section, narrow bottleneck, connected free space, "
                "and two obstacles on the straight start-goal segment. Adjust the "
                "obstacle count or increase the grid size."
            )
