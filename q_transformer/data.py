"""Structured grid MDP parameters and robust value-iteration training data."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from multiprocessing import get_context
from pathlib import Path
import re

import numpy as np
from tqdm import tqdm

from Tabular_Agent import Tabular_Agent


MDP_SHARED_FIELDS = ("bounds", "grid_shape", "gamma", "movement_reward", "goal_reward")
MDP_MAP_FIELDS = ("start_states", "goal_states", "obstacle_masks")


def mdp_data(mdp):
    """Export one nominal grid map as numeric arrays, with a leading map dimension."""
    base = mdp.unwrapped
    return {
        "bounds": np.asarray(base.bounds, dtype=float).copy(),
        "grid_shape": np.asarray(base.grid_shape, dtype=np.int64),
        "gamma": np.asarray(base.gamma, dtype=float),
        "movement_reward": np.asarray(base.movement_reward, dtype=float),
        "goal_reward": np.asarray(base.goal_reward, dtype=float),
        "start_states": np.asarray([base.start_state], dtype=np.int64),
        "goal_states": np.asarray([base.goal_state], dtype=np.int64),
        "obstacle_masks": np.asarray([base.obstacle_mask], dtype=bool),
    }


def mdp_parameters(data, map_index):
    """Turn one saved map's arrays into keyword arguments for its robot MDP class."""
    shape = tuple(map(int, data["grid_shape"]))
    if len(shape) not in (2, 3):
        raise ValueError("Grid MDP parameters require two or three dimensions.")
    return {
        **dict(zip(("x_max", "y_max", "z_max"), map(float, data["bounds"]))),
        "grid_shape": shape,
        "start_cell": tuple(map(int, data["start_states"][map_index])),
        "goal_cell": tuple(map(int, data["goal_states"][map_index])),
        "gamma": float(data["gamma"]),
        "movement_reward": float(data["movement_reward"]),
        "goal_reward": float(data["goal_reward"]),
        "obstacles": [tuple(map(int, cell[::-1]))
                      for cell in np.argwhere(data["obstacle_masks"][map_index])],
    }


def format_prompt(mdp, R, C):
    """Describe a grid task with global R/C and explicit indices, entirely as text."""
    Tabular_Agent._validate_uncertainty(R, C)
    base = mdp.unwrapped
    shape = base.grid_shape
    if len(shape) not in (2, 3):
        raise ValueError("Grid prompts require a two- or three-dimensional workspace.")
    radius = "infinity" if np.isinf(C) else repr(float(C))
    dimensions = "x".join(map(str, shape))
    axes = "Columns increase in x; rows increase in y."
    if len(shape) == 3:
        axes += " Layers increase in z."
    lines = [
        "### Instruction:",
        f"Predict the Q-table for a grid with size {dimensions}.",
        f"Uncertainty parameters: R = {float(R)!r}; C = {radius}.",
        "The same R and C apply to every cell and action.",
        "For a fixed MDP, increasing R or C with the other held fixed cannot increase "
        "any optimal Q(s,a).",
        f"Cell widths: {tuple(map(float, base.cell_widths))}. Indices start at zero.",
        axes,
        f"Action columns: {', '.join(base.ACTION_NAMES)}.",
        "Nominal actions move one cell; outward commands stay in the current cell.",
        f"Rewards: goal command {base.goal_reward!r}; free-cell command "
        f"{base.movement_reward!r}; obstacle/boundary command 0. Terminal rewards are 0.",
        f"Discount: {base.gamma!r}. Goal, obstacle, and outside cells have zero values.",
        "Local cells have Euclidean point-to-cell distance from the nominal successor "
        "center at most C. Mix the nominal next value and the local minimum "
        "with weights 1-R and R.",
        "Columns x: " + " ".join(map(str, range(shape[0]))),
        "0 = free space, 1 = obstacle.\n",
    ]
    layers = base.obstacle_mask[None] if len(shape) == 2 else base.obstacle_mask
    for z, layer in enumerate(layers):
        if len(shape) == 3:
            lines.append(f"z={z}")
        for y, row in enumerate(layer):
            lines.append(f"y={y}: " + " ".join(map(str, row.astype(int))))
    start, goal = tuple(map(int, base.start_state)), tuple(map(int, base.goal_state))
    lines.append(f"Start at {start} and move to the goal state at {goal}.")
    return "\n".join(lines) + "\n"


def grid_cell_spans(text):
    """Locate only binary cell characters, excluding legends and coordinate labels."""
    marker = "0 = free space, 1 = obstacle.\n\n"
    start = text.index(marker) + len(marker)
    end = text.index("Start at", start)
    spans, offset = [], start
    for line in text[start:end].splitlines(keepends=True):
        if line.strip() and not re.fullmatch(r"\s*z=\d+\s*", line):
            label = re.match(r"\s*y=\d+:\s*", line)
            begin = label.end() if label else 0
            cells = line[begin:]
            if not re.fullmatch(r"\s*[01](?:\s+[01])*\s*", cells):
                raise ValueError("Each grid row must contain space-separated binary cells.")
            spans.extend((offset + begin + m.start(), offset + begin + m.end())
                         for m in re.finditer(r"[01]", cells))
        offset += len(line)
    return spans


def _solve_map(mdp, conditions, tolerance, max_iterations, progress=None):
    """Solve every R/C combination in one process, reusing the map's uncertainty cache."""
    targets, residuals = [], []
    for R, C in conditions:
        agent = Tabular_Agent(mdp, R=R, C=C)
        Q, _ = agent.robust_value_iteration(
            tolerance=tolerance, max_iterations=max_iterations,
        )
        targets.append(Q)
        residuals.append(agent.diagnostics["bellman_residual"])
        if progress is not None:
            progress.update(1)
    return np.asarray(targets), residuals


def generate_q_data(
    mdps, R_values, C_values, output_path, *, n_maps=None, n_workers=1,
    tolerance=1e-8, max_iterations=10000,
):
    """Save numeric Q targets and structured parameters for each distinct map.

    mdps is an iterable of nominal MDPs. Environment construction and randomized
    map generation belong to the example, not to this shared learning code.
    Each MDP supplies its discount. Use consistent geometry, action ordering,
    rewards, and discount throughout a dataset.
    Supply n_maps for a progress total when mdps is a generator without a length.
    n_workers>1 solves independent, picklable maps in CPU processes. Map generation
    and saved record order stay unchanged; at most 2*n_workers maps are pending.
    Parallel progress advances by one map's completed Q-tables at a time.
    """
    if not isinstance(n_workers, (int, np.integer)) or n_workers < 1:
        raise ValueError("n_workers must be a positive integer.")
    n_workers = int(n_workers)
    R_values, C_values = tuple(R_values), tuple(C_values)
    if not R_values or not C_values:
        raise ValueError("Provide at least one R and C value.")
    conditions = [(R, C) for R in R_values for C in C_values]
    for R, C in conditions:
        Tabular_Agent._validate_uncertainty(R, C)
    maps, map_indices, radii, levels, targets, residuals = [], [], [], [], [], []
    shared = None
    seen = set()
    if n_maps is None and hasattr(mdps, "__len__"):
        n_maps = len(mdps)
    samples_per_map = len(conditions)
    total = None if n_maps is None else n_maps * samples_per_map
    pool = nullcontext()
    if n_workers > 1:
        pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("spawn"))
    with pool as executor, tqdm(total=total, desc="Generating Q tables", unit="sample") as progress:
        pending = deque()

        def collect(map_index, result):
            Q_tables, errors = result
            if targets and Q_tables.shape[1:] != targets[0].shape:
                raise ValueError("All training maps must have the same Q-table shape.")
            map_indices.extend([map_index] * samples_per_map)
            levels.extend(R for R, _ in conditions)
            radii.extend(C for _, C in conditions)
            targets.extend(Q_tables)
            residuals.extend(errors)
            if executor is not None:
                progress.update(samples_per_map)

        for mdp in mdps:
            parameters = mdp_data(mdp)
            if shared is None:
                shared = {name: parameters[name] for name in MDP_SHARED_FIELDS}
            elif any(not np.array_equal(shared[name], parameters[name])
                     for name in MDP_SHARED_FIELDS):
                raise ValueError(
                    "All MDPs must have the same Q-table shape, geometry, discount, and rewards."
                )
            key = tuple(parameters[name].tobytes() for name in MDP_MAP_FIELDS)
            if key in seen:
                if progress.total is not None:
                    progress.total -= samples_per_map
                    progress.refresh()
                continue
            seen.add(key)
            map_index = len(maps)
            maps.append({name: parameters[name] for name in MDP_MAP_FIELDS})
            if executor is None:
                collect(map_index, _solve_map(mdp, conditions, tolerance, max_iterations, progress))
            else:
                future = executor.submit(_solve_map, mdp, conditions, tolerance, max_iterations)
                pending.append((map_index, future))
                if len(pending) >= 2 * n_workers:
                    index, future = pending.popleft()
                    collect(index, future.result())
        for index, future in pending:
            collect(index, future.result())
    if not targets:
        raise ValueError("Provide at least one MDP.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        np.savez_compressed(
            file, **shared,
            **{name: np.concatenate([item[name] for item in maps]) for name in MDP_MAP_FIELDS},
            map_index=np.asarray(map_indices), R=np.asarray(levels), C=np.asarray(radii),
            Q=np.asarray(targets), bellman_residual=np.asarray(residuals),
        )
    return output_path


def split_samples(n_samples, validation_fraction=0.1, seed=42):
    """Split indices reproducibly into training and validation sets."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1).")
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("Training and validation require at least two samples.")
    indices = np.random.default_rng(seed).permutation(n_samples)
    count = max(1, min(n_samples - 1, int(np.ceil(n_samples * validation_fraction))))
    return indices[count:], indices[:count]
