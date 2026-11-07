"""Dimension-independent MDP descriptions and robust value-iteration training data."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from multiprocessing import get_context
from pathlib import Path
import re

import numpy as np
from tqdm import tqdm

from Tabular_Agent import Tabular_Agent


def format_prompt(description):
    """Describe the map once; planning parameters are separate numerical inputs."""
    return (
        "### Instruction:\n"
        f"{description.rstrip()}\n"
        "Predict the Q-table.\n"
    )


def parse_grid_description(description):
    """Read a full binary grid in increasing x, then y, then higher dimensions."""
    shape_match = re.search(r"grid with size (\d+(?:x\d+)+)", description)
    endpoints = re.search(r"Start at \(([^)]+)\).*goal state at \(([^)]+)\)", description)
    if shape_match is None or endpoints is None:
        raise ValueError("The description must specify grid dimensions, start, and goal.")
    shape = tuple(map(int, shape_match[1].split("x")))
    start, goal = (tuple(map(int, text.split(","))) for text in endpoints.groups())
    if len(start) != len(shape) or len(goal) != len(shape):
        raise ValueError("Start and goal dimensions must match the grid.")
    marker = "0 = free space, 1 = obstacle.\n\n"
    if marker not in description:
        raise ValueError("The description must contain a complete binary grid.")
    grid = description.split(marker, 1)[1].split("Start at", 1)[0]
    entries = grid.split()
    if len(entries) != np.prod(shape) or any(value not in ("0", "1") for value in entries):
        raise ValueError("The description must contain one binary entry per cell.")
    cells = np.asarray(entries, dtype=int).reshape(shape[::-1])
    obstacles = [tuple(map(int, cell[::-1])) for cell in np.argwhere(cells)]
    return shape, start, goal, obstacles


def _solve_map(mdp, conditions, tolerance, max_iterations, progress=None):
    """Solve every R/C combination in one process, reusing the map's uncertainty cache."""
    agent = Tabular_Agent(mdp, lr_init=0.5)
    targets, residuals = [], []
    for R, C in conditions:
        Q, _ = agent.value_iteration(R=R, C=C, tolerance=tolerance, max_iterations=max_iterations)
        targets.append(Q)
        residuals.append(agent.value_iteration_diagnostics["bellman_residual"])
        if progress is not None:
            progress.update(1)
    return np.asarray(targets), residuals


def generate_q_data(
    mdps, R_values, C_values, output_path, *, n_maps=None, n_workers=1,
    tolerance=1e-8, max_iterations=10000,
):
    """Save numeric Q targets, storing each distinct map description only once.

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
    descriptions, map_indices, radii, levels, targets, residuals = [], [], [], [], [], []
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
            description = mdp.unwrapped.q_transformer_context()
            if description in seen:
                if progress.total is not None:
                    progress.total -= samples_per_map
                    progress.refresh()
                continue
            seen.add(description)
            map_index = len(descriptions)
            descriptions.append(description)
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
            file, descriptions=np.asarray(descriptions),
            map_index=np.asarray(map_indices), R=np.asarray(levels), C=np.asarray(radii),
            Q=np.asarray(targets), bellman_residual=np.asarray(residuals),
        )
    return output_path


def split_samples(n_samples, validation_fraction=0.1, seed=42):
    """Match the original dataset's seeded random split of individual samples."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1).")
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("Training and validation require at least two samples.")
    indices = np.random.default_rng(seed).permutation(n_samples)
    count = max(1, min(n_samples - 1, int(np.ceil(n_samples * validation_fraction))))
    return indices[count:], indices[:count]
