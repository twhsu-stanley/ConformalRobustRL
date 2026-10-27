"""Dimension-independent MDP descriptions and robust value-iteration training data."""

from pathlib import Path

import numpy as np

from Tabular_Agent import Tabular_Agent


def format_prompt(description, R, C):
    """One format for both training and deployment; R and C are inputs, never labels."""
    Tabular_Agent._validate_uncertainty(R, C)
    radius = "infinity" if np.isposinf(C) else repr(float(C))
    return (
        "### Instruction:\n"
        f"{description.rstrip()}\n"
        f"Use the localized (R,C)-contamination model with R = {float(R)!r} "
        f"and C = {radius}.\n"
        "Output the Q Table only.\n"
        "Do NOT include extra text or explanations.\n\n"
        "### Response:\n"
    )


def generate_q_data(
    mdps, R_values, C_values, gamma, output_path, *, tolerance=1e-8, max_iterations=10000,
):
    """Save numeric Q targets, storing each distinct map description only once.

    mdps is an iterable of nominal MDPs. Environment construction and randomized
    map generation belong to the example, not to this shared learning code.
    Use consistent geometry, action ordering, and rewards throughout a dataset.
    """
    R_values, C_values = tuple(R_values), tuple(C_values)
    if not R_values or not C_values:
        raise ValueError("Provide at least one R and C value.")
    for R in R_values:
        for C in C_values:
            Tabular_Agent._validate_uncertainty(R, C)
    descriptions, map_indices, radii, levels, targets, residuals = [], [], [], [], [], []
    seen = set()
    for mdp in mdps:
        description = mdp.unwrapped.q_transformer_context()
        if description in seen:
            continue
        seen.add(description)
        map_index = len(descriptions)
        descriptions.append(description)
        agent = Tabular_Agent(mdp, gamma=gamma, lr_init=0.5)
        if targets and agent.Q.shape != targets[0].shape:
            raise ValueError("All training maps must have the same Q-table shape.")
        for R in R_values:
            for C in C_values:
                Q, _ = agent.value_iteration(
                    R=R, C=C, tolerance=tolerance, max_iterations=max_iterations,
                )
                map_indices.append(map_index)
                levels.append(R)
                radii.append(C)
                targets.append(Q)
                residuals.append(agent.value_iteration_diagnostics["bellman_residual"])
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


def split_by_map(map_indices, validation_fraction=0.2, seed=7):
    """Keep every (R,C) variant of a map entirely in one partition."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1).")
    map_indices = np.asarray(map_indices)
    maps = np.unique(map_indices)
    if len(maps) < 2:
        raise ValueError("Training and validation require at least two distinct maps.")
    maps = np.random.default_rng(seed).permutation(maps)
    count = max(1, min(len(maps) - 1, int(np.ceil(len(maps) * validation_fraction))))
    validation = np.isin(map_indices, maps[:count])
    return np.flatnonzero(~validation), np.flatnonzero(validation)
