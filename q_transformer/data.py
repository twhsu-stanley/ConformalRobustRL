"""Dimension-independent MDP descriptions and robust value-iteration training data."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from Tabular_Agent import Tabular_Agent


def format_prompt(description, R, C):
    """One format for both training and deployment; R and C are inputs, never labels."""
    Tabular_Agent._validate_uncertainty(R, C)
    radius = "infinity" if np.isposinf(C) else repr(float(C))
    return (
        "### Instruction:\n"
        f"{description.rstrip()}\n"
        f"Predict the Q-table for R = {float(R)!r} and C = {radius}.\n"
        "Do NOT include extra text or explanations.\n\n"
    )


def generate_q_data(
    mdps, R_values, C_values, gamma, output_path, *, n_maps=None,
    tolerance=1e-8, max_iterations=10000,
):
    """Save numeric Q targets, storing each distinct map description only once.

    mdps is an iterable of nominal MDPs. Environment construction and randomized
    map generation belong to the example, not to this shared learning code.
    Use consistent geometry, action ordering, and rewards throughout a dataset.
    Supply n_maps for a progress total when mdps is a generator without a length.
    """
    R_values, C_values = tuple(R_values), tuple(C_values)
    if not R_values or not C_values:
        raise ValueError("Provide at least one R and C value.")
    for R in R_values:
        for C in C_values:
            Tabular_Agent._validate_uncertainty(R, C)
    descriptions, map_indices, radii, levels, targets, residuals = [], [], [], [], [], []
    seen = set()
    if n_maps is None and hasattr(mdps, "__len__"):
        n_maps = len(mdps)
    samples_per_map = len(R_values) * len(C_values)
    total = None if n_maps is None else n_maps * samples_per_map
    with tqdm(total=total, desc="Generating Q tables", unit="sample") as progress:
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
                    progress.update(1)
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
