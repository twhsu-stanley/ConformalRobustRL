"""One-time, verified conversion of description-based datasets to numeric MDP parameters."""

import hashlib
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from tqdm import tqdm

from q_transformer.data import MDP_MAP_FIELDS, MDP_SHARED_FIELDS
from q_transformer.data import format_prompt, grid_cell_spans, mdp_data, mdp_parameters


RECORD_FIELDS = ("map_index", "R", "C", "Q", "bellman_residual")


def _legacy_parameters(description):
    """Read the old binary-grid description; normal training no longer parses prose."""
    shape_match = re.search(r"grid with size (\d+(?:x\d+)+)", description)
    endpoints = re.search(r"Start at \(([^)]+)\).*goal state at \(([^)]+)\)", description)
    if shape_match is None or endpoints is None:
        raise ValueError("The legacy description must specify dimensions, start, and goal.")
    shape = tuple(map(int, shape_match[1].split("x")))
    start, goal = (tuple(map(int, text.split(","))) for text in endpoints.groups())
    entries = [description[a:b] for a, b in grid_cell_spans(description)]
    if len(start) != len(shape) or len(goal) != len(shape) or len(entries) != np.prod(shape):
        raise ValueError("The legacy grid and endpoints must match its dimensions.")
    mask = np.asarray(entries, dtype=int).reshape(shape[::-1])
    return dict(
        grid_shape=shape, start_cell=start, goal_cell=goal,
        obstacles=[tuple(map(int, cell[::-1])) for cell in np.argwhere(mask)],
    )


def _entry_hash(archive, name):
    digest = hashlib.sha256()
    with archive.open(name) as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def convert_q_data(path, mdp_class, *, bounds, gamma, movement_reward, goal_reward):
    """Convert in place only after validating maps, prompts, and all original record bytes.

    Supply the original geometry, discount, and rewards: legacy descriptions did
    not store them. Q targets are streamed unchanged, never regenerated. A temporary
    file in the same directory is atomically substituted after verification. An
    already-converted dataset is left alone. No checkpoints are read or modified.
    """
    path = Path(path).resolve(strict=True)
    with np.load(path, allow_pickle=False) as data:
        if "descriptions" not in data.files:
            required = set(MDP_SHARED_FIELDS + MDP_MAP_FIELDS + RECORD_FIELDS)
            if not required.issubset(data.files):
                raise ValueError("The file is not a structured Q dataset.")
            return path
        if set(data.files) != {"descriptions", *RECORD_FIELDS}:
            raise ValueError("Unexpected legacy fields; conversion would discard data.")
        descriptions, indices = data["descriptions"], data["map_index"]
        levels, radii = data["R"], data["C"]
    if not len(descriptions) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("The dataset must contain maps and integer map indices.")
    if indices.ndim != 1 or np.any((indices < 0) | (indices >= len(descriptions))):
        raise ValueError("Every record must reference a saved map.")
    if levels.shape != indices.shape or radii.shape != indices.shape:
        raise ValueError("Each record must have R and C values.")
    if len(bounds) not in (2, 3):
        raise ValueError("Supply one workspace bound per dimension, for a 2-D or 3-D grid.")
    settings = dict(
        **dict(zip(("x_max", "y_max", "z_max"), bounds)),
        gamma=gamma, movement_reward=movement_reward, goal_reward=goal_reward,
    )
    maps, shared = [], None
    for description in descriptions:
        mdp = mdp_class(**settings, **_legacy_parameters(str(description)))
        parameters = mdp_data(mdp)
        if shared is None:
            shared = {name: parameters[name] for name in MDP_SHARED_FIELDS}
        elif any(not np.array_equal(shared[name], parameters[name]) for name in MDP_SHARED_FIELDS):
            raise ValueError("Legacy maps must share geometry, discount, and rewards.")
        maps.append({name: parameters[name] for name in MDP_MAP_FIELDS})
    metadata = {
        **shared,
        **{name: np.concatenate([item[name] for item in maps]) for name in MDP_MAP_FIELDS},
    }
    with NamedTemporaryFile(dir=path.parent, prefix=path.stem + "_", suffix=".npz",
                            delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(temporary_path, **metadata)
        hashes = {}
        with ZipFile(path) as source, ZipFile(temporary_path, "a", ZIP_DEFLATED) as destination:
            for field in RECORD_FIELDS:
                name, digest = field + ".npy", hashlib.sha256()
                with (
                    source.open(name) as reader,
                    destination.open(name, "w", force_zip64=True) as writer,
                ):
                    for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                        writer.write(chunk)
                hashes[name] = digest.digest()
        with ZipFile(temporary_path) as archive:
            for name, expected in hashes.items():
                if _entry_hash(archive, name) != expected:
                    raise ValueError(f"Record bytes changed during conversion: {name}.")
        with np.load(temporary_path, allow_pickle=False) as data:
            converted = {name: data[name] for name in metadata}
        for name in metadata:
            if not np.array_equal(metadata[name], converted[name]):
                raise ValueError(f"MDP parameters changed during serialization: {name}.")
        for index, description in enumerate(tqdm(descriptions, desc="Verifying converted maps")):
            original = mdp_class(**settings, **_legacy_parameters(str(description)))
            restored = mdp_class(**mdp_parameters(converted, index))
            for name in (
                "states", "state_coordinates", "nominal_transitions", "rewards",
                "terminal_mask", "boundary_commands",
            ):
                if not np.array_equal(getattr(original, name), getattr(restored, name)):
                    raise ValueError(f"Map {index} changed: {name}.")
            for row in np.flatnonzero(indices == index):
                if format_prompt(original, levels[row], radii[row]) != format_prompt(
                    restored, levels[row], radii[row],
                ):
                    raise ValueError(f"Prompt {row} changed during conversion.")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path
