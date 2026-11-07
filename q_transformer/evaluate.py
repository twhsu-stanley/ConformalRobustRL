"""Shared prediction and paired value-iteration evaluation for Q-transformers."""

from concurrent.futures import ProcessPoolExecutor
import csv
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np

from q_transformer.data import split_samples
from Tabular_Agent import Tabular_Agent


def _evaluate_map(
    mdp, indices, levels, radii, targets, predictions, seconds, tolerance,
    benchmark_offset, benchmark_stride,
):
    """Run the actual solver, alternating order and sharing the same populated MDP cache."""
    terminal = mdp.terminal_mask
    records = []
    for offset, index in enumerate(indices):
        R, C = float(levels[offset]), float(radii[offset])
        target, raw = targets[offset], predictions[offset]
        predicted = raw.copy()
        predicted[terminal] = 0.0
        error = predicted - target
        active = ~terminal
        optimal = target.max(axis=1)
        actions = predicted.argmax(axis=1)
        gaps = optimal - target[np.arange(mdp.n_state), actions]
        positive = active & (optimal > 1e-6)
        record = {
            "index": int(index), "R": R, "C": C,
            "raw_mse": float(np.mean((raw - target)**2)),
            "mse": float(np.mean(error**2)), "mae": float(np.mean(np.abs(error))),
            "free_mse": float(np.mean(error[active]**2)),
            "zero_mse": float(np.mean(target**2)),
            "max_error": float(np.max(np.abs(error))),
            "terminal_mse": float(np.mean(raw[terminal]**2)),
            "negative_free_fraction": float(np.mean(predicted[active] < 0)),
            "policy_optimal_fraction": float(np.mean(gaps[positive] <= 1e-6)),
            "mean_action_gap": float(np.mean(gaps[positive])),
            "inference_seconds": float(seconds[offset]),
        }
        records.append(record)
        if offset % benchmark_stride != benchmark_offset:
            continue
        reference = Tabular_Agent(mdp, 0.5, R=R, C=C)
        reference.initialize_q(predicted)
        record["initial_bellman_residual"] = reference.robust_bellman_residual()
        # The residual above populates every neighborhood used by either timed solve.
        solutions = {}
        order = ("zero", "warm") if offset % 2 == 0 else ("warm", "zero")
        for name in order:
            agent = Tabular_Agent(mdp, 0.5, R=R, C=C)
            initial_Q = None if name == "zero" else predicted
            start = perf_counter()
            Q, _ = agent.value_iteration(initial_Q=initial_Q, tolerance=tolerance)
            record[f"{name}_seconds"] = perf_counter() - start
            record[f"{name}_iterations"] = agent.value_iteration_diagnostics["iterations"]
            record[f"{name}_residual"] = agent.value_iteration_diagnostics["bellman_residual"]
            solutions[name] = Q
            if not np.all(Q[terminal] == 0):
                raise AssertionError("Terminal Q-values must remain zero.")
        record["solution_difference"] = float(np.max(np.abs(solutions["warm"] - solutions["zero"])))
        record["teacher_difference"] = float(np.max(np.abs(solutions["zero"] - target)))
        if record["solution_difference"] > 2 * tolerance / (1 - mdp.gamma):
            raise AssertionError("The two starts did not converge to the same fixed point.")
        if record["teacher_difference"] > 2 * tolerance / (1 - mdp.gamma):
            raise AssertionError("Saved labels do not match the current MDP/solver.")
    benchmark = [row for row in records if "zero_iterations" in row]
    zero = np.mean([row["zero_iterations"] for row in benchmark])
    warm = np.mean([row["warm_iterations"] for row in benchmark])
    print(f"Map at sample {indices[0]}: mean iterations zero={zero:.1f}, warm={warm:.1f}",
          flush=True)
    return records


def _summarize(records):
    """Aggregate squared errors before taking square roots, as in training validation."""
    summary = {name: float(np.mean([row[name] for row in records])) for name in (
        "mae", "policy_optimal_fraction", "mean_action_gap",
        "negative_free_fraction", "inference_seconds",
    )}
    summary["mean_max_error"] = float(np.mean([row["max_error"] for row in records]))
    summary["max_error"] = max(row["max_error"] for row in records)
    benchmark = [row for row in records if "zero_iterations" in row]
    summary.update({name: float(np.mean([row[name] for row in benchmark])) for name in (
        "initial_bellman_residual",
        "zero_iterations", "warm_iterations", "zero_seconds", "warm_seconds",
    )})
    for name in ("raw", "free", "zero", "terminal"):
        summary[f"{name}_rmse"] = float(np.sqrt(np.mean([row[f"{name}_mse"] for row in records])))
    summary["rmse"] = float(np.sqrt(np.mean([row["mse"] for row in records])))
    summary["samples"] = len(records)
    summary["solver_pairs"] = len(benchmark)
    summary["fewer_iterations_fraction"] = float(np.mean([
        row["warm_iterations"] < row["zero_iterations"] for row in benchmark
    ]))
    summary["iteration_speedup"] = summary["zero_iterations"] / summary["warm_iterations"]
    summary["solver_speedup"] = summary["zero_seconds"] / summary["warm_seconds"]
    summary["benchmark_inference_seconds"] = float(np.mean([
        row["inference_seconds"] for row in benchmark
    ]))
    total_warm = summary["warm_seconds"] + summary["benchmark_inference_seconds"]
    summary["including_inference_speedup"] = summary["zero_seconds"] / total_warm
    summary["max_solution_difference"] = max(row["solution_difference"] for row in benchmark)
    summary["max_teacher_difference"] = max(row["teacher_difference"] for row in benchmark)
    return summary


def _plot_evaluation(records, summary, output_dir):
    """Plot prediction accuracy, paired sweep counts, and runtime."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    conditions = summary["by_condition"]
    for C in sorted({row["C"] for row in conditions}):
        rows = [row for row in conditions if row["C"] == C]
        axes[0].plot([row["R"] for row in rows], [row["rmse"] for row in rows],
                     marker="o", label=f"C={C:g}")
    axes[0].set(xlabel="R", ylabel="Q-table RMSE", title="Held-out prediction error")
    axes[0].legend()
    benchmark = [row for row in records if "zero_iterations" in row]
    zero = np.array([row["zero_iterations"] for row in benchmark])
    warm = np.array([row["warm_iterations"] for row in benchmark])
    axes[1].scatter(zero, warm, alpha=0.5, s=20)
    limit = max(zero.max(), warm.max()) * 1.05
    axes[1].plot([0, limit], [0, limit], "k--", label="Equal iterations")
    axes[1].set(xlabel="Zero-start iterations", ylabel="Warm-start iterations",
                title="Same Bellman tolerance", xlim=(0, limit), ylim=(0, limit))
    axes[1].legend()
    average = summary["overall"]
    axes[2].bar(["Zero", "Warm", "Warm +\ninference"], [
        average["zero_seconds"], average["warm_seconds"],
        average["warm_seconds"] + average["benchmark_inference_seconds"],
    ])
    axes[2].set(ylabel="Mean CPU seconds per solve", title="Cached-MDP runtime")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(output_dir / "prediction_evaluation.png", dpi=180)
    plt.close(fig)


def evaluate_q_transformer(
    dataset_path, checkpoint_path, mdp_from_description, *, n_maps=10, seed=42,
    validation_fraction=0.1, tolerance=1e-8, n_workers=2, cpu_threads=4,
):
    """Evaluate held-out Q predictions and balance solver pairs across R/C conditions."""
    import torch
    from tqdm import tqdm

    from q_transformer.model import QTransformerPredictor

    checkpoint_path = Path(checkpoint_path)
    output_dir = checkpoint_path.parent
    torch.set_num_threads(cpu_threads)
    with np.load(dataset_path, allow_pickle=False) as data:
        descriptions, map_indices, levels, radii = (
            data["descriptions"], data["map_index"], data["R"], data["C"],
        )
        maps = np.unique(map_indices)
        _, validation = split_samples(len(maps), validation_fraction, seed)
        chosen = maps[validation[:n_maps]]
        selected = np.flatnonzero(np.isin(map_indices, chosen))
        targets = np.asarray(data["Q"][selected], dtype=float)
    start = perf_counter()
    predictor = QTransformerPredictor(checkpoint_path)
    model_load_seconds = perf_counter() - start
    device = predictor.device
    print(f"Device={device}; maps={len(chosen)}; predictions={len(selected)}; "
          f"tolerance={tolerance}")
    # Exclude one-time kernel initialization from per-prediction deployment timing.
    predictor(mdp_from_description(str(descriptions[chosen[0]])), levels[selected[0]],
              radii[selected[0]])
    futures = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("spawn")) as pool:
        with tqdm(total=len(selected), desc="Q predictions") as progress:
            for map_order, map_index in enumerate(chosen):
                mdp = mdp_from_description(str(descriptions[map_index]))
                if mdp.q_transformer_context() != str(descriptions[map_index]):
                    raise ValueError("Reconstructed MDP does not match its saved description.")
                rows = np.flatnonzero(map_indices[selected] == map_index)
                predictions, seconds = [], []
                for row in rows:
                    index = selected[row]
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    start = perf_counter()
                    Q = predictor(mdp, levels[index], radii[index])
                    predictions.append(Q)
                    seconds.append(perf_counter() - start)
                    progress.update(1)
                benchmark_stride = min(len(chosen), len(rows))
                futures.append(pool.submit(
                    _evaluate_map, mdp, selected[rows],
                    levels[selected[rows]], radii[selected[rows]], targets[rows],
                    np.asarray(predictions), seconds, tolerance,
                    map_order % benchmark_stride, benchmark_stride,
                ))
        records = []
        for future in tqdm(futures, desc="Value-iteration map comparisons"):
            records.extend(future.result())
    records.sort(key=lambda row: row["index"])
    summary = {
        "checkpoint": str(checkpoint_path), "device": str(device), "seed": seed,
        "validation_fraction": validation_fraction, "map_indices": chosen.tolist(),
        "tolerance": tolerance, "model_load_seconds": model_load_seconds,
        "timing_note": f"Solver pairs share warm caches; {n_workers} workers run concurrently.",
        "overall": _summarize(records),
        "by_condition": [
            {"R": R, "C": C, **_summarize([
                row for row in records if row["R"] == R and row["C"] == C
            ])} for R, C in sorted({(row["R"], row["C"]) for row in records})
        ],
    }
    with (output_dir / "evaluation.csv").open("w", newline="") as file:
        columns = list(dict.fromkeys(key for row in records for key in row))
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2))
    _plot_evaluation(records, summary, output_dir)
    print(json.dumps(summary["overall"], indent=2))
    print(f"Saved evaluation.csv, evaluation.json, and prediction_evaluation.png in {output_dir}")
    return summary
