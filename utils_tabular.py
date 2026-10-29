"""Compute statistics for saved tabular learning curves."""

import numpy as np

def calc_evaluation_return_mean_std(evaluation_return):
    """Average full curves, holding each stopped learner at its final value."""
    curves = [np.asarray(curve, dtype=float) for curve in evaluation_return]
    if not curves or any(curve.ndim != 1 or len(curve) == 0 for curve in curves):
        raise ValueError("Provide nonempty one-dimensional learning curves.")
    max_len = max(map(len, curves))
    evaluation_return_np = np.stack([
        np.pad(curve, (0, max_len - len(curve)), mode="edge") for curve in curves
    ])
    evaluation_return_mean = np.mean(evaluation_return_np, axis=0)
    evaluation_return_std = np.std(evaluation_return_np, axis=0)
    return evaluation_return_mean, evaluation_return_std, evaluation_return_np
