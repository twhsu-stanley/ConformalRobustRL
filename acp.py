"""Episode-level adaptive conformal calibration with an exact order statistic."""

from collections import deque
import math

import numpy as np


class ACP:
    """Preserve the paper's unconstrained delta update and extended quantile tails.

    A nonpositive rank gives an empty prediction (radius -infinity); a rank larger
    than the calibration window gives an unbounded prediction (+infinity). Empty
    predictions use radius zero for planning, while ACP still records miscoverage
    against its original empty prediction. Both coverage quantities must be logged.
    """

    def __init__(self, scores, target_failure=0.1, initial_failure=None, eta=0.01, window_size=100):
        initial_failure = target_failure if initial_failure is None else initial_failure
        if not 0 < target_failure < 1 or not 0 < initial_failure < 1:
            raise ValueError("Target and initial failure probabilities must be in (0, 1).")
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("ACP learning rate eta must be positive and finite.")
        if not isinstance(window_size, (int, np.integer)) or window_size < 1:
            raise ValueError("Calibration window size must be a positive integer.")
        scores = [self._validate_score(score) for score in scores]
        if not scores:
            raise ValueError("An initial calibration dataset is required.")
        self.scores = deque(scores, maxlen=int(window_size))
        self.target_failure = float(target_failure)
        self.delta = float(initial_failure)
        self.initial_delta = self.delta
        self.eta = float(eta)
        self.n_updates = 0
        self.n_miscovered = 0

    @staticmethod
    def _validate_score(score):
        if not np.isfinite(score) or score < 0:
            raise ValueError("Calibration scores must be finite and nonnegative.")
        return float(score)

    def quantile(self):
        rank = math.ceil((1 - self.delta) * (len(self.scores) + 1))
        if rank <= 0:
            return -np.inf
        if rank > len(self.scores):
            return np.inf
        return float(np.partition(np.asarray(self.scores), rank - 1)[rank - 1])

    @staticmethod
    def planning_radius(conformal_radius):
        if np.isnan(conformal_radius):
            raise ValueError("Conformal radius cannot be NaN.")
        return max(0.0, float(conformal_radius))

    def update(self, score, predicted_radius=None):
        score = self._validate_score(score)
        predicted_radius = self.quantile() if predicted_radius is None else predicted_radius
        if np.isnan(predicted_radius):
            raise ValueError("Predicted radius cannot be NaN.")
        miscovered = int(score > predicted_radius)
        previous_delta = self.delta
        self.delta += self.eta * (self.target_failure - miscovered)
        self.scores.append(score)
        self.n_updates += 1
        self.n_miscovered += miscovered
        return {
            "miscoverage": miscovered, "delta_before": previous_delta, "delta_after": self.delta,
        }
