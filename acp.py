"""Episode-level adaptive conformal calibration with an exact order statistic."""

from collections import deque
import math

import numpy as np


class ACP:
    """Calibrate C_hat using target delta and the unconstrained adaptive delta_k.

    quantile() returns a nonnegative radius ready for planning. A nonpositive rank
    returns zero; a rank above the calibration window returns positive infinity.
    """

    def __init__(self, scores, delta=0.1, delta_1=None, eta=0.01, window_size=100, quantile_max=None):
        delta_1 = delta if delta_1 is None else delta_1
        if not 0 < delta < 1 or not 0 < delta_1 < 1:
            raise ValueError("Target and initial failure probabilities must be in (0, 1).")
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("ACP learning rate eta must be positive and finite.")
        if not isinstance(window_size, (int, np.integer)) or window_size < 1:
            raise ValueError("Calibration window size must be a positive integer.")
        scores = [self._validate_score(score) for score in scores]
        if not scores:
            raise ValueError("An initial calibration dataset is required.")
        self.scores = deque(scores, maxlen=int(window_size))
        self.delta = float(delta)
        self.delta_k = float(delta_1)
        self.delta_1 = self.delta_k
        self.eta = float(eta)
        self.n_updates = 0
        self.quantile_max = float(quantile_max) if quantile_max is not None else np.inf
        if self.quantile_max <= 0:
            raise ValueError("quantile_max must be positive.")

    @staticmethod
    def _validate_score(score):
        if not np.isfinite(score) or score < 0:
            raise ValueError("Calibration scores must be finite and nonnegative.")
        return float(score)

    def quantile(self):
        rank = math.ceil((1 - self.delta_k) * (len(self.scores) + 1))
        if rank <= 0:
            return 0.0
        if rank > len(self.scores):
            return self.quantile_max
        C_hat = float(np.partition(np.asarray(self.scores), rank - 1)[rank - 1])
        if np.isnan(C_hat):
            raise ValueError("Conformal radius cannot be NaN.")
        return max(0.0, C_hat)

    def update(self, score, predicted_radius=None):
        score = self._validate_score(score)
        predicted_radius = self.quantile() if predicted_radius is None else predicted_radius
        if np.isnan(predicted_radius):
            raise ValueError("Predicted radius cannot be NaN.")
        miscovered = int(score > predicted_radius)
        previous_delta = self.delta_k
        self.delta_k += self.eta * (self.delta - miscovered)
        self.scores.append(score)
        self.n_updates += 1
        return {"miscoverage": miscovered, "delta_k": previous_delta, "delta_next": self.delta_k}
