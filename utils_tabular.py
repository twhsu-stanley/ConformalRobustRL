"""Compute statistics for saved tabular learning curves."""

import numpy as np

def calc_evaluation_return_mean_std(evaluation_return):
    min_len = float('inf')
    for i in range(len(evaluation_return)):
        if len(evaluation_return[i]) < min_len:
            min_len = len(evaluation_return[i])

    evaluation_return_np = np.zeros((len(evaluation_return), min_len))
    for i in range(len(evaluation_return)):
        evaluation_return_np[i,:] = np.array(evaluation_return[i][:min_len])

    evaluation_return_mean = np.mean(evaluation_return_np, axis=0)
    evaluation_return_std = np.std(evaluation_return_np, axis=0)
    return evaluation_return_mean, evaluation_return_std, evaluation_return_np
