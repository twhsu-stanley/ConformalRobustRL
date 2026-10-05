"""General evaluation-return utilities for tabular reinforcement learning."""

import matplotlib.pyplot as plt
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

def plot_evaluation_return(evaluation_return):
    evaluation_return_mean, evaluation_return_std, evaluation_return_np = (
        calc_evaluation_return_mean_std(evaluation_return)
    )

    plt.figure()
    plt.plot(evaluation_return_mean)
    plt.fill_between(
        range(len(evaluation_return_mean)), evaluation_return_mean - evaluation_return_std,
        evaluation_return_mean + evaluation_return_std, alpha=0.2,
    )
    plt.grid()
    plt.xlabel("Culmulative Time Steps")
    plt.ylabel("Evaluation Return V(initial state)")
    plt.title("Evaluation Return over Culmulative Time Steps")

    plt.figure()
    for i in range(len(evaluation_return)):
        plt.plot(evaluation_return_np[i,:])
    plt.grid()
    plt.xlabel("Culmulative Time Steps")
    plt.ylabel("Evaluation Return V(initial state)")
    plt.title("Evaluation Return over Culmulative Time Steps")
    plt.show()
