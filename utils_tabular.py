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
    evaluation_return_mean, evaluation_return_std, evaluation_return_np = calc_evaluation_return_mean_std(evaluation_return)

    plt.figure()
    plt.plot(evaluation_return_mean)
    plt.fill_between(range(len(evaluation_return_mean)), evaluation_return_mean - evaluation_return_std, evaluation_return_mean + evaluation_return_std, alpha=0.2)
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


def plot_robot_motion(mdp, Q, policy, trajectory=None, title="Robot motion"):
    """Plot values, direction commands, and an optional continuous trajectory."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    extent = (0, mdp.bounds[0], 0, mdp.bounds[1])
    values = np.max(Q, axis=1).reshape(mdp.ny, mdp.nx)
    image = axes[0].imshow(values, origin="lower", extent=extent, cmap="YlGnBu")
    fig.colorbar(image, ax=axes[0], label="Learned robust value")
    for axis in axes:
        for state in range(mdp.n_state):
            ix, iy = mdp.state_to_cell(state)
            color = "0.25" if mdp.obstacle_mask[iy, ix] else "none"
            if state == mdp.start_state:
                color = "lightskyblue"
            elif state == mdp.goal_state:
                color = "lightgreen"
            rectangle = plt.Rectangle(
                (ix * mdp.cell_widths[0], iy * mdp.cell_widths[1]), *mdp.cell_widths,
                facecolor=color, edgecolor="0.6", linewidth=0.7,
            )
            axis.add_patch(rectangle)
        axis.set(xlim=extent[:2], ylim=extent[2:], xlabel="x", ylabel="y", aspect="equal")
        axis.set_xticks(np.linspace(0, mdp.bounds[0], mdp.nx + 1))
        axis.set_yticks(np.linspace(0, mdp.bounds[1], mdp.ny + 1))
        for state, label in ((mdp.start_state, "S"), (mdp.goal_state, "G")):
            x, y = mdp.coordinates(state)
            axis.text(x, y, label, ha="center", va="center", weight="bold")
    for state in mdp.trainable_states:
        x, y = mdp.coordinates(state)
        value_y = y - 0.2 * mdp.cell_widths[1] if state == mdp.start_state else y
        axes[0].text(x, value_y, f"{np.max(Q[state]):.2f}", ha="center", va="center", fontsize=7)
        direction = np.asarray(mdp.ACTION_DIRECTIONS[int(policy[state])])
        dx, dy = direction * mdp.cell_widths * 0.28
        axes[1].arrow(
            x, y, dx, dy, head_width=min(mdp.cell_widths) * 0.12,
            length_includes_head=True, color="tab:blue",
        )
    if trajectory is not None:
        trajectory = np.asarray(trajectory)
        axes[1].plot(trajectory[:, 0], trajectory[:, 1], "o-", color="tab:orange", markersize=3)
    axes[0].set_title("Value function")
    axes[1].set_title("Policy and deployment trajectory")
    fig.suptitle(title)
    return fig


def plot_conformal_history(history, target_failure):
    """Keep raw conformal coverage distinct from the radius used by the planner."""
    if not history:
        raise ValueError("At least one deployment episode is required for plotting.")
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    episodes = np.arange(1, len(history) + 1)
    axes = axes.ravel()
    axes[0].plot(episodes, [item["R_used"] for item in history], label="R used for training")
    axes[0].plot(episodes, [item["R_next"] for item in history], label="Updated mismatch estimate")
    axes[0].set(title="Discrete transition mismatches", ylabel="Probability", ylim=(0, 1))
    scores = np.asarray([item["score"] for item in history])
    radii = np.asarray([item["conformal_radius"] for item in history])
    finite_radii = np.where(np.isfinite(radii), radii, np.nan)
    ceiling = max(1.0, np.max(scores), np.max(radii[np.isfinite(radii)], initial=0)) * 1.1
    axes[1].plot(episodes, scores, label="Episode maximum deviation")
    axes[1].plot(episodes, finite_radii, label="Conformal radius")
    if np.any(np.isposinf(radii)):
        axes[1].scatter(
            episodes[np.isposinf(radii)], np.full(np.isposinf(radii).sum(), ceiling),
            marker="^", label="Unbounded prediction",
        )
    if np.any(np.isneginf(radii)):
        axes[1].scatter(
            episodes[np.isneginf(radii)], np.zeros(np.isneginf(radii).sum()),
            marker="x", label="Empty prediction; planner radius 0",
        )
    axes[1].set(title="Continuous uncertainty", ylabel="Distance")
    axes[2].plot(episodes, [item["delta_before"] for item in history], label="Adaptive delta")
    axes[2].axhline(target_failure, color="black", linestyle="--", label="Target failure")
    axes[2].set(title="ACP update", ylabel="Failure parameter")
    miscoverage = np.asarray([item["miscoverage"] for item in history])
    planned = np.asarray([item["planned_miscoverage"] for item in history])
    axes[3].plot(episodes, 1 - np.cumsum(miscoverage) / episodes, label="Conformal coverage")
    axes[3].plot(episodes, 1 - np.cumsum(planned) / episodes, label="Planning-radius coverage")
    axes[3].axhline(1 - target_failure, color="black", linestyle="--", label="Coverage target")
    axes[3].set(title="Cumulative empirical coverage", ylabel="Fraction", ylim=(0, 1.05))
    returns = [item["discounted_return"] for item in history]
    axes[4].plot(episodes, returns, label="Discounted r(s,a)")
    for key, marker, color in (("goal_reached", "o", "green"), ("collision", "x", "red")):
        indices = np.flatnonzero([item[key] for item in history])
        axes[4].scatter(episodes[indices], np.asarray(returns)[indices], marker=marker,
                        color=color, label=key.replace("_", " "))
    axes[4].set(title="Deployment outcomes", ylabel="Return")
    residuals = [max(item["training"]["bellman_residual"], 1e-15) for item in history]
    axes[5].semilogy(episodes, residuals, label="Robust Bellman residual")
    axes[5].set(title="Finite training accuracy", ylabel="Residual")
    for axis in axes:
        axis.set_xlabel("Deployment episode")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    return fig
