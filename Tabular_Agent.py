import numpy as np


class Tabular_Agent:
    """Hold a robust planning problem and the latest tabular solver's results.

    The environment and R/C define the shared problem. Algorithm-specific settings
    belong to solver calls; every call starts a fresh run from initial_Q or zeros.
    diagnostics describes the latest run, including failed convergence attempts.
    """

    def __init__(self, env, R=0.0, C=0.0):
        self.env = env
        for name in (
            "reward", "terminal_mask", "states", "state_index", "values_at", "worst_case_value",
            "nominal_next_state", "trainable_states", "gamma",
        ):
            if not hasattr(env.unwrapped, name):
                raise ValueError(f"The MDP must define {name}.")
        self.n_state = len(env.unwrapped.states)
        self.n_action = env.action_space.n
        self.Q = np.zeros((self.n_state, self.n_action))
        self.diagnostics = {}
        self.evaluation_return = []
        self.visit_counts = None
        self.R = R
        self.C = C
        self._validate_uncertainty(R, C)

    @staticmethod
    def _validate_uncertainty(R, C):
        if not np.isfinite(R) or not 0 <= R <= 1:
            raise ValueError("Contamination level R must be in [0, 1].")
        if np.isnan(C) or C < 0:
            raise ValueError("Deviation radius C must be nonnegative, or positive infinity.")

    def initialize_q(self, initial_Q=None):
        """Set Q with terminal rows zero and invalidate results from the previous run."""
        if initial_Q is None:
            self.Q.fill(0.0)
        else:
            initial_Q = np.asarray(initial_Q, dtype=float)
            if initial_Q.shape != self.Q.shape or not np.all(np.isfinite(initial_Q)):
                raise ValueError(f"initial_Q must be a finite array with shape {self.Q.shape}.")
            self.Q[:] = initial_Q
        self.Q[self.env.unwrapped.terminal_mask] = 0.0
        self.diagnostics = {}
        self.evaluation_return = []
        self.visit_counts = None

    def epsilon_greedy_policy(self, state, *, epsilon, rng):
        """Select an action using the calling algorithm's exploration state."""
        if rng.random() < epsilon:
            return int(rng.integers(self.n_action))
        row = self.env.unwrapped.state_index(state)
        best_actions = np.flatnonzero(self.Q[row] == np.max(self.Q[row]))
        return int(rng.choice(best_actions))

    def robust_td_target(self, reward, nominal_next_state, values):
        base = self.env.unwrapped
        nominal_value = base.values_at(nominal_next_state, values)
        if self.R == 0:
            return float(reward + base.gamma * nominal_value)
        worst_value = base.worst_case_value(nominal_next_state, self.C, values)
        next_value = (1 - self.R) * nominal_value + self.R * worst_value
        return float(reward + base.gamma * next_value)

    def robust_bellman_residual(self):
        """Check the robust Bellman equation at every nonterminal state-action pair."""
        base = self.env.unwrapped
        if not np.all(np.isfinite(self.Q)):
            return np.inf
        values = np.max(self.Q, axis=1)
        residual = 0.0
        for state in base.trainable_states:
            row = base.state_index(state)
            for action in range(self.n_action):
                successor = base.nominal_next_state(state, action)
                target = self.robust_td_target(base.reward(state, action), successor, values)
                if not np.isfinite(target):
                    return np.inf
                residual = max(residual, abs(target - self.Q[row, action]))
        return float(residual)

    def _record_diagnostics(self, algorithm, global_steps, residual, tolerance, **details):
        """Use one result schema for current and future tabular solvers."""
        self.diagnostics = {
            "algorithm": algorithm, "global_steps": int(global_steps),
            "bellman_residual": float(residual), "tolerance": float(tolerance),
            "converged": bool(residual <= tolerance), "R": float(self.R), "C": float(self.C),
            **details,
        }

    def robust_value_iteration(self, *, tolerance=1e-8, max_iterations=10000, initial_Q=None):
        """Solve synchronously from zeros or initial_Q; reject unconverged solutions."""
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be positive and finite.")
        if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer.")
        base = self.env.unwrapped
        self.initialize_q(initial_Q)
        residual = self.robust_bellman_residual()
        iteration = 0
        while residual > tolerance and iteration < max_iterations:
            iteration += 1
            values = self.Q.max(axis=1)
            updated = np.zeros_like(self.Q)
            for state in base.trainable_states:
                row = base.state_index(state)
                for action in range(self.n_action):
                    successor = base.nominal_next_state(state, action)
                    reward = base.reward(state, action)
                    updated[row, action] = self.robust_td_target(reward, successor, values)
            change = float(np.max(np.abs(updated - self.Q)))
            self.Q[:] = updated
            if change <= tolerance or iteration == max_iterations:
                residual = self.robust_bellman_residual()
        self._record_diagnostics(
            "robust_value_iteration", iteration * len(base.trainable_states) * self.n_action,
            residual, tolerance, iterations=iteration,
        )
        if not self.diagnostics["converged"]:
            raise RuntimeError(
                f"Value iteration did not converge in {max_iterations} iterations: "
                f"Bellman residual {residual:.3g} exceeds tolerance {tolerance:.3g}."
            )
        return self.Q.copy(), np.argmax(self.Q, axis=1)

    def robust_q_learning(
        self, max_training_episodes, *, lr_init, step_start_decay_lr=10000, epsilon_init=1.0,
        epsilon_lb=0.01, epsilon_decay_rate=0.999, seed=None, exploring_starts=None,
        record_every=1, verbose=False, initial_Q=None, tolerance=1e-6, check_every=100,
    ):
        """Algorithm 1, with customizable initialization and a deterministic nominal simulator.

        Learning rates stay constant during the initial global warmup, then decay
        by each pair's own post-warmup visit count. Infinite visits consequently
        give an infinite learning-rate sum for each learnable pair. Terminal rows
        are fixed at zero; time-limit truncation does not suppress bootstrapping.
        Stop when the full Bellman residual meets tolerance. max_training_episodes is a
        safety cap; reaching it without convergence raises an error.
        """
        for name, value in (
            ("max_training_episodes", max_training_episodes), ("record_every", record_every),
            ("check_every", check_every),
        ):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be positive and finite.")
        if not np.isfinite(lr_init) or not 0 < lr_init < 1:
            raise ValueError("lr_init must be in (0, 1).")
        if not 0 <= epsilon_lb <= epsilon_init <= 1 or not 0 < epsilon_decay_rate <= 1:
            raise ValueError("Invalid epsilon-greedy schedule.")
        if step_start_decay_lr < 0:
            raise ValueError("step_start_decay_lr must be nonnegative.")
        base = self.env.unwrapped
        if hasattr(base, "is_nominal") and not base.is_nominal:
            raise ValueError("Algorithm 1 requires a deterministic nominal training environment.")
        if self.R > 0 and hasattr(base, "P"):
            transitions = (items for row in base.P.values() for items in row.values())
            if any(len(items) != 1 or items[0][0] != 1 for items in transitions):
                raise ValueError("Robust Q-learning requires deterministic nominal transitions.")
        if exploring_starts is None:
            exploring_starts = True
        self.initialize_q(initial_Q)
        rng = np.random.default_rng(seed)
        values = np.max(self.Q, axis=1)
        zero_initialized = bool(np.all(self.Q == 0.0))
        epsilon = epsilon_init
        self.evaluation_return = [base.values_at(base.start_state, values)]
        visits = np.zeros_like(self.Q, dtype=int)
        decay_visits = np.zeros_like(visits)
        cumulative_step = 0
        episodes = 0
        residual = self.robust_bellman_residual()
        start_pairs = None
        if exploring_starts:
            start_pairs = np.array([
                (base.state_index(state), action) for state in base.trainable_states
                for action in range(self.n_action)
            ])
        while residual > tolerance and episodes < max_training_episodes:
            if verbose:
                print(f"Episode {episodes + 1}/{max_training_episodes}")
            options = None
            first_action = None
            if exploring_starts:
                if episodes % len(start_pairs) == 0:
                    rng.shuffle(start_pairs)
                start_row, first_action = start_pairs[episodes % len(start_pairs)]
                options = {"state": base.states[start_row]}
            state, _ = self.env.reset(options=options)
            while True:
                action = first_action
                if action is None:
                    action = self.epsilon_greedy_policy(state, epsilon=epsilon, rng=rng)
                first_action = None
                successor, reward, terminated, truncated, _ = self.env.step(action)
                row = base.state_index(state)
                visits[row, action] += 1
                if cumulative_step >= step_start_decay_lr:
                    decay_visits[row, action] += 1
                    learning_rate = lr_init / decay_visits[row, action]
                else:
                    learning_rate = lr_init
                target = self.robust_td_target(reward, successor, values)
                self.Q[row, action] += learning_rate * (target - self.Q[row, action])
                values[row] = np.max(self.Q[row])
                cumulative_step += 1
                if cumulative_step % record_every == 0:
                    self.evaluation_return.append(base.values_at(base.start_state, values))
                state = successor
                if terminated or truncated:
                    break
            episodes += 1
            epsilon = max(epsilon_lb, epsilon * epsilon_decay_rate)
            if episodes % check_every == 0 or episodes == max_training_episodes:
                residual = self.robust_bellman_residual()
        if cumulative_step % record_every != 0:
            self.evaluation_return.append(base.values_at(base.start_state, values))
        self.visit_counts = visits
        learnable_visits = visits[~base.terminal_mask]
        self._record_diagnostics(
            "robust_q_learning", cumulative_step, residual, tolerance, episodes=episodes,
            minimum_visits=int(np.min(learnable_visits)),
            unvisited_pairs=int(np.count_nonzero(learnable_visits == 0)),
            initial_state_value=base.values_at(base.start_state, values),
            zero_initialized=zero_initialized,
        )
        if not self.diagnostics["converged"]:
            raise RuntimeError(
                f"Q-learning did not converge in {max_training_episodes} episodes: "
                f"Bellman residual {residual:.3g} exceeds tolerance {tolerance:.3g}."
            )
        if verbose:
            print(f"Converged after {episodes} episodes; Bellman residual={residual:.3g}.")
        return self.Q.copy(), np.argmax(self.Q, axis=1)
