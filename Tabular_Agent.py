import numpy as np
import gymnasium as gym


class Tabular_Agent:
    """Learn Q-values using the supplied MDP's reward, terminal rules, and uncertainty set."""

    def __init__(
        self,
        env: gym.Env,
        gamma: float,
        lr_init: float,
        step_start_decay_lr: int = 10000,
        epsilon_init: float = 1.0,
        epsilon_lb: float = 0.01,
        epsilon_decay_rate: float = 0.999,
        R: float = 0.0,
        C: float = 1.0,
        seed=None,
    ):
        self.env = env
        for name in (
            "reward", "terminal_mask", "states", "state_index", "values_at", "worst_case_value",
        ):
            if not hasattr(env.unwrapped, name):
                raise ValueError(f"The MDP must define {name}.")
        self.gamma = gamma
        self.n_state = len(env.unwrapped.states)
        self.n_action = env.action_space.n
        if not np.isfinite(gamma) or not 0 <= gamma < 1:
            raise ValueError("gamma must be in [0, 1).")
        if not np.isfinite(lr_init) or not 0 < lr_init < 1:
            raise ValueError("lr_init must be in (0, 1).")
        if not 0 <= epsilon_lb <= epsilon_init <= 1 or not 0 < epsilon_decay_rate <= 1:
            raise ValueError("Invalid epsilon-greedy schedule.")
        if step_start_decay_lr < 0:
            raise ValueError("step_start_decay_lr must be nonnegative.")
        self.rng = np.random.default_rng(seed)
        self.initial_state = env.unwrapped.start_state
        self.training_diagnostics = {}

        self.Q = np.zeros((self.n_state, self.n_action))

        # Learning rate parameters
        self.lr_init = lr_init
        self.step_start_decay_lr = step_start_decay_lr

        # Epsilon-greedy policy parameters
        self.epsilon_init = epsilon_init
        self.epsilon = epsilon_init
        self.epsilon_lb = epsilon_lb
        self.epsilon_decay_rate = epsilon_decay_rate

        self.evaluation_return = [0.0]

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
        """Start from zeros, or copy a finite Q-table supplied by any initializer."""
        if initial_Q is None:
            self.Q.fill(0.0)
            return
        initial_Q = np.asarray(initial_Q, dtype=float)
        if initial_Q.shape != self.Q.shape or not np.all(np.isfinite(initial_Q)):
            raise ValueError(f"initial_Q must be a finite array with shape {self.Q.shape}.")
        self.Q[:] = initial_Q

    def epsilon_greedy_policy(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_action))
        row = self.env.unwrapped.state_index(state)
        best_actions = np.flatnonzero(self.Q[row] == np.max(self.Q[row]))
        return int(self.rng.choice(best_actions))

    def robust_td_target(self, reward, nominal_next_state, values, R=None, C=None):
        R = self.R if R is None else R
        C = self.C if C is None else C
        base = self.env.unwrapped
        nominal_value = base.values_at(nominal_next_state, values)
        if R == 0:
            return float(reward + self.gamma * nominal_value)
        worst_value = base.worst_case_value(nominal_next_state, C, values)
        next_value = (1 - R) * nominal_value + R * worst_value
        return float(reward + self.gamma * next_value)

    def robust_bellman_residual(self):
        """Use the nominal model only to diagnose finite-training approximation error."""
        base = self.env.unwrapped
        if not hasattr(base, "nominal_next_state"):
            return None
        values = np.max(self.Q, axis=1)
        residual = 0.0
        for state in base.trainable_states:
            row = base.state_index(state)
            for action in range(self.n_action):
                successor = base.nominal_next_state(state, action)
                target = self.robust_td_target(base.reward(state, action), successor, values)
                residual = max(residual, abs(target - self.Q[row, action]))
        return float(residual)

    def value_iteration(self, *, R=None, C=None, tolerance=1e-8, max_iterations=10000):
        """Solve the known nominal model synchronously; reject unconverged teacher labels."""
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be positive and finite.")
        if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer.")
        self.R = self.R if R is None else R
        self.C = self.C if C is None else C
        self._validate_uncertainty(self.R, self.C)
        base = self.env.unwrapped
        self.initialize_q()
        for iteration in range(1, max_iterations + 1):
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
                if residual <= tolerance:
                    self.value_iteration_diagnostics = {
                        "iterations": iteration, "bellman_residual": residual,
                    }
                    return self.Q.copy(), np.argmax(self.Q, axis=1)
        raise RuntimeError(f"Value iteration did not converge in {max_iterations} iterations.")

    def robust_q_learning(
        self, n_episodes, *, R=None, C=None, exploring_starts=None, record_every=1,
        verbose=False, initial_Q=None,
    ):
        """Algorithm 1, with customizable initialization and a deterministic nominal simulator.

        Learning rates stay constant during the initial global warmup, then decay
        by each pair's own post-warmup visit count. Infinite visits consequently
        give an infinite learning-rate sum for each learnable pair. Terminal rows
        are fixed at zero; time-limit truncation does not suppress bootstrapping.
        """
        if not isinstance(n_episodes, (int, np.integer)) or n_episodes < 1:
            raise ValueError("n_episodes must be a positive integer.")
        if not isinstance(record_every, (int, np.integer)) or record_every < 1:
            raise ValueError("record_every must be a positive integer.")
        base = self.env.unwrapped
        if hasattr(base, "is_nominal") and not base.is_nominal:
            raise ValueError("Algorithm 1 requires a deterministic nominal training environment.")
        self.R = self.R if R is None else R
        self.C = self.C if C is None else C
        self._validate_uncertainty(self.R, self.C)
        if self.R > 0 and hasattr(base, "P"):
            transitions = (items for row in base.P.values() for items in row.values())
            if any(len(items) != 1 or items[0][0] != 1 for items in transitions):
                raise ValueError("Robust Q-learning requires deterministic nominal transitions.")
        if exploring_starts is None:
            exploring_starts = hasattr(base, "trainable_states")
        if exploring_starts and not hasattr(base, "trainable_states"):
            raise ValueError("This environment does not support exploring starts.")
        if initial_Q is None:
            self.initialize_q()
        else:
            self.initialize_q(initial_Q)
        self.Q[base.terminal_mask] = 0.0
        values = np.max(self.Q, axis=1)
        zero_initialized = bool(np.all(self.Q == 0.0))
        self.epsilon = self.epsilon_init
        self.evaluation_return = [base.values_at(self.initial_state, values)]
        visits = np.zeros_like(self.Q, dtype=int)
        decay_visits = np.zeros_like(visits)
        cumulative_step = 0
        start_pairs = None
        if exploring_starts:
            start_pairs = np.array([
                (base.state_index(state), action) for state in base.trainable_states
                for action in range(self.n_action)
            ])
        for episode in range(n_episodes):
            if verbose:
                print(f"Episode {episode + 1}/{n_episodes}")
            options = None
            first_action = None
            if exploring_starts:
                if episode % len(start_pairs) == 0:
                    self.rng.shuffle(start_pairs)
                start_row, first_action = start_pairs[episode % len(start_pairs)]
                options = {"state": base.states[start_row]}
            state, _ = self.env.reset(options=options)
            while True:
                action = self.epsilon_greedy_policy(state) if first_action is None else first_action
                first_action = None
                successor, reward, terminated, truncated, _ = self.env.step(action)
                row = base.state_index(state)
                visits[row, action] += 1
                if cumulative_step >= self.step_start_decay_lr:
                    decay_visits[row, action] += 1
                    learning_rate = self.lr_init / decay_visits[row, action]
                else:
                    learning_rate = self.lr_init
                target = self.robust_td_target(reward, successor, values)
                self.Q[row, action] += learning_rate * (target - self.Q[row, action])
                values[row] = np.max(self.Q[row])
                cumulative_step += 1
                if cumulative_step % record_every == 0:
                    self.evaluation_return.append(base.values_at(self.initial_state, values))
                state = successor
                if terminated or truncated:
                    break
            self.epsilon = max(self.epsilon_lb, self.epsilon * self.epsilon_decay_rate)
        if cumulative_step % record_every != 0:
            self.evaluation_return.append(base.values_at(self.initial_state, values))
        self.visit_counts = visits
        learnable_visits = visits[~base.terminal_mask]
        self.training_diagnostics = {
            "episodes": int(n_episodes), "transitions": cumulative_step,
            "minimum_visits": int(np.min(learnable_visits)),
            "unvisited_pairs": int(np.count_nonzero(learnable_visits == 0)),
            "bellman_residual": self.robust_bellman_residual(),
            "initial_state_value": base.values_at(self.initial_state, values),
            "zero_initialized": zero_initialized, "R": float(self.R), "C": float(self.C),
        }
        return self.Q.copy(), np.argmax(self.Q, axis=1)
