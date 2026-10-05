import numpy as np
import gymnasium as gym

def is_first_visit(state_hist, action_hist, t):
    for i in range(t):
        if state_hist[i] == state_hist[t] and action_hist[i] == action_hist[t]:
            return False
    return True

class Tabular_Agent:
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
        self.gamma = gamma
        self.n_state = env.observation_space.n
        self.n_action = env.action_space.n
        if not np.isfinite(gamma) or not 0 <= gamma < 1:
            raise ValueError("gamma must be in [0, 1).")
        if not np.isfinite(lr_init) or not 0 < lr_init < 1:
            raise ValueError("lr_init must be in (0, 1).")
        if not 0 <= epsilon_lb <= epsilon_init <= 1 or not 0 < epsilon_decay_rate <= 1:
            raise ValueError("Invalid epsilon-greedy schedule.")
        if step_start_decay_lr < 0:
            raise ValueError("step_start_decay_lr must be nonnegative.")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.initial_state = int(getattr(env.unwrapped, "start_state", 0))
        self.terminal_mask = self._build_terminal_mask()
        self._uncertainty_cache = {}
        self.training_diagnostics = {}

        self.Q = np.zeros((self.n_state, self.n_action))

        # Learning rate parameters
        self.lr_init = lr_init
        self.lr = lr_init
        self.step_start_decay_lr = step_start_decay_lr

        # Epsilon-greedy policy parameters
        self.epsilon_init = epsilon_init
        self.epsilon = epsilon_init
        self.epsilon_lb = epsilon_lb
        self.epsilon_decay_rate = epsilon_decay_rate

        self.evaluation_return = []
        self.evaluation_return.append(np.max(self.Q[self.initial_state, :]))

        ####################################################
        # Below are parameters for robust tabular RL
        self.R = R # Contamination level for the R-C uncertainty set
        self.C = C # Deviation radius for the R-C uncertainty set
        self._validate_uncertainty(R, C)
        self.state_coordinates = self._build_state_coordinates()

    @staticmethod
    def _validate_uncertainty(R, C):
        if not np.isfinite(R) or not 0 <= R <= 1:
            raise ValueError("Contamination level R must be in [0, 1].")
        if np.isnan(C) or C < 0:
            raise ValueError("Deviation radius C must be nonnegative, or positive infinity.")

    def _build_terminal_mask(self):
        base = self.env.unwrapped
        if hasattr(base, "terminal_mask"):
            return np.asarray(base.terminal_mask, dtype=bool).copy()
        if hasattr(base, "desc"):
            return np.isin(base.desc.astype(str).ravel(), ("H", "G"))
        return np.zeros(self.n_state, dtype=bool)

    def _build_state_coordinates(self):
        """Return coordinates used by the R-C model distance metric.

        For FrozenLake-like grid environments, states are mapped to
        (row, col) grid coordinates. For other tabular environments, the
        fallback is a one-dimensional coordinate equal to the state index.
        """
        if hasattr(self.env.unwrapped, "state_coordinates"):
            return np.asarray(self.env.unwrapped.state_coordinates, dtype=float).copy()
        if hasattr(self.env.unwrapped, "desc"):
            n_row, n_col = self.env.unwrapped.desc.shape
            return np.array(
                [[state // n_col, state % n_col] for state in range(self.n_state)],
                dtype=float,
            )

        return np.arange(self.n_state, dtype=float).reshape(-1, 1)

    def _localized_uncertainty_states(self, nominal_next_state, C=None):
        """Use the environment's physical cell geometry, or legacy center distances.

        In the deterministic simulator setting of Section IV-B, one observed
        transition reveals f(s,a), so the observed next state is used as the
        nominal next state. Setting C=np.inf recovers the original
        R-contamination set over the whole state space.
        """
        if C is None:
            C = self.C
        self._validate_uncertainty(self.R, C)
        nominal_next_state = int(nominal_next_state)
        key = (nominal_next_state, float(C))
        if key in self._uncertainty_cache:
            return self._uncertainty_cache[key]
        base = self.env.unwrapped
        if hasattr(base, "localized_states_around"):
            states = base.localized_states_around(nominal_next_state, C)
        elif np.isposinf(C):
            states = np.arange(self.n_state)
        else:
            distances = np.linalg.norm(
                self.state_coordinates - self.state_coordinates[nominal_next_state], axis=1,
            )
            states = np.flatnonzero(distances <= C + 1e-12)
        self._uncertainty_cache[key] = states
        return states

    def epsilon_greedy_policy(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_action))
        best_actions = np.flatnonzero(self.Q[state] == np.max(self.Q[state]))
        return int(self.rng.choice(best_actions))

    def Q_learning(self, n_episodes, **kwargs):
        """Ordinary Q-learning is the R=0 special case of the robust update."""
        return self.robust_q_learning(n_episodes, R=0.0, **kwargs)

    def robust_td_target(self, reward, nominal_next_state, values, R=None, C=None):
        R = self.R if R is None else R
        C = self.C if C is None else C
        if R == 0:
            return float(reward + self.gamma * values[nominal_next_state])
        states = self._localized_uncertainty_states(nominal_next_state, C)
        next_value = (1 - R) * values[nominal_next_state] + R * np.min(values[states])
        return float(reward + self.gamma * next_value)

    def robust_bellman_residual(self):
        """Use the nominal model only to diagnose finite-training approximation error."""
        base = self.env.unwrapped
        if not hasattr(base, "nominal_next_state") or not hasattr(base, "reward"):
            return None
        values = np.max(self.Q, axis=1)
        values[self.terminal_mask] = 0.0
        residual = 0.0
        for state in np.flatnonzero(~self.terminal_mask):
            for action in range(self.n_action):
                successor = base.nominal_next_state(state, action)
                target = self.robust_td_target(base.reward(state, action), successor, values)
                residual = max(residual, abs(target - self.Q[state, action]))
        return float(residual)

    def robust_q_learning(
        self, n_episodes, *, R=None, C=None, exploring_starts=None, record_every=1,
        verbose=False,
    ):
        """Algorithm 1, with zero initialization and a deterministic nominal simulator.

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
        self.Q.fill(0.0)
        self.lr = self.lr_init
        self.epsilon = self.epsilon_init
        self.rng = np.random.default_rng(self.seed)
        self._uncertainty_cache.clear()
        self.evaluation_return = [0.0]
        self.evaluation_steps = [0]
        visits = np.zeros_like(self.Q, dtype=int)
        decay_visits = np.zeros_like(visits)
        values = np.zeros(self.n_state)
        cumulative_step = 0
        start_pairs = None
        if exploring_starts:
            start_pairs = np.array([
                (state, action) for state in base.trainable_states
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
                start_state, first_action = start_pairs[episode % len(start_pairs)]
                options = {"state": int(start_state)}
            reset_seed = int(self.rng.integers(0, 2**32 - 1))
            state, _ = self.env.reset(seed=reset_seed, options=options)
            while True:
                action = self.epsilon_greedy_policy(state) if first_action is None else first_action
                first_action = None
                successor, reward, terminated, truncated, _ = self.env.step(action)
                visits[state, action] += 1
                if cumulative_step >= self.step_start_decay_lr:
                    decay_visits[state, action] += 1
                    self.lr = self.lr_init / decay_visits[state, action]
                else:
                    self.lr = self.lr_init
                target = self.robust_td_target(reward, successor, values)
                self.Q[state, action] += self.lr * (target - self.Q[state, action])
                values[state] = np.max(self.Q[state])
                values[self.terminal_mask] = 0.0
                cumulative_step += 1
                if cumulative_step % record_every == 0:
                    self.evaluation_return.append(float(values[self.initial_state]))
                    self.evaluation_steps.append(cumulative_step)
                state = successor
                if terminated or truncated:
                    break
            self.epsilon = max(self.epsilon_lb, self.epsilon * self.epsilon_decay_rate)
        if self.evaluation_steps[-1] != cumulative_step:
            self.evaluation_return.append(float(values[self.initial_state]))
            self.evaluation_steps.append(cumulative_step)
        self.visit_counts = visits
        learnable_visits = visits[~self.terminal_mask]
        self.training_diagnostics = {
            "episodes": int(n_episodes), "transitions": cumulative_step,
            "minimum_visits": int(np.min(learnable_visits)),
            "unvisited_pairs": int(np.count_nonzero(learnable_visits == 0)),
            "bellman_residual": self.robust_bellman_residual(),
            "initial_state_value": float(values[self.initial_state]),
            "zero_initialized": True, "R": float(self.R), "C": float(self.C),
        }
        return self.Q.copy(), np.argmax(self.Q, axis=1)

    def _set_env_state(self, state):
        """Set the underlying tabular environment state when supported."""
        if hasattr(self.env.unwrapped, "s"):
            self.env.unwrapped.s = int(state)

    def _reward_done_from_state(self, state):
        """Return FrozenLake-style reward/done for an arbitrary state.

        FrozenLake rewards and termination depend on the tile reached, so when
        sim_perturbed manually replaces the nominal next state with an R-C
        perturbed next state, the reward and terminal flag must be recomputed.
        For non-FrozenLake tabular environments, return None so the caller can
        keep the environment-provided reward/done values.
        """
        if not hasattr(self.env.unwrapped, "desc"):
            return None

        desc = self.env.unwrapped.desc
        _, n_col = desc.shape
        row, col = divmod(int(state), n_col)
        tile = desc[row, col]
        if isinstance(tile, bytes):
            tile = tile.decode("utf-8")
        elif hasattr(tile, "item"):
            tile = tile.item()
            if isinstance(tile, bytes):
                tile = tile.decode("utf-8")

        reward = 1.0 if tile == "G" else 0.0
        done = tile in {"H", "G"}
        return reward, done

    def sim_perturbed(self, p=None, C=None):
        """Evaluate the greedy policy in an R-C perturbed environment.

        At each step, the agent first selects the greedy action. The simulator
        produces the nominal deterministic next state s' = f(s,a). With
        probability p, nature replaces s' with the worst-value state inside

            C(s,a) = {s_tilde : ||s' - s_tilde|| <= C}.

        With probability 1-p, the nominal transition is used. This matches the
        localized R-C perturbation model used by robust_q_learning.
        """
        if p is None:
            p = self.R

        if C is None:
            C = self.C

        if not 0.0 <= p <= 1.0:
            raise ValueError("Perturbation probability p must be in [0, 1].")

        # Initialize the environment and state.
        state, info = self.env.reset() # starting state at 0

        G = 0
        I = 1
        V = np.max(self.Q, axis=1)
        while True:
            # Agent acts according to its learned greedy policy.
            action = np.argmax(self.Q[state, :])

            # First take the nominal simulator transition to obtain f(s,a).
            nominal_next_state, reward, done, truncated, info = self.env.step(action)
            state_plus = int(nominal_next_state)

            if np.random.rand() <= p:
                # R-C perturbation: adversarially choose the lowest-value state
                # within radius C of the nominal next state f(s,a).
                uncertainty_states = self._localized_uncertainty_states(state_plus, C)
                local_values = V[uncertainty_states]
                min_value = np.min(local_values)
                worst_states = uncertainty_states[np.isclose(local_values, min_value)]
                state_plus = int(np.random.choice(worst_states))

                # Keep the wrapped simulator state consistent with the
                # manually perturbed next state.
                self._set_env_state(state_plus)

                # For FrozenLake, recompute reward/termination for the perturbed
                # next state. For unsupported environments, keep env.step values.
                reward_done = self._reward_done_from_state(state_plus)
                if reward_done is not None:
                    reward, done = reward_done

            G += I * reward
            I = I * self.gamma

            # Move to the next state.
            state = state_plus

            if done or truncated:
                break

        return G
