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
        C: float = 1.0
    ):
        self.env = env
        self.gamma = gamma
        self.n_state = env.observation_space.n
        self.n_action = env.action_space.n

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
        self.evaluation_return.append(np.max(self.Q[0, :]))

        ####################################################
        # Below are parameters for robust tabular RL
        self.R = R # Contamination level for the R-C uncertainty set
        self.C = C # Deviation radius for the R-C uncertainty set
        self.state_coordinates = self._build_state_coordinates()

    def _build_state_coordinates(self):
        """Return coordinates used by the R-C model distance metric.

        For FrozenLake-like grid environments, states are mapped to
        (row, col) grid coordinates. For other tabular environments, the
        fallback is a one-dimensional coordinate equal to the state index.
        """
        if hasattr(self.env.unwrapped, "desc"):
            n_row, n_col = self.env.unwrapped.desc.shape
            return np.array(
                [[state // n_col, state % n_col] for state in range(self.n_state)],
                dtype=float,
            )

        return np.arange(self.n_state, dtype=float).reshape(-1, 1)

    def _localized_uncertainty_states(self, nominal_next_state, C=None):
        """Compute C(s,a) = {s_tilde : ||f(s,a)-s_tilde|| <= C}.

        In the deterministic simulator setting of Section IV-B, one observed
        transition reveals f(s,a), so the observed next state is used as the
        nominal next state. Setting C=np.inf recovers the original
        R-contamination set over the whole state space.
        """
        if C is None:
            C = self.C
        if C < 0:
            raise ValueError("Deviation radius C must be non-negative.")
        if np.isinf(C):
            return np.arange(self.n_state)

        nominal_next_state = int(nominal_next_state)
        distances = np.linalg.norm(
            self.state_coordinates - self.state_coordinates[nominal_next_state],
            axis=1,
        )
        return np.where(distances <= C + 1e-12)[0]

    def epsilon_greedy_policy(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.n_action)
        else:
            return np.argmax(self.Q[state, :])

    def MC_Control(self, n_episodes):
        """On-policy first-visit MC control algorithm"""
        
        cum_step = 0

        for episode in range(n_episodes):
            print(f"Episode {episode+1}/{n_episodes}")

            state, info = self.env.reset() # starting state at 0
            state_hist = [state]
            action_hist = []
            reward_hist = []
            while True:
                # Select action using epsilon-greedy policy given current Q
                action = self.epsilon_greedy_policy(state)
                action_hist.append(action)

                state, reward, done, truncated, info = self.env.step(action)
                state_hist.append(state)
                reward_hist.append(reward)
                
                if done or truncated:
                    break

            G = 0
            for t in reversed(range(len(action_hist))):
                cum_step += 1

                # Schedule learning rate decay
                if cum_step >= self.step_start_decay_lr:
                    self.lr = self.lr_init / (cum_step - self.step_start_decay_lr + 1)

                G = self.gamma * G + reward_hist[t]

                # check first visit
                if is_first_visit(state_hist, action_hist, t):
                    self.Q[state_hist[t], action_hist[t]] += self.lr * (G - self.Q[state_hist[t], action_hist[t]])
                    self.evaluation_return.append(np.max(self.Q[0, :]))

            # Schedule epsilon decay to impose GLIE
            self.epsilon = max(self.epsilon_lb, self.epsilon * self.epsilon_decay_rate)

        Q_star = self.Q.copy()
        policy_star = np.argmax(Q_star, axis=1)
        
        return Q_star, policy_star
    
    def SARSA(self, n_episodes):
        """On-policy TD(0) Sarsa control"""

        cum_step = 0

        for episode in range(n_episodes):
            print(f"Episode {episode+1}/{n_episodes}")
            
            state, info = self.env.reset() # starting state at 0

            action = self.epsilon_greedy_policy(state)
            
            while True:
                state_plus, reward, done, truncated, info = self.env.step(action)

                action_plus = self.epsilon_greedy_policy(state_plus)

                self.Q[state, action] += self.lr * (reward + self.gamma * self.Q[state_plus, action_plus] - self.Q[state, action])

                state = state_plus
                action = action_plus

                self.evaluation_return.append(np.max(self.Q[0, :]))

                if done or truncated:
                    break

                cum_step += 1

                # Schedule learning rate decay
                if cum_step >= self.step_start_decay_lr:
                    self.lr = self.lr_init / (cum_step - self.step_start_decay_lr + 1)
            
            # Schedule epsilon decay to impose GLIE
            self.epsilon = max(self.epsilon_lb, self.epsilon * self.epsilon_decay_rate)

    def Q_learning(self, n_episodes):
        """Off-policy TD(0) Q-learning control"""

        cum_step = 0

        for episode in range(n_episodes):
            print(f"Episode {episode+1}/{n_episodes}")
            
            state, info = self.env.reset() # starting state at 0

            while True:
                action = self.epsilon_greedy_policy(state)

                state_plus, reward, done, truncated, info = self.env.step(action)

                self.Q[state, action] += self.lr * (reward + self.gamma * np.max(self.Q[state_plus,:]) - self.Q[state, action]) 
                
                state = state_plus

                self.evaluation_return.append(np.max(self.Q[0, :]))

                if done or truncated:
                    break

                cum_step += 1

                # Schedule learning rate decay
                if cum_step >= self.step_start_decay_lr:
                    self.lr = self.lr_init / (cum_step - self.step_start_decay_lr + 1)
            
            # Schedule epsilon decay to impose GLIE
            self.epsilon = max(self.epsilon_lb, self.epsilon * self.epsilon_decay_rate)

    def Robust_Q_learning(self, n_episodes):
        """Robust Q-learning with the localized R-C contamination model.

        The robust TD target implements Section IV-B:

            r + gamma * (1 - R) * V(s')
              + gamma * R * min_{s_tilde in C(s,a)} V(s_tilde),

        where C(s,a) = {s_tilde : ||s' - s_tilde|| <= C} because, in the
        deterministic simulator setting, the observed next state s' equals
        f(s,a).
        """

        cum_step = 0

        for episode in range(n_episodes):
            print(f"Episode {episode+1}/{n_episodes}")
            
            state, info = self.env.reset() # starting state at 0

            while True:
                action = self.epsilon_greedy_policy(state)

                state_plus, reward, done, truncated, info = self.env.step(action)

                # V(s) = max_a Q(s,a) for all tabular states.
                V = np.max(self.Q, axis=1)

                # Localized R-C uncertainty set centered at the observed
                # deterministic nominal next state s' = f(s,a).
                uncertainty_states = self._localized_uncertainty_states(state_plus, self.C)
                worst_case_next_value = np.min(V[uncertainty_states])

                target = (
                    reward
                    + self.gamma * (1 - self.R) * V[state_plus]
                    + self.gamma * self.R * worst_case_next_value
                )
                
                self.Q[state, action] += self.lr * (target - self.Q[state, action]) 
                
                state = state_plus

                self.evaluation_return.append(np.max(self.Q[0, :]))

                cum_step += 1

                # Schedule learning rate decay
                if cum_step >= self.step_start_decay_lr:
                    self.lr = self.lr_init / (cum_step - self.step_start_decay_lr + 1)

                if done or truncated:
                    break

            # Schedule epsilon decay to impose GLIE
            self.epsilon = max(self.epsilon_lb, self.epsilon * self.epsilon_decay_rate)

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
        localized R-C perturbation model used by Robust_Q_learning.
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
