"""Robot motion on a global grid with a finite workspace Q-table."""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Robot2DMotionMDP(gym.Env):
    """A grid robot MDP with separate nominal and uncertain rollout instances.

    States are integer grid coordinates (ix, iy), including outside the workspace.
    Workspace states have Q-table row ix + nx * iy. Actions 0, 1, 2, 3
    command +x, -x, +y, -y, respectively. Rewards depend on (state, action), even
    during uncertain deployment. Goal and obstacle cells have nominal self-loops.
    Deployment applies noise to every observed transition and stops at the goal,
    an obstacle, workspace exit, or the episode step limit.
    Every outside cell is a distinct terminal state with implicit value zero.

    A custom noise_sampler(nominal_position, rng) returns a bounded 2-D deviation.
    The default sampler mixes zero deviation with uniform noise in a disk.
    Noise is never projected or resampled. Segment contacts stop uncertain rollouts
    at the first terminal cell. Mismatches use the original sampled endpoint.
    Every sampled disturbance, including the terminal one, enters the ACP score.
    """

    metadata = {"render_modes": []}
    ACTION_NAMES = ("+x", "-x", "+y", "-y")
    ACTION_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
    DEFAULT_OBSTACLES = ((2, 1), (5, 1), (1, 3), (4, 3), (6, 4), (2, 5), (5, 6))

    def __init__(
        self, x_max=8.0, y_max=8.0, grid_shape=(8, 8), obstacles=None,
        start_cell=(0, 0), goal_cell=None, movement_reward=0.01, goal_reward=1.0,
        noise_probability=0.0, noise_radius=1.2, noise_sampler=None,
        max_episode_steps=100, seed=None,
    ):
        self.bounds = np.asarray((x_max, y_max), dtype=float)
        if not np.all(np.isfinite(self.bounds)) or np.any(self.bounds <= 0):
            raise ValueError("Domain bounds must be positive and finite.")
        shape = np.asarray(grid_shape)
        if shape.shape != (2,) or np.any(shape != shape.astype(int)) or np.any(shape < 2):
            raise ValueError("grid_shape must contain two integers of at least 2.")
        self.nx, self.ny = map(int, shape)
        self.grid_shape = (self.nx, self.ny)
        self.cell_widths = self.bounds / shape
        self.n_state = self.nx * self.ny
        limits = np.iinfo(np.int64)
        self.observation_space = spaces.Box(limits.min, limits.max, shape=(2,), dtype=np.int64)
        self.action_space = spaces.Discrete(4)
        if not isinstance(max_episode_steps, (int, np.integer)) or max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer.")
        self.max_episode_steps = int(max_episode_steps)
        self.movement_reward = self._probability(movement_reward, "movement_reward")
        self.goal_reward = self._probability(goal_reward, "goal_reward")
        if self.goal_reward <= self.movement_reward:
            raise ValueError("goal_reward must exceed movement_reward.")
        self.noise_probability = self._probability(noise_probability, "noise_probability")
        if not np.isfinite(noise_radius) or noise_radius < 0:
            raise ValueError("noise_radius must be finite and nonnegative.")
        if noise_sampler is not None and not callable(noise_sampler):
            raise ValueError("noise_sampler must be callable.")
        self.noise_radius = float(noise_radius)
        self.noise_sampler = noise_sampler
        self.is_nominal = noise_sampler is None and self.noise_probability == 0.0

        if obstacles is None:
            obstacles = self.DEFAULT_OBSTACLES if self.grid_shape == (8, 8) else ()
        self.obstacle_mask = np.zeros((self.ny, self.nx), dtype=bool)
        self.start_state = self._validate_state(start_cell)
        goal_cell = (self.nx - 1, self.ny - 1) if goal_cell is None else goal_cell
        self.goal_state = self._validate_state(goal_cell)
        self.state_index(self.start_state)
        self.state_index(self.goal_state)
        if self.start_state == self.goal_state:
            raise ValueError("Start and goal must be different cells.")
        rows = np.arange(self.n_state)
        self.states = np.column_stack((rows % self.nx, rows // self.nx))
        self.state_coordinates = (self.states + 0.5) * self.cell_widths
        self.nominal_transitions = np.zeros((self.n_state, 4, 2), dtype=int)
        self.rewards = np.zeros((self.n_state, 4), dtype=float)
        self.boundary_commands = np.zeros((self.n_state, 4), dtype=bool)
        self.set_obstacles(obstacles)
        self.reset(seed=seed)

    @staticmethod
    def _probability(value, name):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
        return float(value)

    def _validate_state(self, state):
        values = np.asarray(state)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("A state must be a pair of finite integer grid coordinates.")
        if np.any(values != np.floor(values)):
            raise ValueError("A state must have integer grid coordinates.")
        return tuple(map(int, values))

    def _validate_action(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}.")
        return int(action)

    def state_index(self, state):
        """Map a workspace state to its finite Q-table row."""
        state = self._validate_state(state)
        if self.is_outside(state):
            raise ValueError(f"Outside state {state} has no Q-table row.")
        return state[0] + self.nx * state[1]

    def is_outside(self, state):
        return not (0 <= state[0] < self.nx and 0 <= state[1] < self.ny)

    def is_terminal(self, state):
        return self.is_outside(state) or bool(self.terminal_mask[self.state_index(state)])

    def coordinates(self, state):
        return (np.asarray(self._validate_state(state)) + 0.5) * self.cell_widths

    def discretize(self, position):
        """Global Phi: half-open cells partition the plane, including outside X."""
        position = np.asarray(position, dtype=float)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError("A continuous position must be a finite 2-D vector.")
        return tuple(map(int, np.floor(position / self.cell_widths)))

    def values_at(self, states, values):
        """Look up workspace values; all terminal states, including outside cells, give zero."""
        states = np.asarray(states)
        if states.shape[-1:] != (2,) or not np.issubdtype(states.dtype, np.integer):
            raise ValueError("States must be integer grid-coordinate pairs.")
        inside = np.all((states >= 0) & (states < self.grid_shape), axis=-1)
        result = np.zeros(states.shape[:-1])
        if np.any(inside):
            selected = states[inside]
            rows = selected[:, 0] + self.nx * selected[:, 1]
            result[inside] = np.where(self.terminal_mask[rows], 0.0, values[rows])
        return float(result) if result.ndim == 0 else result

    def _validate_path(self):
        queue = deque([self.start_state])
        visited = {self.start_state}
        while queue:
            cell = queue.popleft()
            if cell == self.goal_state:
                return
            for dx, dy in self.ACTION_DIRECTIONS:
                neighbor = (cell[0] + dx, cell[1] + dy)
                ix, iy = neighbor
                if not 0 <= ix < self.nx or not 0 <= iy < self.ny:
                    continue
                if neighbor not in visited and not self.obstacle_mask[iy, ix]:
                    visited.add(neighbor)
                    queue.append(neighbor)
        raise ValueError("The obstacle map has no nominal path from start to goal.")

    def set_obstacles(self, obstacles):
        """Update the map between episodes, rebuilding its terminal and reward tables."""
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        for cell in obstacles:
            ix, iy = self._validate_state(cell)
            self.state_index((ix, iy))
            mask[iy, ix] = True
        if mask[self.start_state[1], self.start_state[0]]:
            raise ValueError("Start cannot be an obstacle.")
        if mask[self.goal_state[1], self.goal_state[0]]:
            raise ValueError("Goal cannot be an obstacle.")
        previous = self.obstacle_mask
        self.obstacle_mask = mask
        try:
            self._validate_path()
        except ValueError:
            self.obstacle_mask = previous
            raise
        self.terminal_mask = mask.ravel().copy()
        self.terminal_mask[self.state_index(self.goal_state)] = True
        self.trainable_states = self.states[~self.terminal_mask]
        self.rewards.fill(0.0)
        self.boundary_commands.fill(False)
        self._build_nominal_model()

    def _build_nominal_model(self):
        for row, state in enumerate(self.states):
            if self.terminal_mask[row]:
                self.nominal_transitions[row, :] = state
                continue
            ix, iy = state
            for action, (dx, dy) in enumerate(self.ACTION_DIRECTIONS):
                self.nominal_transitions[row, action] = state
                cell = (ix + dx, iy + dy)
                if not 0 <= cell[0] < self.nx or not 0 <= cell[1] < self.ny:
                    self.boundary_commands[row, action] = True
                    continue
                self.nominal_transitions[row, action] = cell
                if cell == self.goal_state:
                    self.rewards[row, action] = self.goal_reward
                elif not self.is_terminal(cell):
                    self.rewards[row, action] = self.movement_reward

    def nominal_next_state(self, state, action):
        state, action = self._validate_state(state), self._validate_action(action)
        if self.is_terminal(state):
            return state
        return tuple(self.nominal_transitions[self.state_index(state), action])

    def nominal_next_position(self, state, action):
        return self.coordinates(self.nominal_next_state(state, action))

    def reward(self, state, action):
        state, action = self._validate_state(state), self._validate_action(action)
        if self.is_terminal(state):
            return 0.0
        return float(self.rewards[self.state_index(state), action])

    def q_transformer_context(self):
        """Return the numeric grid and navigation instructions for the shared prompt."""
        grid = "\n".join(" ".join(map(str, row.astype(int))) for row in self.obstacle_mask)
        return (
            f"You are given a grid with size {self.nx}x{self.ny}.\n"
            "The maze is a 2D grid with rows in increasing y and columns in increasing x.\n"
            "0 = free space, 1 = obstacle.\n\n"
            f"{grid}\n"
            f"Start at {self.start_state} and move to the goal state at {self.goal_state}.\n"
            "Goal state and obstacles have zero Q-values."
        )

    def localized_uncertainty_states(self, nominal_next_state, C):
        """Enumerate the finite-C neighborhood over the global grid, using equation (4)."""
        if not np.isfinite(C) or C < 0:
            raise ValueError("An explicit neighborhood requires finite nonnegative C.")
        nominal_next_state = self._validate_state(nominal_next_state)
        cache = self.__dict__.setdefault("_uncertainty_cache", {})
        key = (nominal_next_state, float(C))
        if key in cache:
            return cache[key]
        reach = np.floor((C + 1e-12) / self.cell_widths + 0.5).astype(int)
        axes = [np.arange(s - r, s + r + 1) for s, r in zip(nominal_next_state, reach)]
        xs, ys = np.meshgrid(*axes)
        candidates = np.column_stack((xs.ravel(), ys.ravel()))
        offsets = np.abs(candidates - nominal_next_state) * self.cell_widths
        distances = np.linalg.norm(np.maximum(offsets - self.cell_widths / 2, 0), axis=1)
        states = candidates[distances <= C + 1e-12]
        cache[key] = states
        return states

    def worst_case_value(self, nominal_next_state, C, values):
        """Evaluate the local minimum; infinitely many outside terminal values are zero."""
        if np.isposinf(C):
            return min(0.0, float(np.min(self.values_at(self.states, values))))
        states = self.localized_uncertainty_states(nominal_next_state, C)
        return float(np.min(self.values_at(states, values)))

    def reset(self, *, seed=None, options=None):
        """Reset to a nonterminal start; an explicit seed restarts the disturbance RNG."""
        super().reset(seed=seed)
        options = options or {}
        if "state" in options and "position" in options:
            raise ValueError("Specify either state or position, not both.")
        if "position" in options:
            position = np.asarray(options["position"], dtype=float)
            state = self.discretize(position)
        else:
            state = self._validate_state(options.get("state", self.start_state))
        if self.is_terminal(state):
            raise ValueError("An episode must start at a nonterminal state inside the workspace.")
        self.s = state
        self.position = position.copy() if "position" in options else self.coordinates(state)
        self.elapsed_steps = 0
        self._episode_finished = False
        return np.asarray(self.s, dtype=np.int64), {"position": self.position.copy()}

    def _sample_deviation(self, nominal_position):
        if self.noise_sampler is not None:
            deviation = np.asarray(self.noise_sampler(nominal_position.copy(), self.np_random))
        elif self.noise_probability > 0 and self.np_random.random() < self.noise_probability:
            angle = self.np_random.uniform(0, 2 * np.pi)
            radius = self.noise_radius * np.sqrt(self.np_random.random())
            deviation = radius * np.array((np.cos(angle), np.sin(angle)))
        else:
            deviation = np.zeros(2)
        if deviation.shape != (2,) or not np.all(np.isfinite(deviation)):
            raise ValueError("Noise sampler must return a finite 2-D deviation.")
        if np.linalg.norm(deviation) > self.noise_radius + 1e-12:
            raise ValueError("Noise sampler exceeded its configured support radius.")
        return deviation.astype(float)

    def _collision_check_segment(self, start_position, end_position):
        """Return (terminal state, contact position), or None, using grid traversal.

        Cells are checked at successive grid-boundary crossings, including all
        cells touched at a corner or along a grid line. Obstacle contact takes
        precedence over simultaneous goal contact. Outside cells are not searched.
        """
        start_position = np.asarray(start_position, dtype=float)
        direction = np.asarray(end_position, dtype=float) - start_position
        shape = np.array(self.grid_shape)
        cell = np.minimum(np.floor(start_position / self.cell_widths), shape - 1)
        boundaries = (cell + (direction > 0)) * self.cell_widths
        crossings, increments = np.full(2, np.inf), np.full(2, np.inf)
        np.divide(boundaries - start_position, direction, out=crossings, where=direction != 0)
        np.divide(self.cell_widths, np.abs(direction), out=increments, where=direction != 0)
        parameter = 0.0
        while True:
            position = start_position + parameter * direction
            grid_position = position / self.cell_widths
            if np.any(grid_position < -1e-12) or np.any(grid_position > shape + 1e-12):
                return None
            rounded = np.rint(grid_position)
            on_boundary = np.abs(grid_position - rounded) <= 1e-12
            ix, iy = np.where(on_boundary, rounded, np.floor(grid_position)).astype(int)
            xs = (ix - 1, ix) if on_boundary[0] else (ix,)
            ys = (iy - 1, iy) if on_boundary[1] else (iy,)
            goal_contact = False
            for x in xs:
                for y in ys:
                    if not 0 <= x < self.nx or not 0 <= y < self.ny:
                        continue
                    state = (x, y)
                    if self.obstacle_mask[y, x]:
                        return state, position
                    goal_contact |= state == self.goal_state
            if goal_contact:
                return self.goal_state, position
            if parameter == 1.0:
                return None
            next_crossing = float(np.min(crossings))
            parameter = min(next_crossing, 1.0)
            crossed = crossings == next_crossing
            crossings[crossed] += increments[crossed]

    def step(self, action):
        """Sample x'=F(x,a)+Delta; terminal contacts stop the uncertain robot."""
        if self._episode_finished:
            raise RuntimeError("Episode has ended; call reset before taking another step.")
        action = self._validate_action(action)
        state = self.s
        nominal_state = self.nominal_next_state(state, action)
        nominal_position = self.nominal_next_position(state, action)
        deviation = self._sample_deviation(nominal_position)
        next_position = nominal_position + deviation
        next_state = self.discretize(next_position)
        contact = None
        if not self.is_nominal:
            contact = self._collision_check_segment(self.position, next_position)
        self.s, self.position = (next_state, next_position) if contact is None else contact
        self.elapsed_steps += 1
        terminated = self.is_terminal(self.s)
        truncated = self.elapsed_steps >= self.max_episode_steps and not terminated
        self._episode_finished = terminated or truncated
        info = {
            "position": self.position.copy(), "nominal_state": nominal_state,
            "nominal_position": nominal_position, "deviation": deviation,
            "deviation_norm": float(np.linalg.norm(deviation)),
            "mismatch": next_state != nominal_state, "goal_reached": self.s == self.goal_state,
            "collision": bool(
                not self.is_outside(self.s) and self.obstacle_mask[self.s[1], self.s[0]]
            ),
            "workspace_exit": self.is_outside(self.s),
            "boundary_attempt": bool(self.boundary_commands[self.state_index(state), action]),
            "boundary_violation": self.is_outside(next_state),
        }
        return (
            np.asarray(self.s, dtype=np.int64), self.reward(state, action),
            terminated, truncated, info,
        )

    def deploy_policy(self, policy, gamma, start_position=None):
        """Record each transition and score every disturbance, including the terminal one."""
        options = None if start_position is None else {"position": start_position}
        _, info = self.reset(options=options)
        positions, states = [info["position"].copy()], [self.s]
        actions, rewards, deviations = [], [], []
        mismatches, nominal_positions, deviation_vectors = [], [], []
        boundary_attempts, boundary_violations = 0, 0
        score = 0.0
        while True:
            state = self.s
            action = int(policy(state) if callable(policy) else policy[self.state_index(state)])
            successor, reward, terminated, truncated, info = self.step(action)
            nominal_positions.append(info["nominal_position"].copy())
            deviation_vectors.append(info["deviation"].copy())
            positions.append(info["position"].copy())
            states.append(successor)
            actions.append(action)
            rewards.append(float(reward))
            deviations.append(info["deviation_norm"])
            mismatches.append(int(info["mismatch"]))
            boundary_attempts += int(info["boundary_attempt"])
            boundary_violations += int(info["boundary_violation"])
            score = max(score, info["deviation_norm"])
            if terminated or truncated:
                break
        length = len(actions)
        return {
            "states": np.asarray(states), "positions": np.asarray(positions),
            "nominal_positions": np.asarray(nominal_positions), "actions": np.asarray(actions),
            "deviation_vectors": np.asarray(deviation_vectors),
            "rewards": np.asarray(rewards), "deviations": np.asarray(deviations),
            "mismatches": np.asarray(mismatches), "episode_length": length,
            "score": score, "episode_mismatches": sum(mismatches),
            "discounted_return": float(np.dot(gamma**np.arange(len(rewards)), rewards)),
            "goal_reached": info["goal_reached"], "collision": info["collision"],
            "workspace_exit": info["workspace_exit"], "timeout": bool(truncated),
            "goal_step": length if info["goal_reached"] else None,
            "collision_step": length if info["collision"] else None,
            "workspace_exit_step": length if info["workspace_exit"] else None,
            "boundary_attempts": boundary_attempts, "boundary_violations": boundary_violations,
        }

    def clone(self, *, nominal=False, max_episode_steps=None, seed=None):
        """Create an independent rollout instance with the same map and reward function."""
        obstacle_cells = [(int(ix), int(iy)) for iy, ix in np.argwhere(self.obstacle_mask)]
        horizon = self.max_episode_steps if max_episode_steps is None else max_episode_steps
        return Robot2DMotionMDP(
            x_max=self.bounds[0], y_max=self.bounds[1], grid_shape=self.grid_shape,
            obstacles=obstacle_cells, start_cell=self.start_state, goal_cell=self.goal_state,
            movement_reward=self.movement_reward, goal_reward=self.goal_reward,
            noise_probability=0.0 if nominal else self.noise_probability,
            noise_radius=self.noise_radius, noise_sampler=None if nominal else self.noise_sampler,
            max_episode_steps=horizon, seed=seed,
        )

