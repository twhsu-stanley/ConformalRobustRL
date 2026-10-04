"""Continuous robot motion with a finite grid model for robust reinforcement learning."""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Robot2DMotionMDP(gym.Env):
    """An 8-by-8 robot MDP with separate nominal and uncertain rollout instances.

    Cells are indexed by (ix, iy), with state = ix + nx * iy. Actions 0, 1, 2, 3
    command +x, -x, +y, -y, respectively. Rewards depend on (state, action), even
    during uncertain deployment. Goal and obstacle cells have zero-valued absorbing
    dynamics. Fixed-horizon mode executes those dynamics until the horizon ends.

    A custom noise_sampler(nominal_position, rng) returns a bounded 2-D deviation.
    The default sampler mixes zero deviation with uniform noise in a disk. Domain
    projection and absorbing states can make mismatch probability state dependent.
    """

    metadata = {"render_modes": []}
    ACTION_NAMES = ("+x", "-x", "+y", "-y")
    ACTION_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
    DEFAULT_OBSTACLES = ((2, 1), (5, 1), (1, 3), (4, 3), (6, 4), (2, 5), (5, 6))

    def __init__(
        self, x_max=8.0, y_max=8.0, grid_shape=(8, 8), obstacles=None,
        start_cell=(0, 0), goal_cell=None, movement_reward=0.01, goal_reward=1.0,
        noise_probability=0.0, noise_radius=1.2, noise_sampler=None,
        max_episode_steps=100, episode_mode="episodic", seed=None,
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
        self.observation_space = spaces.Discrete(self.n_state)
        self.action_space = spaces.Discrete(4)
        if episode_mode not in {"episodic", "fixed_horizon"}:
            raise ValueError("episode_mode must be 'episodic' or 'fixed_horizon'.")
        if not isinstance(max_episode_steps, (int, np.integer)) or max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer.")
        self.max_episode_steps = int(max_episode_steps)
        self.episode_mode = episode_mode
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
        self.initial_seed = seed

        if obstacles is None:
            obstacles = self.DEFAULT_OBSTACLES if self.grid_shape == (8, 8) else ()
        self.obstacle_mask = np.zeros((self.ny, self.nx), dtype=bool)
        for cell in obstacles:
            ix, iy = self._validate_cell(cell)
            self.obstacle_mask[iy, ix] = True
        self.start_cell = self._validate_cell(start_cell)
        goal_cell = (self.nx - 1, self.ny - 1) if goal_cell is None else goal_cell
        self.goal_cell = self._validate_cell(goal_cell)
        self.start_state = self.cell_to_state(self.start_cell)
        self.goal_state = self.cell_to_state(self.goal_cell)
        if self.start_state == self.goal_state:
            raise ValueError("Start and goal must be different cells.")
        if self.obstacle_mask[self.start_cell[1], self.start_cell[0]]:
            raise ValueError("Start cannot be an obstacle.")
        if self.obstacle_mask[self.goal_cell[1], self.goal_cell[0]]:
            raise ValueError("Goal cannot be an obstacle.")
        self._validate_path()
        states = np.arange(self.n_state)
        indices = np.column_stack((states % self.nx, states // self.nx))
        self.state_coordinates = (indices + 0.5) * self.cell_widths
        self.terminal_mask = self.obstacle_mask.ravel().copy()
        self.terminal_mask[self.goal_state] = True
        self.trainable_states = np.flatnonzero(~self.terminal_mask)
        self.nominal_transitions = np.zeros((self.n_state, 4), dtype=int)
        self.rewards = np.zeros((self.n_state, 4), dtype=float)
        self.boundary_commands = np.zeros((self.n_state, 4), dtype=bool)
        self._build_nominal_model()
        self.reset(seed=seed)

    @staticmethod
    def _probability(value, name):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
        return float(value)

    def _validate_cell(self, cell):
        values = np.asarray(cell)
        if values.shape != (2,) or np.any(values != values.astype(int)):
            raise ValueError("A cell must be a pair of integer indices (ix, iy).")
        ix, iy = map(int, values)
        if not 0 <= ix < self.nx or not 0 <= iy < self.ny:
            raise ValueError(f"Cell {(ix, iy)} is outside the grid.")
        return ix, iy

    def _validate_state(self, state):
        if not self.observation_space.contains(state):
            raise ValueError(f"Invalid state: {state}.")
        return int(state)

    def _validate_action(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}.")
        return int(action)

    def cell_to_state(self, cell):
        ix, iy = self._validate_cell(cell)
        return ix + self.nx * iy

    def state_to_cell(self, state):
        state = self._validate_state(state)
        return state % self.nx, state // self.nx

    def coordinates(self, state):
        return self.state_coordinates[self._validate_state(state)].copy()

    def discretize(self, position):
        """Map positions to cells; an upper outer boundary belongs to the last cell."""
        position = np.asarray(position, dtype=float)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError("A continuous position must be a finite 2-D vector.")
        if np.any(position < 0) or np.any(position > self.bounds):
            raise ValueError("Position is outside the bounded state space.")
        indices = np.minimum((position / self.cell_widths).astype(int), (self.nx - 1, self.ny - 1))
        return int(indices[0] + self.nx * indices[1])

    def _validate_path(self):
        queue = deque([self.start_cell])
        visited = {self.start_cell}
        while queue:
            cell = queue.popleft()
            if cell == self.goal_cell:
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

    def _build_nominal_model(self):
        for state in range(self.n_state):
            ix, iy = self.state_to_cell(state)
            for action, (dx, dy) in enumerate(self.ACTION_DIRECTIONS):
                self.nominal_transitions[state, action] = state
                if self.terminal_mask[state]:
                    continue
                cell = (ix + dx, iy + dy)
                if not 0 <= cell[0] < self.nx or not 0 <= cell[1] < self.ny:
                    self.boundary_commands[state, action] = True
                    continue
                next_state = self.cell_to_state(cell)
                self.nominal_transitions[state, action] = next_state
                if next_state == self.goal_state:
                    self.rewards[state, action] = self.goal_reward
                elif not self.terminal_mask[next_state]:
                    self.rewards[state, action] = self.movement_reward

    def nominal_next_state(self, state, action):
        state, action = self._validate_state(state), self._validate_action(action)
        return int(self.nominal_transitions[state, action])

    def nominal_next_position(self, state, action):
        return self.coordinates(self.nominal_next_state(state, action))

    def reward(self, state, action):
        return float(self.rewards[self._validate_state(state), self._validate_action(action)])

    def localized_states_around(self, nominal_state, radius):
        """Equation (3): include every cell intersected by the continuous deviation ball."""
        if np.isnan(radius) or radius < 0:
            raise ValueError("Deviation radius must be nonnegative, or positive infinity.")
        nominal_state = self._validate_state(nominal_state)
        if np.isposinf(radius):
            return np.arange(self.n_state)
        offsets = np.abs(self.state_coordinates - self.state_coordinates[nominal_state])
        distances = np.linalg.norm(np.maximum(offsets - self.cell_widths / 2, 0), axis=1)
        return np.flatnonzero(distances <= radius + 1e-12)

    def localized_uncertainty_states(self, state, action, radius):
        return self.localized_states_around(self.nominal_next_state(state, action), radius)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "state" in options and "position" in options:
            raise ValueError("Specify either state or position, not both.")
        if "position" in options:
            self.s = self.discretize(options["position"])
            self.position = np.asarray(options["position"], dtype=float).copy()
        else:
            self.s = self._validate_state(options.get("state", self.start_state))
            self.position = self.coordinates(self.s)
        self.elapsed_steps = 0
        self._episode_finished = False
        return self.s, {"position": self.position.copy()}

    def _sample_deviation(self, nominal_position):
        if self.noise_sampler is not None:
            deviation = np.asarray(self.noise_sampler(nominal_position.copy(), self.np_random))
        elif self.np_random.random() < self.noise_probability:
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

    def step(self, action):
        if self._episode_finished:
            raise RuntimeError("Episode has ended; call reset before taking another step.")
        action = self._validate_action(action)
        state = self.s
        nominal_state = self.nominal_next_state(state, action)
        nominal_position = self.coordinates(nominal_state)
        absorbing = bool(self.terminal_mask[state])
        deviation = np.zeros(2) if absorbing else self._sample_deviation(nominal_position)
        raw_position = nominal_position + deviation
        self.position = np.clip(raw_position, 0, self.bounds)
        self.s = self.discretize(self.position)
        deviation = self.position - nominal_position
        self.elapsed_steps += 1
        terminal = bool(self.terminal_mask[self.s])
        terminated = terminal and self.episode_mode == "episodic"
        truncated = self.elapsed_steps >= self.max_episode_steps and not terminated
        self._episode_finished = terminated or truncated
        info = {
            "position": self.position.copy(), "nominal_state": nominal_state,
            "nominal_position": nominal_position, "deviation": deviation,
            "deviation_norm": float(np.linalg.norm(deviation)),
            "mismatch": self.s != nominal_state, "goal_reached": self.s == self.goal_state,
            "collision": bool(self.obstacle_mask.ravel()[self.s]),
            "boundary_attempt": bool(self.boundary_commands[state, action]),
            "boundary_violation": bool(
                np.any(raw_position < 0) or np.any(raw_position > self.bounds)
            ),
            "absorbing_transition": absorbing,
        }
        return self.s, self.reward(state, action), terminated, truncated, info

    def clone(self, *, nominal=False, episode_mode=None, max_episode_steps=None, seed=None):
        """Create an independent rollout instance with the same map and reward function."""
        obstacle_cells = [(int(ix), int(iy)) for iy, ix in np.argwhere(self.obstacle_mask)]
        horizon = self.max_episode_steps if max_episode_steps is None else max_episode_steps
        return Robot2DMotionMDP(
            x_max=self.bounds[0], y_max=self.bounds[1], grid_shape=self.grid_shape,
            obstacles=obstacle_cells, start_cell=self.start_cell, goal_cell=self.goal_cell,
            movement_reward=self.movement_reward, goal_reward=self.goal_reward,
            noise_probability=0.0 if nominal else self.noise_probability,
            noise_radius=self.noise_radius, noise_sampler=None if nominal else self.noise_sampler,
            max_episode_steps=horizon,
            episode_mode=self.episode_mode if episode_mode is None else episode_mode, seed=seed,
        )

