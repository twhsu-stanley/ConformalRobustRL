"""Continuous robot motion with a finite grid model for robust reinforcement learning."""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Robot2DMotionMDP(gym.Env):
    """An 8-by-8 robot MDP with separate nominal and uncertain rollout instances.

    Cells are indexed by (ix, iy), with state = ix + nx * iy. Actions 0, 1, 2, 3
    command +x, -x, +y, -y, respectively. Rewards depend on (state, action), even
    during uncertain deployment. Goal and obstacle cells have nominal self-loops.
    Deployment applies noise at every step, including at the goal. Workspace exits
    and obstacle landings restart from the pre-action position without an extra step.
    Outside grid cells share one zero-valued outcome in the finite planning model.

    A custom noise_sampler(nominal_position, rng) returns a bounded 2-D deviation.
    The default sampler mixes zero deviation with uniform noise in a disk.
    Noise is never projected or resampled. Each deployment records H noisy attempts.
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
        self.n_grid_states = self.nx * self.ny
        self.failure_state = self.n_grid_states
        self.n_state = self.n_grid_states + 1
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
        states = np.arange(self.n_grid_states)
        indices = np.column_stack((states % self.nx, states // self.nx))
        self.state_coordinates = np.vstack(((indices + 0.5) * self.cell_widths, [np.nan, np.nan]))
        self.terminal_mask = np.append(self.obstacle_mask.ravel(), True)
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
        if state == self.failure_state:
            raise ValueError("The workspace-failure state is not a grid cell.")
        return state % self.nx, state // self.nx

    def coordinates(self, state):
        state = self._validate_state(state)
        if state == self.failure_state:
            raise ValueError("The workspace-failure state has no cell center.")
        return self.state_coordinates[state].copy()

    def quantize(self, position):
        """Phi on the uniform grid extended to R^2, using half-open cells."""
        position = np.asarray(position, dtype=float)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError("A continuous position must be a finite 2-D vector.")
        return (np.floor(position / self.cell_widths) + 0.5) * self.cell_widths

    def discretize(self, position):
        """Index Phi(x) for planning; aggregate outside cells into the exit outcome."""
        center = self.quantize(position)
        if np.any(center < 0) or np.any(center > self.bounds):
            return self.failure_state
        indices = np.rint(center / self.cell_widths - 0.5).astype(int)
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
            if self.terminal_mask[state]:
                self.nominal_transitions[state, :] = state
                continue
            ix, iy = self.state_to_cell(state)
            for action, (dx, dy) in enumerate(self.ACTION_DIRECTIONS):
                self.nominal_transitions[state, action] = state
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
        if nominal_state == self.failure_state:
            return np.array([self.failure_state])
        center = self.coordinates(nominal_state)
        offsets = np.abs(self.state_coordinates[:self.n_grid_states] - center)
        distances = np.linalg.norm(np.maximum(offsets - self.cell_widths / 2, 0), axis=1)
        states = np.flatnonzero(distances <= radius + 1e-12)
        crosses_lower = radius > np.min(center) + 1e-12
        reaches_upper = radius >= np.min(self.bounds - center) - 1e-12
        if crosses_lower or reaches_upper:
            states = np.append(states, self.failure_state)
        return states

    def localized_uncertainty_states(self, state, action, radius):
        return self.localized_states_around(self.nominal_next_state(state, action), radius)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "state" in options and "position" in options:
            raise ValueError("Specify either state or position, not both.")
        if "position" in options:
            self.s = self.discretize(options["position"])
            if self.s == self.failure_state:
                raise ValueError("An episode must start inside the workspace.")
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
        """Observe an attempt, then apply an uncounted restart in fixed-horizon mode.

        The returned state is the next decision state. info retains the attempted
        position, quantized position, observed state, and original disturbance.
        """
        if self._episode_finished:
            raise RuntimeError("Episode has ended; call reset before taking another step.")
        action = self._validate_action(action)
        state = self.s
        source_position = self.position.copy()
        nominal_state = self.nominal_next_state(state, action)
        nominal_position = self.nominal_next_position(state, action)
        deviation = self._sample_deviation(nominal_position)
        self.position = nominal_position + deviation
        self.s = self.discretize(self.position)
        self.elapsed_steps += 1
        terminal = bool(self.terminal_mask[self.s])
        terminated = terminal and self.episode_mode == "episodic"
        truncated = self.elapsed_steps >= self.max_episode_steps and not terminated
        self._episode_finished = terminated or truncated
        info = {
            "position": self.position.copy(), "nominal_state": nominal_state,
            "nominal_position": nominal_position, "deviation": deviation,
            "observed_state": self.s, "quantized_position": self.quantize(self.position),
            "deviation_norm": float(np.linalg.norm(deviation)),
            "mismatch": self.s != nominal_state, "goal_reached": self.s == self.goal_state,
            "collision": bool(
                self.s < self.n_grid_states and self.obstacle_mask.ravel()[self.s]
            ),
            "workspace_exit": self.s == self.failure_state,
            "boundary_attempt": bool(self.boundary_commands[state, action]),
            "boundary_violation": self.s == self.failure_state,
        }
        if self.episode_mode == "fixed_horizon" and (info["collision"] or info["workspace_exit"]):
            self.s, self.position = state, source_position
        return self.s, self.reward(state, action), terminated, truncated, info

    def deploy_policy(self, policy, gamma, seed=None, start_position=None):
        """Run a full deployment horizon with a policy table or policy(state) callable.

        Record x'=F(x,a)+Delta before any restart. The next action uses the restored
        source position after an exit or collision; resets add no recorded step.
        This routine does not update any controller estimates or calibration.
        """
        if self.episode_mode != "fixed_horizon":
            raise ValueError("Policy deployment requires fixed_horizon mode.")
        options = None if start_position is None else {"position": start_position}
        _, info = self.reset(seed=seed, options=options)
        positions, states = [info["position"].copy()], [self.s]
        actions, rewards, deviations = [], [], []
        mismatches, nominal_positions, deviation_vectors = [], [], []
        source_states, source_positions, quantized_positions = [], [], []
        collisions, workspace_exits = [], []
        boundary_attempts, boundary_violations = 0, 0
        goal_step, collision_step, workspace_exit_step = None, None, None
        for step in range(self.max_episode_steps):
            state = self.s
            source_states.append(state)
            source_positions.append(self.position.copy())
            action = int(policy(state) if callable(policy) else policy[state])
            _, reward, terminated, truncated, info = self.step(action)
            nominal_positions.append(info["nominal_position"].copy())
            deviation_vectors.append(info["deviation"].copy())
            positions.append(info["position"].copy())
            states.append(info["observed_state"])
            quantized_positions.append(info["quantized_position"])
            actions.append(action)
            rewards.append(float(reward))
            deviations.append(info["deviation_norm"])
            mismatches.append(int(info["mismatch"]))
            collisions.append(info["collision"])
            workspace_exits.append(info["workspace_exit"])
            boundary_attempts += int(info["boundary_attempt"])
            boundary_violations += int(info["boundary_violation"])
            if info["goal_reached"] and goal_step is None:
                goal_step = step + 1
            if info["collision"] and collision_step is None:
                collision_step = step + 1
            if info["workspace_exit"] and workspace_exit_step is None:
                workspace_exit_step = step + 1
            if step + 1 < self.max_episode_steps and (terminated or truncated):
                raise RuntimeError("Deployment must execute the full fixed horizon.")
        return {
            "states": np.asarray(states), "positions": np.asarray(positions),
            "source_states": np.asarray(source_states),
            "source_positions": np.asarray(source_positions),
            "quantized_positions": np.asarray(quantized_positions),
            "nominal_positions": np.asarray(nominal_positions), "actions": np.asarray(actions),
            "deviation_vectors": np.asarray(deviation_vectors),
            "rewards": np.asarray(rewards), "deviations": np.asarray(deviations),
            "mismatches": np.asarray(mismatches), "collisions": np.asarray(collisions),
            "workspace_exits": np.asarray(workspace_exits),
            "score": max(deviations), "episode_mismatches": sum(mismatches),
            "collision_count": sum(collisions), "workspace_exit_count": sum(workspace_exits),
            "discounted_return": float(np.dot(gamma**np.arange(len(rewards)), rewards)),
            "goal_reached": goal_step is not None, "collision": collision_step is not None,
            "workspace_exit": workspace_exit_step is not None,
            "timeout": goal_step is None,
            "goal_step": goal_step, "collision_step": collision_step,
            "workspace_exit_step": workspace_exit_step,
            "boundary_attempts": boundary_attempts, "boundary_violations": boundary_violations,
        }

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

