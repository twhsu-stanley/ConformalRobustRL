"""Robot motion on a global grid with a finite workspace Q-table."""

import numpy as np
from gymnasium import spaces

from grid_robot_mdp import GridRobotMotionMDP


class Robot2DMotionMDP(GridRobotMotionMDP):
    """A grid robot MDP with separate nominal and uncertain rollout instances.

    States are integer grid coordinates (ix, iy), including outside the workspace.
    Workspace states have Q-table row ix + nx * iy. Actions 0, 1, 2, 3
    command +x, -x, +y, -y, respectively. Rewards depend on (state, action), even
    during uncertain deployment. Goal and obstacle cells have nominal self-loops.
    Deployment applies noise to every observed transition and stops at the goal,
    an obstacle, workspace exit, or the episode step limit.
    Every outside cell is a distinct terminal state with implicit value zero.
    gamma defines the discounted objective for learning and deployment returns.

    A custom noise_sampler(nominal_position, rng) returns a bounded 2-D deviation.
    The default sampler mixes zero deviation with uniform noise in a disk.
    Noise is never projected or resampled. Segment contacts stop uncertain rollouts
    at the first terminal cell. Mismatches use the original sampled endpoint.
    Every sampled disturbance, including the terminal one, enters the ACP score.
    """

    metadata = {"render_modes": []}
    ACTION_NAMES = ("+x", "-x", "+y", "-y")
    ACTION_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def __init__(
        self, *, x_max, y_max, grid_shape, start_cell, goal_cell, gamma,
        movement_reward, goal_reward, obstacles=None,
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
        if not np.isfinite(gamma) or not 0 <= gamma < 1:
            raise ValueError("gamma must be in [0, 1).")
        self.gamma = float(gamma)
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

        self.obstacle_mask = np.zeros((self.ny, self.nx), dtype=bool)
        self.start_state = self._validate_state(start_cell)
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
        self.set_obstacles(() if obstacles is None else obstacles)
        self.reset(seed=seed)

    def _validate_state(self, state):
        values = np.asarray(state)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("A state must be a pair of finite integer grid coordinates.")
        if np.any(values != np.floor(values)):
            raise ValueError("A state must have integer grid coordinates.")
        return tuple(map(int, values))

    def state_index(self, state):
        """Map a workspace state to its finite Q-table row."""
        state = self._validate_state(state)
        if self.is_outside(state):
            raise ValueError(f"Outside state {state} has no Q-table row.")
        return state[0] + self.nx * state[1]

    def is_outside(self, state):
        return not (0 <= state[0] < self.nx and 0 <= state[1] < self.ny)

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
        if C >= np.linalg.norm(self.bounds):
            return min(0.0, float(np.min(self.values_at(self.states, values))))
        states = self.localized_uncertainty_states(nominal_next_state, C)
        return float(np.min(self.values_at(states, values)))

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

    def clone(self, *, nominal=False, max_episode_steps=None, seed=None):
        """Create an independent rollout instance with the same map, rewards, and discount."""
        obstacle_cells = [(int(ix), int(iy)) for iy, ix in np.argwhere(self.obstacle_mask)]
        horizon = self.max_episode_steps if max_episode_steps is None else max_episode_steps
        return Robot2DMotionMDP(
            x_max=self.bounds[0], y_max=self.bounds[1], grid_shape=self.grid_shape,
            obstacles=obstacle_cells, start_cell=self.start_state, goal_cell=self.goal_state,
            gamma=self.gamma, movement_reward=self.movement_reward, goal_reward=self.goal_reward,
            noise_probability=0.0 if nominal else self.noise_probability,
            noise_radius=self.noise_radius, noise_sampler=None if nominal else self.noise_sampler,
            max_episode_steps=horizon, seed=seed,
        )

