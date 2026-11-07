"""Shared reset, nominal model, collision checking, and deployment for grid robots."""

from collections import deque
from itertools import product

import gymnasium as gym
import numpy as np


class GridRobotMotionMDP(gym.Env):
    """Common behavior; examples define geometry, indexing, noise, and their prompts."""

    @staticmethod
    def _probability(value, name):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
        return float(value)

    def _validate_action(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}.")
        return int(action)

    def is_terminal(self, state):
        return self.is_outside(state) or bool(self.terminal_mask[self.state_index(state)])

    def coordinates(self, state):
        return (np.asarray(self._validate_state(state)) + 0.5) * self.cell_widths

    def _validate_path(self):
        queue = deque([self.start_state])
        visited = {self.start_state}
        while queue:
            cell = queue.popleft()
            if cell == self.goal_state:
                return
            for direction in self.ACTION_DIRECTIONS:
                neighbor = tuple(c + d for c, d in zip(cell, direction))
                if self.is_outside(neighbor):
                    continue
                if neighbor not in visited and not self.obstacle_mask[neighbor[::-1]]:
                    visited.add(neighbor)
                    queue.append(neighbor)
        raise ValueError("The obstacle map has no nominal path from start to goal.")

    def _build_nominal_model(self):
        for row, state in enumerate(self.states):
            if self.terminal_mask[row]:
                self.nominal_transitions[row, :] = state
                continue
            for action, direction in enumerate(self.ACTION_DIRECTIONS):
                self.nominal_transitions[row, action] = state
                cell = tuple(c + d for c, d in zip(state, direction))
                if self.is_outside(cell):
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

    def reward(self, state, action):
        state, action = self._validate_state(state), self._validate_action(action)
        if self.is_terminal(state):
            return 0.0
        return float(self.rewards[self.state_index(state), action])

    def q_transformer_features(self, R, C):
        """Build per-condition cell features from the known MDP, never from Q labels."""
        R, C = np.broadcast_arrays(np.atleast_1d(R).astype(float), np.atleast_1d(C).astype(float))
        if R.ndim != 1 or np.any(~np.isfinite(R)) or np.any((R < 0) | (R > 1)):
            raise ValueError("R must contain probabilities in [0, 1].")
        if np.any(np.isnan(C)) or np.any(C < 0):
            raise ValueError("C must contain nonnegative radii, including positive infinity.")
        scale = np.asarray(self.grid_shape) - 1
        coordinates = self.states / scale
        relative_goal = (self.goal_state - self.states) / scale
        successors = self.nominal_transitions
        rows = np.ravel_multi_index(
            tuple(successors[..., axis] for axis in reversed(range(len(scale)))),
            self.grid_shape[::-1],
        )
        nominal_terminal = self.terminal_mask[rows]
        # Zero at a terminal cell, one elsewhere: the MDP supplies the exact neighborhood.
        values = np.ones(self.n_state)
        terminal_nearby = {}
        for radius in np.unique(C):
            flags = np.array([
                self.worst_case_value(tuple(state), float(radius), values) == 0
                for state in self.states
            ])
            terminal_nearby[radius] = flags[rows]
        normalized_C = np.divide(C, 1 + C, out=np.ones_like(C), where=np.isfinite(C))
        geometry = np.column_stack((coordinates, relative_goal))
        local = np.column_stack((nominal_terminal, self.rewards))
        n_conditions = len(R)
        parameters = np.broadcast_to(
            np.column_stack((R, normalized_C))[:, None, :], (n_conditions, self.n_state, 2),
        )
        return np.concatenate((
            np.broadcast_to(geometry, (n_conditions, *geometry.shape)), parameters,
            np.broadcast_to(local, (n_conditions, *local.shape)),
            np.stack([terminal_nearby[radius] for radius in C]),
        ), axis=-1).astype(np.float32)

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
        crossings = np.full(len(self.grid_shape), np.inf)
        increments = np.full_like(crossings, np.inf)
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
            indices = np.where(on_boundary, rounded, np.floor(grid_position)).astype(int)
            axes = [(i - 1, i) if edge else (i,) for i, edge in zip(indices, on_boundary)]
            goal_contact = False
            for state in product(*axes):
                if self.is_outside(state):
                    continue
                if self.obstacle_mask[state[::-1]]:
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
        nominal_position = self.coordinates(nominal_state)
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
                not self.is_outside(self.s) and self.obstacle_mask[self.s[::-1]]
            ),
            "workspace_exit": self.is_outside(self.s),
            "boundary_attempt": bool(self.boundary_commands[self.state_index(state), action]),
            "boundary_violation": self.is_outside(next_state),
        }
        return (
            np.asarray(self.s, dtype=np.int64), self.reward(state, action),
            terminated, truncated, info,
        )

    def deploy_policy(self, policy, start_position=None):
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
            "discounted_return": float(np.dot(self.gamma**np.arange(len(rewards)), rewards)),
            "goal_reached": info["goal_reached"], "collision": info["collision"],
            "workspace_exit": info["workspace_exit"], "timeout": bool(truncated),
            "goal_step": length if info["goal_reached"] else None,
            "collision_step": length if info["collision"] else None,
            "workspace_exit_step": length if info["workspace_exit"] else None,
            "boundary_attempts": boundary_attempts, "boundary_violations": boundary_violations,
        }
