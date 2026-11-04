"""Check generated maps independently of the route-construction implementation."""

from collections import deque
import unittest
from unittest.mock import patch

import numpy as np

from grid_q_data import (
    _all_free_states_connected, _segment_obstacle_count, _wide_corridor, training_maps,
)
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP
from robot_3d.robot_3d_motion_mdp import Robot3DMotionMDP


def _has_wide_route(mdp):
    """Check passage width independently using sliding windows over the obstacle map."""
    dimension = len(mdp.grid_shape)
    free = ~mdp.obstacle_mask.transpose(tuple(reversed(range(dimension))))
    windows = np.lib.stride_tricks.sliding_window_view(free, (2,) * dimension)
    allowed = windows.all(axis=tuple(range(dimension, 2 * dimension)))
    anchors = set(map(tuple, np.argwhere(allowed)))
    starts = {a for a in anchors if all(x <= s < x + 2 for x, s in zip(a, mdp.start_state))}
    goals = {a for a in anchors if all(x <= s < x + 2 for x, s in zip(a, mdp.goal_state))}
    queue, visited = deque(starts), set(starts)
    while queue:
        state = queue.popleft()
        if state in goals:
            return True
        for direction in mdp.ACTION_DIRECTIONS:
            neighbor = tuple(np.asarray(state) + direction)
            if neighbor not in visited and neighbor in anchors:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def _route_count(mdp):
    """Send two unit flows through a vertex-split free-cell graph."""
    graph = {}

    def edge(source, target, capacity):
        graph.setdefault(source, {})[target] = capacity
        graph.setdefault(target, {}).setdefault(source, 0)

    for state in map(tuple, mdp.trainable_states):
        row = mdp.state_index(state)
        edge(2 * row, 2 * row + 1, 2 if state == mdp.start_state else 1)
        for direction in mdp.ACTION_DIRECTIONS:
            neighbor = tuple(np.asarray(state) + direction)
            if not mdp.is_outside(neighbor) and not mdp.obstacle_mask[neighbor[::-1]]:
                edge(2 * row + 1, 2 * mdp.state_index(neighbor), 1)
    source, target = 2 * mdp.state_index(mdp.start_state) + 1, 2 * mdp.state_index(mdp.goal_state)
    for count in range(2):
        queue, parents = deque([source]), {source: None}
        while queue and target not in parents:
            node = queue.popleft()
            for neighbor, capacity in graph[node].items():
                if capacity and neighbor not in parents:
                    parents[neighbor] = node
                    queue.append(neighbor)
        if target not in parents:
            return count
        node = target
        while parents[node] is not None:
            previous = parents[node]
            graph[previous][node] -= 1
            graph[node][previous] += 1
            node = previous
    return 2


def _has_narrow_bottleneck(mdp):
    for state in mdp.trainable_states:
        neighbors = state + np.asarray(mdp.ACTION_DIRECTIONS)
        if any(mdp.is_outside(neighbor) for neighbor in neighbors):
            continue
        free = [neighbor for neighbor in neighbors if not mdp.obstacle_mask[tuple(neighbor[::-1])]]
        if len(free) == 2 and np.all(np.sum(free, axis=0) == 2 * state):
            return True
    return False


def _all_states_reach_endpoints(mdp):
    """Check reachability to both endpoints using the actual nominal transition graph."""
    free = set(np.flatnonzero(~mdp.obstacle_mask.ravel()))
    weights = np.cumprod((1,) + mdp.grid_shape[:-1])
    successors = (mdp.nominal_transitions * weights).sum(axis=-1)
    predecessors = {row: set() for row in free}
    for row in free:
        for successor in successors[row]:
            if successor in free:
                predecessors[successor].add(row)

    def reachable_to(destination):
        queue, visited = deque([destination]), {destination}
        while queue:
            for row in predecessors[queue.popleft()]:
                if row not in visited:
                    visited.add(row)
                    queue.append(row)
        return visited

    goal = mdp.state_index(mdp.goal_state)
    return (reachable_to(mdp.state_index(mdp.start_state)) == free - {goal}
            and reachable_to(goal) == free)


def _has_two_segment_obstacles(mdp):
    """Count hits independently by checking the segment against each obstacle separately."""
    obstacles = mdp.states[mdp.obstacle_mask.ravel()]
    start, goal = mdp.coordinates(mdp.start_state), mdp.coordinates(mdp.goal_state)
    hits = 0
    with patch.object(mdp, "obstacle_mask", np.zeros_like(mdp.obstacle_mask)):
        for state in map(tuple, obstacles):
            mdp.obstacle_mask[state[::-1]] = True
            contact = mdp._collision_check_segment(start, goal)
            mdp.obstacle_mask[state[::-1]] = False
            hits += int(contact is not None and contact[0] == state)
            if hits >= 2:
                return True
    return False


class GridDatasetMapTests(unittest.TestCase):
    def corridor_mdp(self, free, start, goal):
        dimension = free.ndim
        settings = dict(zip(("x_max", "y_max", "z_max"), map(float, free.shape)))
        cls = Robot2DMotionMDP if dimension == 2 else Robot3DMotionMDP
        return cls(
            **settings, grid_shape=free.shape, start_cell=start, goal_cell=goal,
            gamma=0.9, movement_reward=0.0, goal_reward=1.0,
            obstacles=list(map(tuple, np.argwhere(~free))),
        )

    def mdp(self, dimension):
        settings = dict(zip(("x_max", "y_max", "z_max"), (10.0,) * dimension))
        cls = Robot2DMotionMDP if dimension == 2 else Robot3DMotionMDP
        return cls(
            **settings, grid_shape=(10,) * dimension, start_cell=(0,) * dimension,
            goal_cell=(9,) * dimension, gamma=0.9, movement_reward=0.0, goal_reward=1.0,
        )

    def test_two_routes_passage_width_and_endpoint_buffers_in_both_dimensions(self):
        for dimension, counts in ((2, (25, 40)), (3, (300, 400))):
            template = self.mdp(dimension)
            for seed in (7, 19, 31):
                with self.subTest(dimension=dimension, seed=seed):
                    maps = list(training_maps(template, 4, n_obstacles=counts, seed=seed))
                    signatures = {(m.goal_state, m.obstacle_mask.tobytes()) for m in maps}
                    self.assertEqual(len(signatures), 4)
                    for mdp in maps:
                        self.assertEqual(mdp.start_state, (1,) * dimension)
                        self.assertTrue(all(5 <= coordinate <= 8 for coordinate in mdp.goal_state))
                        self.assertTrue(counts[0] <= mdp.obstacle_mask.sum() <= counts[1])
                        obstacles = mdp.states[mdp.obstacle_mask.ravel()]
                        lower = obstacles * mdp.cell_widths
                        upper = (obstacles + 1) * mdp.cell_widths
                        endpoints = [
                            mdp.state_index(mdp.start_state), mdp.state_index(mdp.goal_state),
                        ]
                        positions = mdp.state_coordinates[endpoints, None]
                        closest = np.clip(positions, lower[None], upper[None])
                        clearance = np.linalg.norm(positions - closest, axis=-1).min(axis=1)
                        self.assertTrue(np.all(clearance >= 2.0))
                        self.assertTrue(_has_two_segment_obstacles(mdp))
                        self.assertTrue(_has_wide_route(mdp))
                        self.assertEqual(_route_count(mdp), 2)
                        self.assertTrue(_has_narrow_bottleneck(mdp))
                        self.assertTrue(_all_states_reach_endpoints(mdp))
                    repeated = next(training_maps(template, 2, n_obstacles=counts, seed=seed))
                    np.testing.assert_array_equal(maps[0].obstacle_mask, repeated.obstacle_mask)
                    self.assertEqual(maps[0].goal_state, repeated.goal_state)
            self.assertEqual(template.start_state, (0,) * dimension)
            self.assertFalse(template.obstacle_mask.any())

    def test_straight_passages_accept_two_cells_or_two_by_two_and_reject_smaller_sections(self):
        for dimension, sections in ((2, ((1,), (2,))), (3, ((2, 1), (2, 2)))):
            for section in sections:
                with self.subTest(dimension=dimension, section=section):
                    free = np.zeros((6,) * dimension, dtype=bool)
                    free[(slice(None),) + tuple(slice(1, 1 + width) for width in section)] = True
                    start, goal = (0,) + (1,) * (dimension - 1), (5,) + (1,) * (dimension - 1)
                    mdp = self.corridor_mdp(free, start, goal)
                    blocked = mdp.states[mdp.obstacle_mask.ravel()]
                    corridor = _wide_corridor(mdp, blocked, np.random.default_rng(7))
                    expected = all(width == 2 for width in section)
                    self.assertEqual(corridor is not None, expected)
                    self.assertEqual(_has_wide_route(mdp), expected)
                    if corridor is not None:
                        self.assertFalse(np.any(corridor & mdp.obstacle_mask.ravel()))

    def test_wide_passages_remain_connected_through_turns(self):
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                free = np.zeros((6,) * dimension, dtype=bool)
                free[(slice(0, 4), slice(1, 3)) + (slice(1, 3),) * (dimension - 2)] = True
                if dimension == 2:
                    free[2:4, 1:6] = True
                    start, goal = (0, 1), (2, 5)
                else:
                    free[2:4, 1:3, 1:6] = True
                    start, goal = (0, 1, 1), (2, 1, 5)
                mdp = self.corridor_mdp(free, start, goal)
                blocked = mdp.states[mdp.obstacle_mask.ravel()]
                corridor = _wide_corridor(mdp, blocked, np.random.default_rng(7))
                self.assertIsNotNone(corridor)
                self.assertFalse(np.any(corridor & mdp.obstacle_mask.ravel()))
                axes = tuple(reversed(range(dimension)))
                corridor_grid = corridor.reshape(mdp.obstacle_mask.shape).transpose(axes)
                recovered = self.corridor_mdp(corridor_grid, start, goal)
                self.assertTrue(_has_wide_route(recovered))

    def test_corner_contacts_cannot_substitute_for_a_wide_connection(self):
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                free = np.zeros((4, 4) + (2,) * (dimension - 2), dtype=bool)
                free[:2, :2] = True
                free[2:4, 2:4] = True
                free[1, 2] = True
                start, goal = (0,) * dimension, (3, 3) + (0,) * (dimension - 2)
                mdp = self.corridor_mdp(free, start, goal)
                blocked = mdp.states[mdp.obstacle_mask.ravel()]
                self.assertIsNone(_wide_corridor(mdp, blocked, np.random.default_rng(7)))
                self.assertFalse(_has_wide_route(mdp))

    def test_goals_without_space_for_a_terminal_free_C3_region_are_allowed(self):
        for dimension, counts in ((2, (35, 40)), (3, (250, 350))):
            template = self.mdp(dimension)
            start = (1,) * dimension
            goal = (5, 6) if dimension == 2 else (5,) * dimension
            with self.subTest(dimension=dimension), patch(
                "grid_q_data.randomize_start_goal", return_value=(start, goal),
            ):
                maps = list(training_maps(template, 2, n_obstacles=counts))
                self.assertEqual(len(maps), 2)
                for mdp in maps:
                    self.assertEqual(mdp.goal_state, goal)
                    self.assertEqual(_route_count(mdp), 2)

    def test_connectivity_rejects_isolated_cells_and_goal_bridges(self):
        for dimension in (2, 3):
            cases = []
            free = np.ones((5,) * dimension, dtype=bool)
            corner = np.full(dimension, 4)
            walls = corner - np.eye(dimension, dtype=int)
            for wall in walls:
                free[tuple(wall)] = False
            start, goal = (0,) * dimension, (2,) * dimension
            cases.append(("island", free.copy(), start, goal, False))
            free[tuple(walls[0])] = True
            cases.append(("connected island", free.copy(), start, goal, True))

            free = np.zeros((5,) * dimension, dtype=bool)
            free[(slice(None),) + (1,) * (dimension - 1)] = True
            start, goal = (0,) + (1,) * (dimension - 1), (2,) + (1,) * (dimension - 1)
            cases.append(("goal bridge", free.copy(), start, goal, False))
            free[(slice(None), 2) + (1,) * (dimension - 2)] = True
            cases.append(("goal bypass", free.copy(), start, goal, True))
            for name, free, start, goal, expected in cases:
                with self.subTest(dimension=dimension, case=name):
                    mdp = self.corridor_mdp(free, start, goal)
                    obstacles = mdp.states[mdp.obstacle_mask.ravel()]
                    self.assertEqual(_all_free_states_connected(mdp, obstacles), expected)
                    self.assertEqual(_all_states_reach_endpoints(mdp), expected)

    def test_fewer_than_two_segment_obstacles_are_rejected_in_both_dimensions(self):
        for dimension in (2, 3):
            template = self.mdp(dimension)
            start, goal = (1,) * dimension, (8,) * dimension
            clear = [(4,) + (1,) * (dimension - 1)]
            single = [(4,) * dimension]
            blocked = [[(3,) * dimension, (5,) * dimension], [(4,) * dimension, (6,) * dimension]]
            with (
                self.subTest(dimension=dimension),
                patch("grid_q_data.randomize_start_goal", return_value=(start, goal)),
                patch(
                    "grid_q_data._sample_obstacles", side_effect=[clear, single, *blocked],
                ) as sample,
            ):
                maps = list(training_maps(template, 2, n_obstacles=(1, 2)))
                self.assertEqual(sample.call_count, 4)
                for mdp, expected in zip(maps, blocked):
                    actual = set(map(tuple, mdp.states[mdp.obstacle_mask.ravel()]))
                    self.assertEqual(actual, set(expected))

    def test_segment_counts_distinct_cells_corner_contacts_and_parallel_directions(self):
        for dimension in (2, 3):
            mdp = self.corridor_mdp(
                np.ones((10,) * dimension, dtype=bool), (1,) * dimension, (8,) * dimension,
            )
            first, second = (4,) * dimension, (5,) * dimension
            corners = [(3, 4) + (3,) * (dimension - 2), (4, 3) + (4,) * (dimension - 2)]
            cases = [([], 0), ([first], 1), ([first, first], 1), ([first, second], 2),
                     ([(0,) * dimension, (9,) * dimension], 0), (corners, 2)]
            for obstacles, expected in cases:
                with self.subTest(dimension=dimension, obstacles=obstacles):
                    self.assertEqual(_segment_obstacle_count(mdp, obstacles), expected)
            for start_x, goal_x in ((1, 8), (8, 1)):
                start = (start_x,) + (1,) * (dimension - 1)
                goal = (goal_x,) + (1,) * (dimension - 1)
                mdp = self.corridor_mdp(np.ones((10,) * dimension, dtype=bool), start, goal)
                obstacles = [(4,) + (1,) * (dimension - 1), (5,) + (1,) * (dimension - 1),
                             (4, 3) + (1,) * (dimension - 2)]
                self.assertEqual(_segment_obstacle_count(mdp, obstacles), 2)

    def test_infeasible_counts_fail_without_relaxing_geometry(self):
        for dimension, counts in ((2, (99, 99)), (3, (999, 999))):
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                next(training_maps(self.mdp(dimension), 2, n_obstacles=counts))


if __name__ == "__main__":
    unittest.main()
