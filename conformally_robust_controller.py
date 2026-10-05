"""Algorithm 2: retrain robust Q-learning, deploy, and quantify uncertainty online."""

import numpy as np

from acp import ACP
from Tabular_Agent import Tabular_Agent


class ConformallyRobustController:
    """Own independent nominal/deployment environments and persistent calibration.

    Every outer episode starts Algorithm 1 from zero. Only actual deployment
    transitions enter the cumulative mismatch estimate. Pilot calibration does
    not enter that estimate. Absorbing transitions are executed and counted, so
    the uniform conditional mismatch assumption must be assessed separately.
    """

    def __init__(
        self, mdp, *, gamma=0.95, horizon=100, training_episodes=2000,
        training_horizon=100, initial_R=0.1, target_failure=0.1, initial_failure=None,
        eta=0.01, calibration_scores=None, calibration_window=100, pilot_episodes=20,
        lr_init=0.5, step_start_decay_lr=100000, epsilon_init=1.0, epsilon_lb=0.1,
        epsilon_decay_rate=0.995, seed=None,
    ):
        for name, value in (
            ("horizon", horizon), ("training_episodes", training_episodes),
            ("training_horizon", training_horizon), ("pilot_episodes", pilot_episodes),
        ):
            minimum = 0 if name == "pilot_episodes" else 1
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}.")
        if not np.isfinite(gamma) or not 0 <= gamma < 1:
            raise ValueError("gamma must be in [0, 1).")
        if not np.isfinite(initial_R) or not 0 <= initial_R <= 1:
            raise ValueError("initial_R must be in [0, 1].")
        self.gamma = float(gamma)
        self.horizon = int(horizon)
        self.training_episodes = int(training_episodes)
        self.pilot_episodes = int(pilot_episodes)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.training_env = mdp.clone(
            nominal=True, episode_mode="episodic", max_episode_steps=training_horizon,
        )
        self.deployment_env = mdp.clone(episode_mode="fixed_horizon", max_episode_steps=horizon)
        self.R_estimate = float(initial_R)
        self.initial_R = float(initial_R)
        self.transition_count = 0
        self.mismatch_count = 0
        self.history = []
        self.pilot_scores = []
        self.initial_calibration_scores = []
        self.agent = None
        self.Q = None
        self.policy = None
        self.agent_options = {
            "gamma": self.gamma, "lr_init": lr_init,
            "step_start_decay_lr": step_start_decay_lr, "epsilon_init": epsilon_init,
            "epsilon_lb": epsilon_lb, "epsilon_decay_rate": epsilon_decay_rate,
        }
        self.calibration_options = {
            "target_failure": target_failure, "initial_failure": initial_failure,
            "eta": eta, "window_size": calibration_window,
        }
        # Validate calibration settings before any potentially expensive pilot run.
        validated = ACP([0.0], **self.calibration_options)
        self.calibrator = None
        if calibration_scores is not None:
            self.calibrator = ACP(calibration_scores, **self.calibration_options)
            self.initial_calibration_scores = list(self.calibrator.scores)
        elif self.pilot_episodes == 0:
            raise ValueError("Provide calibration_scores or request at least one pilot episode.")
        self.target_failure = validated.target_failure

    def _next_seed(self):
        return int(self.rng.integers(0, 2**32 - 1))

    def initialize_calibration(self):
        if self.calibrator is not None:
            return list(self.calibrator.scores)
        for _ in range(self.pilot_episodes):
            state, _ = self.deployment_env.reset(seed=self._next_seed())
            score = 0.0
            for step in range(self.horizon):
                action = int(self.rng.integers(self.deployment_env.action_space.n))
                position = self.training_env.nominal_next_position(state, action)
                state, _, terminated, truncated, info = self.deployment_env.step(action)
                score = max(score, float(np.linalg.norm(info["position"] - position)))
                self._check_horizon(step, terminated, truncated)
            self.pilot_scores.append(score)
        self.calibrator = ACP(self.pilot_scores, **self.calibration_options)
        self.initial_calibration_scores = list(self.calibrator.scores)
        return list(self.calibrator.scores)

    def _check_horizon(self, step, terminated, truncated):
        if step + 1 < self.horizon and (terminated or truncated):
            raise RuntimeError("Deployment ended before the required fixed horizon.")

    def run_episode(self):
        self.initialize_calibration()
        conformal_radius = self.calibrator.quantile()
        planning_radius = self.calibrator.planning_radius(conformal_radius)
        R_used = self.R_estimate
        self.agent = Tabular_Agent(
            self.training_env, R=R_used, C=planning_radius, seed=self._next_seed(),
            **self.agent_options,
        )
        self.Q, self.policy = self.agent.robust_q_learning(
            self.training_episodes, exploring_starts=True, record_every=100,
        )
        state, info = self.deployment_env.reset(seed=self._next_seed())
        states = [state]
        positions = [info["position"].copy()]
        actions, rewards, deviations, mismatches = [], [], [], []
        absorbing, boundary_attempts, boundary_violations = [], [], []
        goal_step = None
        collision_step = None
        discounted_return = 0.0
        discount = 1.0
        for step in range(self.horizon):
            action = int(self.policy[state])
            nominal_state = self.training_env.nominal_next_state(state, action)
            nominal_position = self.training_env.coordinates(nominal_state)
            successor, reward, terminated, truncated, info = self.deployment_env.step(action)
            deviation = float(np.linalg.norm(info["position"] - nominal_position))
            mismatch = int(successor != nominal_state)
            self.transition_count += 1
            self.mismatch_count += mismatch
            discounted_return += discount * reward
            discount *= self.gamma
            actions.append(action)
            rewards.append(float(reward))
            deviations.append(deviation)
            mismatches.append(mismatch)
            absorbing.append(bool(info["absorbing_transition"]))
            boundary_attempts.append(bool(info["boundary_attempt"]))
            boundary_violations.append(bool(info["boundary_violation"]))
            if info["goal_reached"] and goal_step is None:
                goal_step = step + 1
            if info["collision"] and collision_step is None:
                collision_step = step + 1
            state = successor
            states.append(state)
            positions.append(info["position"].copy())
            self._check_horizon(step, terminated, truncated)
        score = max(deviations)
        self.R_estimate = self.mismatch_count / self.transition_count
        update = self.calibrator.update(score, predicted_radius=conformal_radius)
        active = ~np.asarray(absorbing)
        active_mismatches = np.asarray(mismatches)[active]
        active_rate = float(np.mean(active_mismatches)) if np.any(active) else None
        record = {
            "episode": len(self.history) + 1, "R_used": R_used,
            "R_next": self.R_estimate, "conformal_radius": float(conformal_radius),
            "planning_radius": planning_radius, "score": score,
            "planned_miscoverage": int(score > planning_radius), **update,
            "transition_count": self.transition_count, "mismatch_count": self.mismatch_count,
            "episode_mismatches": sum(mismatches), "active_mismatch_rate": active_rate,
            "absorbing_transitions": sum(absorbing), "discounted_return": discounted_return,
            "goal_reached": goal_step is not None, "collision": collision_step is not None,
            "goal_step": goal_step, "collision_step": collision_step,
            "boundary_attempts": sum(boundary_attempts),
            "boundary_violations": sum(boundary_violations),
            "training": self.agent.training_diagnostics.copy(),
            "states": np.asarray(states), "positions": np.asarray(positions),
            "actions": np.asarray(actions), "rewards": np.asarray(rewards),
            "deviations": np.asarray(deviations), "mismatches": np.asarray(mismatches),
            "absorbing": np.asarray(absorbing),
        }
        self.history.append(record)
        return record

    def run(self, n_episodes):
        if not isinstance(n_episodes, (int, np.integer)) or n_episodes < 1:
            raise ValueError("n_episodes must be a positive integer.")
        return [self.run_episode() for _ in range(n_episodes)]

