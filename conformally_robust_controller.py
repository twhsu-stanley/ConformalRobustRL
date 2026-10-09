"""Algorithm 2: retrain robust Q-learning, deploy, and quantify uncertainty online."""

import numpy as np

from acp import ACP
from Tabular_Agent import Tabular_Agent


class ConformallyRobustController:
    """Own independent nominal/deployment environments and persistent calibration.

    Algorithm 2 starts Algorithm 1 from zero every episode; the nominal baseline
    reuses its fixed policy. The MDP supplies clone() and deploy_policy(). Only
    actual deployment transitions enter the mismatch estimate; pilot calibration
    and evaluation do not. Absorbing transitions are executed and counted.

    R/C=None selects online estimation/calibration; numeric values hold them fixed.
    method='nominal' plans with R=C=0; method='global' plans with C=infinity.
    """

    def __init__(
        self, mdp, *, gamma=0.95, horizon=100, training_episodes=2000,
        training_horizon=100, initial_R=0.1, target_failure=0.1, initial_failure=None,
        eta=0.01, calibration_scores=None, calibration_window=100, pilot_episodes=20,
        lr_init=0.5, step_start_decay_lr=100000, epsilon_init=1.0, epsilon_lb=0.1,
        epsilon_decay_rate=0.995, seed=None, method="proposed", R=None, C=None,
    ):
        if method not in ("proposed", "nominal", "global"):
            raise ValueError("method must be 'proposed', 'nominal', or 'global'.")
        Tabular_Agent._validate_uncertainty(initial_R if R is None else R, 0 if C is None else C)
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
        self.method = method
        self.fixed_R = None if R is None else float(R)
        self.fixed_C = None if C is None else float(C)
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

        def random_policy(state):
            return int(self.rng.integers(self.deployment_env.action_space.n))

        for _ in range(self.pilot_episodes):
            record = self.deployment_env.deploy_policy(
                random_policy, self.gamma, seed=self._next_seed(),
            )
            self.pilot_scores.append(record["score"])
        self.calibrator = ACP(self.pilot_scores, **self.calibration_options)
        self.initial_calibration_scores = list(self.calibrator.scores)
        return list(self.calibrator.scores)

    def run_episode(self):
        """Train and deploy once, then update uncertainty from this episode only."""
        self.initialize_calibration()
        conformal_radius = self.calibrator.quantile() if self.fixed_C is None else self.fixed_C
        planning_radius = self.calibrator.planning_radius(conformal_radius)
        R_used = self.R_estimate if self.fixed_R is None else self.fixed_R
        if self.method == "nominal":
            R_used, planning_radius = 0.0, 0.0
        elif self.method == "global":
            planning_radius = np.inf
        reused = self.method == "nominal" and self.agent is not None
        if not reused:
            self.agent = Tabular_Agent(
                self.training_env, R=R_used, C=planning_radius, seed=self._next_seed(),
                **self.agent_options,
            )
            self.Q, self.policy = self.agent.robust_q_learning(
                self.training_episodes, exploring_starts=True, record_every=100,
            )
        record = self.deployment_env.deploy_policy(
            self.policy, self.gamma, seed=self._next_seed(),
        )
        self.transition_count += len(record["actions"])
        self.mismatch_count += record["episode_mismatches"]
        self.R_estimate = self.mismatch_count / self.transition_count
        if self.fixed_C is None:
            update = self.calibrator.update(record["score"], predicted_radius=conformal_radius)
        else:
            update = {
                "miscoverage": int(record["score"] > conformal_radius),
                "delta_before": self.calibrator.delta, "delta_after": self.calibrator.delta,
            }
            self.calibrator.scores.append(record["score"])
        record.update({
            "episode": len(self.history) + 1, "R_used": R_used,
            "R_next": self.R_estimate, "conformal_radius": float(conformal_radius),
            "planning_radius": float(planning_radius),
            "planned_miscoverage": int(record["score"] > planning_radius), **update,
            "transition_count": self.transition_count, "mismatch_count": self.mismatch_count,
            "training": self.agent.training_diagnostics.copy(), "training_reused": reused,
        })
        self.history.append(record)
        return record

    def run(self, n_episodes, *, verbose=False):
        if not isinstance(n_episodes, (int, np.integer)) or n_episodes < 1:
            raise ValueError("n_episodes must be a positive integer.")
        records = []
        for _ in range(n_episodes):
            record = self.run_episode()
            records.append(record)
            if verbose:
                print(
                    f"{self.method}, seed {self.seed}, episode {record['episode']}: "
                    f"R={record['R_used']:.3f}, C={record['planning_radius']:.3f}, "
                    f"goal={record['goal_reached']}",
                )
        return records

