"""Algorithm 2: retrain robust Q-learning, deploy, and quantify uncertainty online."""

import numpy as np

from acp import ACP
from Tabular_Agent import Tabular_Agent


class ConformallyRobustController:
    """Own independent nominal/deployment environments and persistent calibration.

    Algorithm 2 starts Algorithm 1 from zero every episode; the nominal baseline
    reuses its policy when the environment stays fixed. The MDP supplies clone()
    and deploy_policy(). Only actual deployment transitions enter the mismatch
    estimate; pilot calibration does not. Episodes stop at a terminal condition,
    with horizon acting as a step limit. R uses the observed transition count tau_k.

    R/C=None selects online estimation/calibration; numeric values hold them fixed.
    method='nominal' plans with R=C=0. The R-contamination baselines fix R at
    0.1 or 0.3 and use C=infinity.
    R_hat is the current estimate and C_hat is the nonnegative ACP prediction.
    R_used and C_used are the actual Algorithm 1 inputs after method overrides.
    episode_setup(controller, index) runs before each episode and returns metadata.
    Indices are zero-based online and negative for the initial pilot episodes.
    """

    def __init__(
        self, mdp, *, gamma=0.95, horizon=100, training_episodes=2000,
        training_horizon=100, R_hat_init=0.1, delta=0.1, delta_1=None,
        eta=0.01, calibration_scores=None, calibration_window=100, pilot_episodes=20,
        lr_init=0.5, step_start_decay_lr=100000, epsilon_init=1.0, epsilon_lb=0.1,
        epsilon_decay_rate=0.995, seed=None, method="proposed", R=None, C=None,
        episode_setup=None,
    ):
        if method not in ("proposed", "nominal", "R_contamination_0.1", "R_contamination_0.3"):
            raise ValueError(f"Unknown method: {method}.")
        Tabular_Agent._validate_uncertainty(R_hat_init if R is None else R, 0 if C is None else C)
        for name, value in (
            ("horizon", horizon), ("training_episodes", training_episodes),
            ("training_horizon", training_horizon), ("pilot_episodes", pilot_episodes),
        ):
            minimum = 0 if name == "pilot_episodes" else 1
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}.")
        if not np.isfinite(gamma) or not 0 <= gamma < 1:
            raise ValueError("gamma must be in [0, 1).")
        if not np.isfinite(R_hat_init) or not 0 <= R_hat_init <= 1:
            raise ValueError("R_hat_init must be in [0, 1].")
        self.gamma = float(gamma)
        self.horizon = int(horizon)
        self.training_episodes = int(training_episodes)
        self.pilot_episodes = int(pilot_episodes)
        self.method = method
        self.fixed_R = None if R is None else float(R)
        self.fixed_C = None if C is None else float(C)
        self.episode_setup = episode_setup
        self.rng = np.random.default_rng(seed)
        self.training_env = mdp.clone(nominal=True, max_episode_steps=training_horizon)
        self.deployment_env = mdp.clone(max_episode_steps=horizon, seed=self._next_seed())
        self.R_hat = float(R_hat_init)
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
        self.acp_options = {
            "delta": delta, "delta_1": delta_1,
            "eta": eta, "window_size": calibration_window,
        }
        # Validate calibration settings before any potentially expensive pilot run.
        validated = ACP([0.0], **self.acp_options)
        self.acp = None
        if calibration_scores is not None:
            self.acp = ACP(calibration_scores, **self.acp_options)
            self.initial_calibration_scores = list(self.acp.scores)
        elif self.pilot_episodes == 0:
            raise ValueError("Provide calibration_scores or request at least one pilot episode.")
        self.delta = validated.delta

    def _next_seed(self):
        return int(self.rng.integers(0, 2**32 - 1))

    def initialize_calibration(self):
        if self.acp is not None:
            return

        def random_policy(state):
            return int(self.rng.integers(self.deployment_env.action_space.n))

        for episode in range(-self.pilot_episodes, 0):
            if self.episode_setup is not None:
                self.episode_setup(self, episode)
            record = self.deployment_env.deploy_policy(random_policy, self.gamma)
            self.pilot_scores.append(record["score"])
        self.acp = ACP(self.pilot_scores, **self.acp_options)
        self.initial_calibration_scores = list(self.acp.scores)

    def run_episode(self):
        """Train and deploy once, then update uncertainty from this episode only."""
        self.initialize_calibration()
        settings = {}
        if self.episode_setup is not None:
            settings = self.episode_setup(self, len(self.history))
        if self.fixed_C is None:
            C_hat = self.acp.quantile()
        else:
            Tabular_Agent._validate_uncertainty(0.0, self.fixed_C)
            C_hat = max(0.0, float(self.fixed_C))
        C_used = C_hat
        R_used = self.R_hat if self.fixed_R is None else self.fixed_R
        if self.method == "nominal":
            R_used, C_used = 0.0, 0.0
        elif self.method == "R_contamination_0.1":
            R_used, C_used = 0.1, np.inf
        elif self.method == "R_contamination_0.3":
            R_used, C_used = 0.3, np.inf
        reused = self.method == "nominal" and self.agent is not None
        if not reused:
            self.agent = Tabular_Agent(
                self.training_env, R=R_used, C=C_used, seed=self._next_seed(),
                **self.agent_options,
            )
            self.Q, self.policy = self.agent.robust_q_learning(
                self.training_episodes, exploring_starts=True, record_every=100,
            )
        record = self.deployment_env.deploy_policy(self.policy, self.gamma)
        self.transition_count += len(record["actions"])
        self.mismatch_count += record["episode_mismatches"]
        self.R_hat = self.mismatch_count / self.transition_count
        if self.fixed_C is None:
            update = self.acp.update(record["score"], predicted_radius=C_hat)
        else:
            update = {
                "miscoverage": int(record["score"] > C_hat),
                "delta_k": self.acp.delta_k, "delta_next": self.acp.delta_k,
            }
            self.acp.scores.append(record["score"])
        record.update({
            "episode": len(self.history) + 1, "R_used": R_used,
            "R_hat_next": self.R_hat, "C_hat": float(C_hat),
            "C_used": float(C_used),
            "planned_miscoverage": int(record["score"] > C_used), **update,
            "transition_count": self.transition_count, "mismatch_count": self.mismatch_count,
            "training": self.agent.training_diagnostics.copy(), "training_reused": reused,
        })
        record.update(settings)
        self.history.append(record)
        return record

    def run(self, n_deployment_episodes, *, verbose=False):
        if not isinstance(n_deployment_episodes, (int, np.integer)) or n_deployment_episodes < 1:
            raise ValueError("n_deployment_episodes must be a positive integer.")
        records = []
        for _ in range(n_deployment_episodes):
            record = self.run_episode()
            records.append(record)
            if verbose:
                print(
                    f"{self.method}, episode {record['episode']}: "
                    f"R={record['R_used']:.3f}, C={record['C_used']:.3f}, "
                    f"steps={record['episode_length']}, goal={record['goal_reached']}",
                )
        return records

