"""Algorithm 2: retrain robust Q-learning, deploy, and quantify uncertainty online."""

import numpy as np

from acp import ACP
from Tabular_Agent import Tabular_Agent


class ConformallyRobustController:
    """Own independent nominal/deployment environments and persistent calibration.

    Algorithm 1 starts from zero unless a q_initializer is supplied. Fixed R and C
    reuse the policy on the fixed nominal training MDP. The MDP supplies clone()
    and deploy_policy(). Only actual deployment transitions enter the mismatch
    estimate; pilot calibration does not. Episodes stop at a terminal condition,
    with horizon acting as a step limit. R uses the observed transition count tau_k.
    ACP scores include every sampled disturbance, including terminal transitions.

    R=None enables mismatch counting and estimation; numeric R disables them.
    C=None enables pilots and ACP; numeric C disables them, including C=infinity.
    These choices are independent.
    R_hat is the current estimate and C_hat is the nonnegative ACP prediction.
    R_used and C_used are the actual Algorithm 1 inputs.
    episode_setup(controller, index) runs before each episode and returns metadata.
    Indices are zero-based online and negative for the initial pilot episodes.
    q_initializer(mdp, R_used, C_used) optionally returns the initial Q-table;
    the predictor remains outside the agent and saved experiment results.
    """

    def __init__(
        self, mdp, *, gamma=0.95, horizon=100, training_episodes=2000,
        training_horizon=100, R_hat_init=0.1, delta=0.1, delta_1=None,
        eta=0.01, quantile_max=3.0, calibration_scores=None, calibration_window=100,
        pilot_episodes=20,
        lr_init=0.5, step_start_decay_lr=100000, epsilon_init=1.0, epsilon_lb=0.1,
        epsilon_decay_rate=0.995, seed=None, R=None, C=None,
        episode_setup=None, q_initializer=None,
    ):
        Tabular_Agent._validate_uncertainty(R_hat_init if R is None else R, 0 if C is None else C)
        for name, value in (
            ("horizon", horizon), ("training_episodes", training_episodes),
            ("training_horizon", training_horizon), ("pilot_episodes", pilot_episodes),
        ):
            if name == "pilot_episodes" and C is not None:
                continue
            minimum = 0 if name == "pilot_episodes" else 1
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}.")
        if not np.isfinite(gamma) or not 0 <= gamma < 1:
            raise ValueError("gamma must be in [0, 1).")
        self.gamma = float(gamma)
        self.horizon = int(horizon)
        self.training_episodes = int(training_episodes)
        self.pilot_episodes = int(pilot_episodes) if C is None else 0
        self.fixed_R = None if R is None else float(R)
        self.fixed_C = None if C is None else float(C)
        self.episode_setup = episode_setup
        self.q_initializer = q_initializer
        self.rng = np.random.default_rng(seed)
        self.training_env = mdp.clone(nominal=True, max_episode_steps=training_horizon)
        self.deployment_env = mdp.clone(max_episode_steps=horizon, seed=self._next_seed())
        self.R_hat = float(R_hat_init) if R is None else None
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
        self.acp = None
        self.delta = None
        if self.fixed_C is None:
            self.acp_options = {
                "delta": delta, "delta_1": delta_1,
                "eta": eta, "window_size": calibration_window,
                "quantile_max": quantile_max,
            }
            # Validate calibration settings before any potentially expensive pilot run.
            validated = ACP([0.0], **self.acp_options)
            if calibration_scores is not None:
                self.acp = ACP(calibration_scores, **self.acp_options)
                self.initial_calibration_scores = list(self.acp.scores)
            elif self.pilot_episodes == 0:
                raise ValueError("Provide calibration_scores or set pilot_episodes > 0.")
            self.delta = validated.delta

    def _next_seed(self):
        return int(self.rng.integers(0, 2**32 - 1))

    def initialize_calibration(self):
        if self.fixed_C is not None or self.acp is not None:
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
        if self.fixed_C is None:
            self.initialize_calibration()
        settings = {}
        if self.episode_setup is not None:
            settings = self.episode_setup(self, len(self.history))
        R_used = self.R_hat if self.fixed_R is None else self.fixed_R
        C_used = self.acp.quantile() if self.fixed_C is None else self.fixed_C
        Tabular_Agent._validate_uncertainty(R_used, C_used)
        reused = self.fixed_R is not None and self.fixed_C is not None and self.agent is not None
        if not reused:
            self.agent = Tabular_Agent(
                self.training_env, R=R_used, C=C_used, seed=self._next_seed(),
                **self.agent_options,
            )
            initial_Q = None
            if self.q_initializer is not None:
                initial_Q = self.q_initializer(self.training_env, R_used, C_used)
            self.Q, self.policy = self.agent.robust_q_learning(
                self.training_episodes, exploring_starts=True, record_every=100,
                initial_Q=initial_Q,
            )
        record = self.deployment_env.deploy_policy(self.policy, self.gamma)
        record.update({
            "episode": len(self.history) + 1, "R_used": R_used,
            "C_used": float(C_used),
            "training": self.agent.training_diagnostics.copy(), "training_reused": reused,
        })
        if self.fixed_R is None:
            self.transition_count += len(record["actions"])
            self.mismatch_count += record["episode_mismatches"]
            self.R_hat = self.mismatch_count / self.transition_count
            record.update({
                "R_hat_next": self.R_hat, "transition_count": self.transition_count,
                "mismatch_count": self.mismatch_count,
            })
        if self.fixed_C is None:
            update = self.acp.update(record["score"], predicted_radius=C_used)
            record.update({"C_hat": float(C_used), **update})
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
                    f"Episode {record['episode']}: "
                    f"R={record['R_used']:.3f}, C={record['C_used']:.3f}, "
                    f"steps={record['episode_length']}, goal={record['goal_reached']}",
                )
        return records

