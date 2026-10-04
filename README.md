# Conformally robust 2-D robot motion

`Robot2DMotionMDP` in `robot_2d_motion_mdp.py` replaces FrozenLake for the robot
example. It uses an 8-by-8 grid over `[0, x_max] x [0, y_max]`, sparse obstacle
cells, and four commands: `0: +x`, `1: -x`, `2: +y`, `3: -y`. Grid cells are
specified as `(ix, iy)`; physical positions are `(x, y)`.

Install the runtime dependencies with `python -m pip install -r requirements.txt`.
The workspace also contains a local environment; on Windows it can be used directly
as `.\.venv\Scripts\python.exe` in place of `python` in the commands below.

## Run Algorithms 1 and 2

```powershell
python main_conformal_robust_robot_motion.py
```

`ConformallyRobustController` creates independent nominal and deployment instances.
Every deployment episode calls Algorithm 1 with a fresh zero Q-table. There is no
transformer initialization or reuse of the previous Q-table. `R` is estimated from
all deployment mismatches; the deviation radius is calibrated from episode maximum
continuous deviations. Pilot episodes initialize the calibration window and do not
enter the deployment mismatch counts.

Results are saved under `results/conformal_robot_motion/`, including `results.pkl`
with episode history and calibration scores, and `final_policy.npz` with the Q-table.
The runner does not save an MDP configuration or JSON files.

The main function constructs `Robot2DMotionMDP` directly and lists every constructor
argument. Edit those calls to change the bounds, obstacles, start/goal, rewards,
disturbance, or episode mode. Training and experiment settings are explicit keyword
arguments of `main()`; there is no command-line parser. For example:

```python
from main_conformal_robust_robot_motion import main

controller = main(n_episodes=10, training_episodes=2000, save_plots=False)
```

Provide initial episode maximum deviations through `calibration_scores`, or use
the pilot episodes to initialize calibration.

Set `n_episodes=1` for one outer Algorithm 2 episode: calibrate the radius, train
Algorithm 1 from zero, deploy for `horizon` transitions, and update uncertainty.
`training_episodes` separately controls the inner Q-learning training budget.

## Fixed uncertainty: Algorithm 1

Fixed-`R,C` training uses the same `Tabular_Agent.Robust_Q_learning()` method called
by Algorithm 2. It does not need another main script:

```python
from robot_2d_motion_mdp import Robot2DMotionMDP
from Tabular_Agent import Tabular_Agent

nominal_mdp = Robot2DMotionMDP(noise_probability=0.0, noise_radius=0.0, seed=7)
agent = Tabular_Agent(
    nominal_mdp, gamma=0.95, lr_init=0.5, step_start_decay_lr=100000,
    epsilon_lb=0.1, epsilon_decay_rate=0.995, R=0.15, C=1.2, seed=7,
)
Q, policy = agent.Robust_Q_learning(n_episodes=2000)
```

## Modeling conventions

- Nominal commands snap to an adjacent cell center, including when the robot starts
  away from its current center. Outward commands remain in the current cell.
- Rewards are fixed `r(s, a)`: a nominal command into the goal receives 1, a valid
  free move receives 0.01, and obstacle/outward commands receive 0. Terminal cells
  receive no further reward. Actual noisy success and collision are separate metrics.
  For an episodic nominal task, keep movement reward below `(1 - gamma) * goal_reward`
  so indefinite motion is not preferable to eventual success.
- Obstacle cells remain in the state space and uncertainty neighborhoods. Goal and
  obstacle states have zero-valued absorbing dynamics.
- The localized neighborhood follows Equation (3): it includes cells intersected by
  the continuous deviation ball, rather than only centers inside that ball. Radii
  use physical distance. Rectangular domains use their two cell widths.
- The default disturbance mixes zero deviation with uniform noise in a bounded disk.
  Its injection probability is not the discrete mismatch probability. Positions are
  projected onto the domain, and deviations are measured after projection. A custom
  `noise_sampler(nominal_position, rng)` can supply another bounded disturbance.
- Algorithm 2 executes exactly `H` transitions, including genuine absorbing dynamics
  after a terminal outcome. No observations are invented after an early stop. Those
  dynamics and boundary projection can violate the paper's uniform conditional
  mismatch assumption; the implementation does not claim to verify that assumption.
- ACP uses the exact `ceil((1 - delta_k) * (N + 1))` order statistic and does not clip
  its adaptive failure parameter. Out-of-range ranks produce an unbounded or empty
  conformal prediction. An empty prediction uses radius zero for planning, while
  ACP still counts its prediction as uncovered. Both coverage quantities are saved.
  Pickle and NumPy result files retain infinite radii as numeric values.
- Finite training approximates the optimal robust Q-function. Saved diagnostics
  include the robust Bellman residual and unvisited state-action pairs. The existing
  `evaluation_return` attribute records a learned initial-state value, not a sampled
  deployment return.
- Exploring starts cycle through shuffled nonterminal state-action pairs, ensuring
  complete coverage in each sweep. The controller keeps constant learning rates
  for the first 100,000 training transitions, then use per-pair harmonic decay.

## Validation

```powershell
python -B -m unittest discover -s tests -v
```

Checks include Gymnasium compatibility, geometry, rewards, uncertainty support,
robust value-iteration comparisons, limiting cases, calibration order statistics,
fresh training per episode, cumulative mismatch accounting, and the runner with
one or multiple outer episodes.
