# Conformally robust 2-D robot motion

This repository implements Algorithms 1 and 2 of the paper. Algorithm 1 trains
a robust Q-function using deterministic nominal transitions. Algorithm 2 deploys
the policy, estimates the discrete mismatch probability R, and calibrates the
continuous deviation radius C using adaptive conformal prediction (ACP).

## Run

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m robot_2d.main_robot_2d_online
python -m robot_2d.plot_online_comparison
```

On Windows, `.\.venv\Scripts\python.exe` can replace `python`.
The scripts also support direct execution. Configure experiments through the
ordinary arguments and the settings under `if __name__ == "__main__"`; there is
no command-line parser.

The online script runs five trials with seeds `(23, 67, 87, 101, 922)`, 100
deployment episodes per method, 5,000 nominal training episodes per policy
computation, and 100 random-policy pilot episodes for adaptive C. It uses discount
0.9, a 100-transition episode limit, initial R estimate 0.1, ACP target and initial
delta 0.1, learning rate eta 0.1, and a calibration window of 100 scores.

The map is fixed across all episodes, methods, and trials: 28 hard-coded obstacle
cells on a 10-by-10 grid, start center (1.5, 1.5), and goal center (7.5, 6.5).
The reward is 1 when a nonterminal state's nominal command targets the goal,
and 0 otherwise. Methods share the same episode-by-episode disturbance-support
schedule and use separate random draws.

| Method | R used for training | C used for training |
| --- | --- | --- |
| `proposed` | Cumulative mismatch estimate | Pre-episode ACP radius |
| `nominal` | 0 | 0 |
| `R_contamination_0.1` | Fixed at 0.1 | Positive infinity |
| `R_contamination_0.5` | Fixed at 0.5 | Positive infinity |

The R-contamination baselines use prescribed R values as in Wang and Zou.
When both R and C are fixed, the controller trains once and reuses the policy.
Otherwise it recomputes the policy each deployment episode.

## Global grid and terminal states

The grid extends over the entire plane. A state is an integer coordinate pair
`(ix, iy)`, whose physical cell center is
`((ix + 0.5) * dx, (iy + 0.5) * dy)`. Cells are half-open:
`[ix*dx, (ix+1)*dx) x [iy*dy, (iy+1)*dy)`.

`mdp.discretize(x)` implements the global Phi using coordinate-wise floor,
and `mdp.coordinates(state)` returns the center of any grid cell. The configured
workspace is `[0, 10) x [0, 10)`. In particular, a position exactly on an upper
workspace boundary belongs to an outside cell.

Every outside cell is a distinct terminal state. There is no symbolic exit state,
extra exit row, projection, or clipping of outside observations. For example:

```python
s = mdp.discretize((10.2, 2.7))  # (10, 2)
center = mdp.coordinates(s)      # (10.5, 2.5)
assert mdp.is_outside(s)
assert mdp.is_terminal(s)
```

Goal, obstacle, and outside states are terminal, with reward and continuation
value zero. Their nominal transitions are self-loops. Episodes stop at terminal
arrival or the practical step limit; no resets or padding occur within an episode.

Q and policy remain dense arrays for the finite workspace cells. Row
`ix + nx*iy` corresponds to workspace state `(ix, iy)`; obtain it through
`mdp.state_index(state)`. On the configured grid, Q has shape `(100, 4)` and
policy has shape `(100,)`. Outside cells need no stored rows.
`mdp.values_at(states, values)` returns zero for any terminal state and looks
up nonterminal workspace values. The finite `terminal_mask`, `states`, and
`state_coordinates` arrays describe the workspace table.

## Nominal training and localized uncertainty

Actions 0, 1, 2, 3 command +x, -x, +y, -y. Each nominal command targets a
neighboring cell center. An outward command retains the current cell center:
this is a choice of the nominal model f. During deployment its disturbed outcome
can still leave the workspace.

Algorithm 1 samples only nominal transitions `s_next = f(s, a)`.
`Tabular_Agent.initialize_q()` currently fills Q with zero and is the single
extension point for a custom or LLM initializer. Terminal rows are zeroed after
initialization and never receive TD updates. Initial cached values and diagnostics
are derived from the initialized table.

For finite C, `mdp.localized_uncertainty_states(nominal_next_state, C)` returns
actual grid-coordinate pairs, including outside cells, using equations (3)-(4):

```text
distance_to_cell = ||max(abs(f(s,a) - cell_center) - cell_widths/2, 0)||_2
include the cell when distance_to_cell <= C
```

The distance is the infimum distance to the cell, including equality for half-open
cells. The same formula applies when the nominal successor is itself a goal,
obstacle, or outside state. A terminal nominal successor does not reduce the
neighborhood to that state alone.

The MDP caches finite neighborhoods. `mdp.worst_case_value(...)` combines these
neighborhoods with terminal-aware value lookup. For C=infinity it minimizes the
workspace values together with zero, representing all outside terminal values
without enumerating an infinite grid. Explicit neighborhood enumeration therefore
requires finite C.

The agent uses the robust TD target:

```text
reward + gamma * ((1-R) * V(nominal_next_state)
                  + R * worst_case_next_value)
```

A terminal successor contributes V=0. A time-limit truncation at a nonterminal
successor retains its continuation value. Exploring starts cycle through shuffled
nonterminal state-action pairs. The controller uses learning rate 0.5 for the
first 100,000 training transitions, followed by per-pair harmonic decay, and an
epsilon schedule starting at 1.0 with decay 0.995 and floor 0.1.

## Deployment, estimation, and ACP

The shared `Robot2DMotionMDP.deploy_policy()` routine serves both pilots and
online deployment. From continuous position x, it samples:

```text
s = Phi(x)
nominal_position = f(s, a)
sampled_position = nominal_position + Delta
sampled_state = Phi(sampled_position)
mismatch = (sampled_state != f(s,a))
```

The original sampled endpoint and disturbance are never projected or resampled.
The segment checker finds the first obstacle or goal contact. On contact,
the recorded final state is that terminal cell and the final position is its
contact marker. The raw sampled endpoint remains recoverable as
`nominal_positions + deviation_vectors` and determines the mismatch.
An outside landing without earlier contact retains its sampled position and
its actual outside cell coordinates.

For each deployment episode, the method-agnostic controller:

1. Applies the episode's disturbance law and chooses R_used and C_used.
2. Trains Algorithm 1, or reuses a policy when both inputs are fixed.
3. Deploys until termination or the episode limit, recording every transition.
4. If R is adaptive, updates R_hat from cumulative mismatches divided by cumulative
   deployment transitions. Pilot transitions do not enter this estimator.
5. If C is adaptive, scores the maximum disturbance norm over **all** episode
   transitions, including the final terminal transition, and updates ACP.

Every episode has at least one transition and therefore a score, including
episodes terminating on their first transition. Zero is a valid score.
Pilot episodes use the same scoring routine.

ACP predicts before appending the current score. Its rank is
`ceil((1-delta_k)*(N+1))`. Nonpositive ranks return zero; ranks above N return
`quantile_max`, which defaults to 3 in this experiment. Delta is not clipped:

```text
miscoverage = 1{score > C_hat}
delta_next = delta_k + eta * (delta - miscoverage)
```

The calibration window has fixed maximum length. R instead uses the entire online
transition history. A calibrated radius need not equal the true support bound;
coverage concerns the episode deviation score, rather than navigation success.
Theorem 3's regret concerns R estimation at the same calibrated C, rather than
the empirical deployment returns.

## Disturbance distribution and shift

The unknown simulator law stays fixed within each episode. With zero-based
episode index j and period P=100:

```text
C_star_j = 0.5 + cos(pi*((j % P) + 0.5)/P)**2
```

This varies smoothly between approximately 0.5 and 1.5. At each transition,
Delta is zero with probability 1-R_star. Otherwise a signed coordinate direction
is chosen uniformly and the displacement length is uniform between
`(0.5+C_star_j)/2` and `C_star_j`. Every nonzero draw leaves the nominal unit
cell, so the discrete mismatch probability is exactly R_star, set to 0.4 in the
script. This preserves the paper's common conditional mismatch mean while the
support changes. The learner is not given the disturbance law or its parameters.

Pilots use schedule indices `-pilot_episodes, ..., -1`; deployment starts at zero.
The schedule at a deployment index is identical across methods even when pilots
are disabled. Controller continuation preserves the map, schedule, and accumulated
R/ACP history. It does not rewrite previously saved files.

## Results and figures

Algorithm 2 saves one dictionary per run in `robot_2d/saved_results/online/`:

```text
results_{method}_R{fixed_R}_C{fixed_C}_trial_{trial}.pkl
```

The seed is stored inside the file. Results contain the configured MDP, settings,
history, final Q/policy, and visit counts. Adaptive components add their estimates
and calibration settings. No separate policy file duplicates Q or policy.

Each history stores grid-coordinate states and continuous positions, of shape
`(episode_length + 1, 2)`, along with actions, rewards, original disturbances,
nominal positions, mismatches, score, outcomes, and training diagnostics.
The grid center of a stored state is computed using `mdp.coordinates(state)`;
a separate array of quantized positions is unnecessary.

`plot_online_comparison()` takes no path argument and reads this fixed directory.
It writes:

- `r_estimation.png`: cumulative R estimates and the true R_star.
- `acp_calibration.png`: first-trial scores/radii/support and delta history.
- `policy_comparison.png`: cumulative mean discounted returns and outcome rates.
- `trajectories_{method}_R{R}_C{C}_trial_{trial}.png`: all deployment trajectories.

The comparison uses the recorded Algorithm 2 episodes, with no additional
evaluation rollouts. Thin return curves and outcome dots show individual trials.
Single-run plots can also be requested through `main(save_plots=True)`.

**Regenerate saved runs from the previous symbolic-exit formulation before using
the updated plotters.** Old pickles contain different state encodings, table
shapes, and ACP scores. No compatibility conversion is provided. The existing
filenames and output locations are retained; rerunning a setting replaces its file.

## Algorithm 1 alone

```powershell
python -m robot_2d.main_robot_2d
python -m robot_2d.plot_fixedRC_eval_returns
python -m robot_2d.plot_fixedRC_value_policy
```

Edit the fixed-R/C calls at the bottom of `main_robot_2d.py`. Outputs go to
`robot_2d/saved_results/fixedRC/`. Each setting saves the initial-state value
curves in `robot_2d_R{R}_C{C}.pkl` and the MDP/trained agents in
`robot_2d_R{R}_C{C}_agents.pkl`. The curves show learned V(start) against
training transitions; they are not sampled deployment returns.

## Validation

```powershell
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s robot_2d/tests -v
```

Tests cover global discretization, distinct outside cells, cell-distance
neighborhoods, terminal values and initialization, nominal/reference learning,
terminal-inclusive calibration, mismatch accounting, controller continuation,
saved results, and plotting.
