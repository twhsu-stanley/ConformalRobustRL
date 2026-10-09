# Conformally robust 2-D robot motion

This repository implements Algorithms 1 and 2 for 2-D robot motion experiments.
Algorithm 1 learns a tabular robust Q-function from the nominal simulator.
Algorithm 2 deploys the policy, estimates the discrete mismatch probability R,
and uses adaptive conformal prediction (ACP) to calibrate a continuous deviation
radius C. Q-learning starts from zero; the transformer/LLM warm start is not implemented.
Deployment records H unprojected noisy transitions. Exits and obstacle landings
restart from the pre-action position without recording or counting a reset step.
The usual disturbance distribution also applies at the goal.

## Setup and run the comparison

Run commands from the repository root. Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

On Windows, the workspace's `.\.venv\Scripts\python.exe` can replace `python`.
The robot scripts also support direct execution and an editor's Run Python File action.

```powershell
python -m robot_2d.main_robot_2d_online
```

The settings at the bottom of `robot_2d/main_robot_2d_online.py` run three methods
for seeds `(7, 8, 9)`, with 30 outer episodes, 4,000 inner training episodes per
training call, 100 pilot episodes, and 500 final-policy evaluation episodes.
Results and comparison figures go to `robot_2d/saved_results/online_comparison/`.
These defaults provide a starting experiment; increase the outer episode and seed
counts for stronger calibration and performance evidence.

Settings are ordinary function arguments. There is no command-line parser or
separate experiment framework. Edit the variables and `main()` call under
`if __name__ == '__main__'` to configure the comparison. Call `main()` directly
for a single run. For example:

```python
from robot_2d.main_robot_2d_online import main
from robot_2d.plot_online_comparison import plot_comparison

# Run Algorithm 2 alone, with independent evaluation of its final policy.
controller = main(
    n_episodes=30, training_episodes=4000, pilot_episodes=100,
    evaluation_episodes=500, save_plots=False,
)

# Replot saved comparisons without training or deployment.
plot_comparison('robot_2d/saved_results/online_comparison/comparison.pkl')
```

`main()` returns the controller. Its experiment settings persist when calling
`controller.run_episode()` or `controller.run(n_episodes)` to continue in memory.
These continuation calls do not rewrite the files previously saved by `main()`.

## Methods and settings

| Method | Planning R | Planning C | Training |
| --- | --- | --- | --- |
| `proposed` | Cumulative mismatch estimate | Pre-episode ACP radius | Fresh zero Q-table each episode |
| `nominal` | 0 | 0 | Train once per seed and reuse the policy |
| `global` | Its own cumulative mismatch estimate | Positive infinity | Fresh zero Q-table each episode |

The global method implements the globally supported R-contamination comparison
associated with [Wang and Zou, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/3a4496776767aaa99f9804d0905fe584-Paper.pdf).
Using `np.inf` includes all 65 next states: 64 grid states and the workspace-failure state.
Our comparison supplies this baseline with the same estimation procedure for R;
that online estimator is part of our experimental protocol.

```python
controller = main(method='proposed', R=None, C=None, save_plots=False)
controller = main(method='nominal', save_plots=False)
controller = main(method='global', save_plots=False)
controller = main(method='proposed', R=0.2, C=0.7, save_plots=False)
```

`R=None` uses the cumulative estimate for planning; numeric R holds the planning
parameter fixed while mismatch observations continue to be recorded. `C=None`
uses ACP; numeric C holds the radius fixed and disables the adaptive failure
parameter update while retaining observed scores. Method overrides take priority:
nominal planning always uses R=C=0, and global planning always uses C=infinity.
With `C=None`, ACP diagnostics are still collected for the nominal and global methods.

| Argument to `main()` | Default | Meaning |
| --- | --- | --- |
| `n_episodes` | 30 | Outer deployment episodes |
| `training_episodes` | 4000 | Inner Algorithm 1 training episodes per call |
| `horizon` | 100 | Deployment transitions and nominal training horizon |
| `gamma` | 0.95 | Reward discount |
| `initial_R` | 0.1 | Estimate used before any online deployment |
| `target_failure` | 0.1 | Target long-run episode miscoverage |
| `initial_failure` | `None` | Initial ACP parameter; `None` uses the target |
| `eta` | 0.01 | ACP update step size |
| `calibration_window` | 100 | Maximum number of stored calibration scores |
| `pilot_episodes` | 20 | Random-policy pilots when scores are not supplied |
| `calibration_scores` | `None` | Optional initial episode maximum deviations |
| `noise_probability` | 0.3 | Simulator disturbance injection probability |
| `noise_radius` | 1.2 | Simulator disk support radius |
| `evaluation_episodes` | 0 | Independent final-policy rollouts; 0 skips evaluation |
| `seed` | 7 | Random seed for a single run |
| `save_plots` | `True` | Save the two single-run figures |
| `verbose` | `True` | Print episode progress and the saved path |

The script entry point runs all three methods for seeds `(7, 8, 9)`, with 500
final-policy evaluation episodes. Its comparison loop calls `main()` and combines
the saved results. The script explicitly sets 100 pilots; calling `main()`
without that override uses the 20-pilot function default.

Within each comparison trial, methods share the initial pilot calibration scores.
Each method maintains its own subsequent R estimate and calibration history.
Simulation draws ordinary random disturbances, with separate run seeds for the
methods and fresh draws for evaluation. No disturbance tapes are used. The stored
`trial_seed` groups methods for comparison; the actual run seed is
`trial_seed + 100000 * method_index` in the requested method order.
All methods deploy under the same specified physical noise law.

## Robot model and continuous deployment

The workspace is `[0, 8) x [0, 8)`, divided into 8-by-8 unit cells. Half-open cells
give a consistent tie convention at every grid boundary, including outside cells.
Start and goal positions are `(0.5, 0.5)` and `(7.5, 7.5)`. Obstacle cells are
`(2, 1), (5, 1), (1, 3), (4, 3), (6, 4), (2, 5), (5, 6)`.
Cells are indexed by `(ix, iy)` with discrete state `s = ix + nx * iy`.
Actions `0, 1, 2, 3` command `+x, -x, +y, -y`, respectively.
The additional state `failure_state=64` represents workspace exit, giving 65 Q-table
rows. It aggregates outside cells for finite planning and has no cell center. The
nominal function f(s,a) is unchanged on the original grid states, including outward
commands that remain in the current cell. The failure row has zero value and reward.
Edit the direct `Robot2DMotionMDP` construction in the main scripts to change
bounds, grid, obstacles, start/goal, or rewards.

Section III defines the nominal dynamics as `F(x,a)=f(Phi(x),a)`:
the action targets the neighboring cell center even when the current continuous
position is away from its cell center. An outward command targets the current
cell center. At the goal, `f(s,a)=s` for every action. `mdp.quantize(x)` extends Phi
to the uniform grid throughout R^2, without clipping:

```text
Phi(x) = (floor(x / cell_widths) + 0.5) * cell_widths
```

`mdp.discretize(x)` returns the finite grid index inside the workspace and the
exit outcome outside. Deployment retains continuous x and performs:

```text
s = discretize(x)
a = policy[s]
Delta = sample_from_unknown_distribution()
x_observed = F(x,a) + Delta
s_observed = Phi(x_observed)
# Record this attempted transition, its mismatch, and Delta.
x = x if outside_or_obstacle(x_observed) else x_observed
```

At every recorded transition, Delta is zero with probability 0.7 and otherwise uniform in a
disk of radius 1.2. The sampled vector is neither projected nor resampled, including
when it leaves the workspace. The exiting position is recorded outside X, together
with the original vector and norm. An exit or obstacle landing restores the exact
pre-action position before the next action. This recovery produces no recorded
transition, mismatch, disturbance, or reward and does not reset the H-step counter.
Goal states receive the same fresh disturbance draws as all other grid states;
they are never padded with forced zero-noise transitions. Deployment records
exactly H attempts, with R/C and the policy fixed throughout the block. Training
uses a separate noise-free episodic clone. The physical
noise law is not supplied to the learner.

The extended Phi and external restart convention should be stated in the paper.
Outside grid cells share a zero planning value through the finite exit outcome;
the agent never takes an action from that outcome during deployment.
A custom `noise_sampler(nominal_position, rng)` can return a bounded 2-D disturbance;
the comparison plotter's analytic mismatch reference assumes the default disk law.

Rewards remain the paper's fixed `r(s,a)` during noisy deployment: a nominal goal
command receives 1, a valid nominal free move receives 0.01, and obstacle or outward
commands receive 0. Terminal sources receive zero. Thus a nominal goal command can
earn reward without actual goal arrival. Record actual success, collision, and
workspace exit separately. When changing rewards, keep movement reward below
`(1-gamma)*goal_reward` for the nominal episodic goal-reaching task.

The localized uncertainty neighborhood contains every cell intersecting the
continuous deviation ball, as in Equation (3), including obstacle and goal cells.
It also includes the zero-valued failure outcome whenever that ball contains a
point in an outside grid cell. This allows robust Q-learning to account for exit risk
although nominal training itself never leaves the grid. With half-open cells, touching
an upper workspace boundary can reach an outside cell; touching a lower boundary cannot.
It does not select only cell centers inside the ball. C uses physical distance;
rectangular grids account for both cell widths.

## Implementation and update order

`main_robot_2d_online.py` configures `ConformallyRobustController`, calls its
`run()`, then evaluates, saves, and plots. The controller's `run_episode()` is the
single implementation of training and R/ACP updates for all experiment settings.
`Robot2DMotionMDP.deploy_policy()` is the single continuous rollout routine shared
by random-policy pilots, online deployment, and frozen-policy evaluation.
Robot-specific dynamics and plotting remain under `robot_2d/`.

For each proposed or global episode:

1. Predict the conformal radius using scores available before deployment, and
   select the pre-episode R estimate.
2. Train Algorithm 1 from zero on the nominal simulator with the chosen R and C.
3. Deploy the resulting policy for exactly H attempts and record each actual source
   and observed landing, reward, mismatch, and disturbance. Restart after exits or
   collisions without adding a transition. Continue applying noise at the goal.
4. Update R as total discrete mismatches divided by total online transitions.
5. Compute the maximum continuous deviation over the episode, score the
   pre-episode prediction, and update ACP and its score window.

Pilot and evaluation transitions never enter the R estimator. Evaluation freezes
the final policy and never updates R or ACP. All H online attempts enter R;
external restarts add no samples. The calibration window applies only to C scores;
R uses the entire online history.

Exploring starts cycle through shuffled nonterminal state-action pairs. The
controller uses learning rate 0.5 for the first 100,000 training transitions,
then per-pair harmonic decay. Its epsilon schedule starts at 1.0, decays by 0.995
per training episode, and has a floor of 0.1. All changing planning problems use
fresh zero Q-tables. The nominal baseline reuses its fixed policy.
Finite Q-learning approximates the robust optimum; inspect saved Bellman residuals
and unvisited pairs when selecting a common training budget across methods.

## Figures and what they establish

The separate `plot_online_comparison.py` loads `comparison.pkl` and writes three
figures in PNG and PDF formats. It does not run the robot or retrain policies.

### R estimation: `r_estimation`

The first panel shows individual seed estimates, their mean, and the common true R-star.
The second shows RMSE over seeds around R-star versus
the number of online transitions, with an `n^(-1/2)` reference slope.

The disturbance injection probability 0.3 is not the discrete mismatch probability.
Under the default unprojected noise law, the mismatch probability for every grid
state-action pair is approximately 0.2336854404, including goal, obstacle, edge, and
corner nominal successors. Remaining in the nominal unit cell means that Delta lies
in the same centered unit square at every grid location. Leaving X counts as a
mismatch on the extended grid. Thus the reference is
`R_star = 0.3 * (1 - 1/(pi*1.2**2))`.
The plotter computes the disk/square intersection for this offline reference only.

After k online episodes, the controller estimate is:

```text
R_hat_next = sum of observed mismatch indicators / (k*H)
```

There is no absorption weighting: every counted transition samples the same
disturbance law, including transitions from the goal. Restarts use past observations
to select the next source, without rejecting or redrawing any disturbance. Under
the default independent sampler and extended uniform Phi, the mismatch
mean is R-star at every counted step. The H-sample blocks therefore retain
`tau_k=k*H` and the centered-martingale condition used by Lemma 5. A custom sampler
must independently satisfy these conditions; accepting a callable does not verify them.

### ACP calibration: `acp_calibration`

The four panels show first-seed continuous episode scores and pre-episode radii,
cumulative episode coverage, trailing-window coverage, and the mean number of
states in localized neighborhoods compared with the 65-state global support.
The support-size diagnostic averages over nominal targets of nonterminal
state-action pairs. `plot_comparison(..., window=100)` controls the trailing
coverage window independently of the calibration window used during the run.

Coverage compares ACP with a rolling quantile at the fixed target failure
parameter and with a frozen initial pilot quantile. These two alternatives are
replayed on the same saved score stream. They isolate calibration behavior;
they do not simulate counterfactual policies or establish navigation benefits
for separate frozen-calibration controllers.

ACP uses rank `ceil((1-delta_k)*(N+1))` without interpolation. After observing
score `score_k=max_t ||Delta_t||`, it updates:

```text
miss_k = 1{score_k > predicted_C_k}
delta_next = delta_k + eta*(target_failure - miss_k)
```

The adaptive failure parameter is not clipped. Nonpositive ranks produce an
empty prediction with raw C=-infinity; planning uses C=0. Ranks above N produce
C=+infinity. The plot marks both tails and separately shows planning-radius
coverage when it differs from raw conformal coverage. Infinite radii are retained
in saved data. A calibrated radius need not equal the physical support radius 1.2.

For adaptive runs, the exact bookkeeping identity after K episodes is:

```text
coverage(K) = 1 - target_failure + (delta_after_K - initial_delta)/(eta*K)
```

The coverage target concerns long-run episode coverage of continuous deviations.
It is not a goal-success guarantee, a collision bound, or a guarantee for every
short window. Assess coverage together with prediction efficiency and extended
prediction tails. Under a stationary law, a frozen quantile may already work well;
improved ACP performance must be demonstrated by the data.

### Navigation comparison: `policy_comparison`

Each method's final policy is evaluated on fresh random rollouts under the same
specified noise law. Bars show means across run seeds; dots show individual seed
results. The six displayed quantities are:

| Quantity | Meaning |
| --- | --- |
| Goal success | Fraction reaching the actual goal within H transitions |
| Collision | Fraction entering an actual obstacle cell |
| Workspace exit | Fraction with at least one outside landing within H attempts |
| Goal not reached by H | Fraction never reaching the goal within H attempts |
| Actual discounted reward | Mean of `sum_t gamma^t r(s_t,a_t)` during deployment |
| Discounted actual goal arrival | Mean of `gamma^(T_goal-1)` for successes and 0 for failures |

Saved evaluation data also include the mean of the worst 10% of discounted returns,
using fractional tail mass when necessary. First goal/collision/exit steps and
collision/exit counts are stored in online histories. Because deployment continues
after goal arrival and restarts after failures, success, collision, and exit rates
can overlap; they are not mutually exclusive outcomes. Goal success and timeout
sum to one. Returns include H attempted transitions across restarts, rather than
the value of one uninterrupted rollout. Examine all outcomes and reward together.

The current performance figure compares final policies; it does not evaluate
checkpoint policies throughout training or add confidence intervals. To assess
repeatability, increase independent run seeds and evaluation rollouts. Independent
training/deployment trials, rather than overlapping episode windows, are the units
for uncertainty across seeds. Learned robust values alone do not establish better
physical performance, and a short development run is not conclusive evidence.

For this nonnegative-reward map, nominal terminal states have robust V=0, so the global worst
case has `min_s V(s)=0`. Its Bellman target reduces to
`r(s,a)+gamma*(1-R)*V(f(s,a))`. Global and nominal planning can therefore select
the same route, depending on action ties and finite training. Localization can
distinguish nearby obstacle exposure; globally depressed planning values do not
by themselves establish poor deployed navigation. Report the observed comparison.

The R and ACP figures use runs labeled `proposed`. The performance figure is
produced only when every saved run has final evaluation data. To obtain all three,
include the proposed method and set `evaluation_episodes` to a positive number.
Single-run `main(save_plots=True)` instead saves `online_uncertainty.png` and
`final_policy.png`.

## Stationarity and a changing true R

The main comparison holds the unknown physical disturbance law fixed throughout
all H transitions of every run, including at the goal. Changing observed scores,
estimates, or calibrated radii does not itself
mean that the underlying law changed. Separate calls can compare different fixed
noise settings; they do not implement a distribution shift within one run.

If a true uniform mismatch mean R-star(j) changes across episodes, the cumulative
estimator used before episode k is centered on its historical average, not the
current R-star(k). With equal horizons, that reference is the average over episodes
1 through k-1. Error relative to the current value includes both sampling error
and the difference between the historical average and current value.
For example, 500 episodes at R=0.1 followed by 500 at R=0.3 produce expected
cumulative R=0.2 when the current value is 0.3.

Consequently, the fixed-R concentration and vanishing average-regret claims do
not automatically extend to arbitrary time variation. Tracking a changing R would
require a different estimator, such as a sliding window or discounting, and new
drift assumptions and bounds. Algorithm 1 can still solve each supplied fixed
planning problem, and ACP's coverage bookkeeping remains meaningful; ACP coverage
does not correct an underestimated R. Increasing the disk radius also changes
discrete mismatch probabilities, so a support shift is not generally a C-only shift.
Such an experiment would be an extension beyond the current stationary comparison.

## Saved data and paths

Default paths are resolved relative to the scripts, independently of the working
directory. A user-supplied relative `output_dir` is relative to the working directory.

| Run | Default directory | Files |
| --- | --- | --- |
| `main(method='proposed')` | `robot_2d/saved_results/online/` | `results.pkl`, `final_policy.npz`, optional single-run PNGs |
| `main(method='nominal'/'global')` | Corresponding subfolder of `online/` | Same per-run files |
| Script comparison | `robot_2d/saved_results/online_comparison/` | Per-method/trial folders, `comparison.pkl`, comparison PNGs/PDFs |

`results.pkl` contains settings, the configured MDP, initial and pilot scores,
final R and ACP parameter, full online histories, final Q/policy, and optional
evaluation metrics and samples. `final_policy.npz` contains Q, policy, and
state-action visit counts. Q has shape `(65, 4)` and policy has shape `(65,)` on the
default grid; the failure row is zero. Each history records pre-episode planning R/C, raw
conformal radius, score, miscoverage, ACP parameters before/after, cumulative
counts, training diagnostics, original deviation vectors, attempted continuous
landings, extended-grid quantized positions, and collision/exit events and counts.
`source_states` and `source_positions` record the actual source of each attempt;
`states[1:]` and `positions[1:]` record its landing before any restart. Their
successive entries need not form a continuous path because recovery is unrecorded.
Trajectory plots draw each source-to-landing segment without drawing reset moves.
`comparison.pkl` collects run records and adds the grouping `trial_seed`.

```python
import pickle
import numpy as np

with open('robot_2d/saved_results/online_comparison/comparison.pkl', 'rb') as file:
    runs = pickle.load(file)
first_run = runs[0]
history = first_run['history']
evaluation = first_run['evaluation']

with np.load('robot_2d/saved_results/online/final_policy.npz') as file:
    Q, policy = file['Q'], file['policy']
```

These paths describe different example runs; use the per-method folder for a
policy saved by the comparison script. Rerunning with the same output directory
replaces files at those names, so use distinct directories for separate experiments.
Results from previous projected or absorbing deployments require rerunning to
produce the current restart observations and trajectory fields.

## Algorithm 1 alone and its learning curves

`robot_2d/main_robot_2d.py` trains on the nominal simulator at fixed R/C, without
ACP or online deployment. Its script entry point runs several fixed-R/C cases;
edit the calls at the bottom to select them. The `main()` function defaults to
15 trials and 5,000 training episodes per trial. For a single selected case:

```python
from robot_2d.main_robot_2d import main as train_fixed

curves = train_fixed(R=0.15, C=1.2, training_episodes=2000, n_trials=1)
```

Each call saves `robot_2d_R{R}_C{C}.pkl` under `robot_2d/saved_results/`, containing
one initial-state value curve per trial. Each starts at zero and records
`max_a Q(start,a)` after every training transition. Its index is cumulative
training steps; the agent attribute `evaluation_return` is this learned value,
not a sampled deployment return. General learning-curve and return utilities are
in `utils_tabular.py`.

The companion `robot_2d_R{R}_C{C}_agents.pkl` contains the MDP and trained agents.
Older curve files need retraining to produce this companion file. Use
`python -m robot_2d.plot_eval_returns` for the saved fixed-parameter learning curves
and `python -m robot_2d.plot_value_policy` for saved values and greedy policies.
The value/policy plotter defaults to R=0.2, C=1.0, trial 0; select another case with:

```python
from robot_2d.plot_value_policy import main as plot_policy

fig = plot_policy(R=0.15, C=1.2, trial=0)
```

It shows `V(s)=max_a Q(s,a)` at physical cell coordinates, marks start, goal, and
obstacles, and draws action arrows only at nonterminal cells. Robot trajectory,
value/policy, and uncertainty-history plotting helpers are in `robot_2d/utils.py`.

## Validation

```powershell
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s robot_2d/tests -v
```

The suites cover ACP update order and extended tails, zero-initialized robust
learning and reference solutions, cumulative mismatch accounting, continuous
noise without projection, workspace-exit planning, uncounted restarts, noise at the goal,
saved run files and policy plots,
script imports, and controller continuation with retained experiment settings.
