# Conformally robust 2-D robot motion

This repository implements Algorithms 1 and 2 for 2-D robot motion experiments.
Algorithm 1 learns a tabular robust Q-function from the nominal simulator.
Algorithm 2 deploys the policy, estimates the discrete mismatch probability R,
and uses adaptive conformal prediction (ACP) to calibrate a continuous deviation
radius C. Q-learning starts from zero; the transformer/LLM warm start is not implemented.
Deployment records unprojected noisy transitions until goal arrival, obstacle
collision, workspace exit, or the episode step limit. Episode lengths may vary;
the final transition contributes to both R estimation and the episode C score.

## Setup and run the comparison

Run commands from the repository root. Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

On Windows, the workspace's `.\.venv\Scripts\python.exe` can replace `python`.
The robot scripts also support direct execution and an editor's Run Python File action.

```powershell
python -m robot_2d.main_robot_2d_online
python -m robot_2d.plot_online_comparison
```

The settings at the bottom of `robot_2d/main_robot_2d_online.py` run three methods
for seeds `(87,)`, with 30 outer episodes, 4,000 inner training episodes per
training call and 100 pilot episodes.
All online results and figures go to `robot_2d/saved_results/online/`.
Each call to `main()` saves one file named
`results_{method}_R{R}_C{C}_trial_{trial}.pkl`. Seeds are stored in the data.
These defaults provide a starting experiment; increase the outer episode and seed
counts for stronger calibration and performance evidence.

Settings are ordinary function arguments. There is no command-line parser or
separate experiment framework. Edit the variables and `main()` call under
`if __name__ == '__main__'` to configure the comparison. Call `main()` directly
for a single run. For example:

```python
from robot_2d.main_robot_2d_online import main
from robot_2d.plot_online_comparison import plot_comparison

# Run Algorithm 2 alone.
controller = main(
    n_episodes=30, training_episodes=4000, pilot_episodes=100,
    save_plots=False,
)

# Replot saved comparisons without training or deployment.
plot_comparison()
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
| `horizon` | 100 | Maximum deployment and nominal training episode length |
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
| `seed` | 7 | Random seed for a single run |
| `trial` | 1 | Trial number used in the saved filename |
| `save_plots` | `True` | Save the two single-run figures |
| `verbose` | `True` | Print episode progress and the saved path |

The script entry point runs all three methods for seeds `(87,)`.
It calls `main()` once per method and trial;
each call saves its own file. The script explicitly sets 100 pilots; calling `main()`
without that override uses the 20-pilot function default.

Within each comparison trial, methods share the initial pilot calibration scores.
Each method maintains its own subsequent R estimate and calibration history.
Simulation draws ordinary random disturbances, with separate run seeds for the
methods. No disturbance tapes are used. The stored
run seed is `seed + 100000 * method_index` in the requested method order.
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
s_observed = discretize(x_observed)
# Record the transition, including its mismatch and the original Delta.
x = x_observed
# End the episode at a goal, obstacle, workspace exit, or the step limit.
```

At every recorded transition, Delta is zero with probability 0.7 and otherwise
uniform in a disk of radius 1.2. The sampled vector is neither projected nor
resampled, including when it leaves the workspace. The final position, original
vector, and norm are recorded before checking the terminal condition. There are
no subsequent transitions from terminal states and no padding to the step limit.
R, C, and the policy stay fixed within each episode. Training uses a separate
noise-free episodic clone. The physical noise law is not supplied to the learner.

The finite exit outcome is an implementation convention for terminating outside
landings: extended-grid Phi retains their distinct coordinates for observations,
while outside cells share a zero planning value. The agent never takes an action
from the exit outcome during deployment.
A custom `noise_sampler(nominal_position, rng)` can return a bounded 2-D disturbance;
the comparison plotter's analytic mismatch reference assumes the default disk law.

Rewards remain the paper's fixed `r(s,a)` during noisy deployment: a nominal goal
command receives 1, a valid nominal free move receives 0.01, and obstacle or outward
commands receive 0. Terminal sources receive zero. Thus a nominal goal command can
earn reward without actual goal arrival. Record actual success, collision, and
workspace exit separately. When changing rewards, keep movement reward below
`(1-gamma)*goal_reward` for the nominal episodic goal-reaching task.

The localized uncertainty neighborhood contains every cell intersecting the
continuous deviation ball, as in Equations (3)-(4), including obstacle and goal cells.
It also includes the zero-valued failure outcome whenever that ball contains a
point in an outside grid cell. This allows robust Q-learning to account for exit risk
although nominal training itself never leaves the grid. With half-open cells, touching
an upper workspace boundary can reach an outside cell; touching a lower boundary cannot.
It does not select only cell centers inside the ball. C uses physical distance;
rectangular grids account for both cell widths.

## Implementation and update order

`main_robot_2d_online.py` configures `ConformallyRobustController`, calls its
`run()`, then saves and plots. The controller's `run_episode()` is the
single implementation of training and R/ACP updates for all experiment settings.
`Robot2DMotionMDP.deploy_policy()` is the single continuous rollout routine shared
by random-policy pilots and online deployment.
Robot-specific dynamics and plotting remain under `robot_2d/`.

The MDP owns `reward(state, action)`, `terminal_mask`, `state_coordinates`, and
`localized_uncertainty_states(nominal_next_state, C)`. The last method returns
candidate state indices around the observed nominal successor and caches them in
the MDP. `Tabular_Agent` uses these definitions directly; it does not construct
geometry, copy terminal masks, or infer task definitions from Gym environments.
Training uses rewards returned by `env.step()` and the MDP's terminal mask;
Bellman diagnostics use `env.reward()`. Environments must supply this interface
explicitly, including when adapting a bare Gym environment such as FrozenLake.

For each proposed or global episode:

1. Predict the conformal radius using scores available before deployment, and
   select the pre-episode R estimate.
2. Train Algorithm 1 from zero on the nominal simulator with the chosen R and C.
3. Deploy until a terminal condition or the step limit, recording every landing,
   reward, mismatch, and disturbance, including the final transition.
4. Update R as total discrete mismatches divided by total online transitions.
5. Compute the maximum continuous deviation over the episode, score the
   pre-episode prediction, and update ACP and its score window.

Pilot transitions never enter the R estimator. All actual online transitions enter R.
The calibration window applies only to C scores; R uses the entire online history.
The step limit ensures a finite episode length, as required by Assumption 6;
episodes contain at least one transition and need not have a common length.

Exploring starts cycle through shuffled nonterminal state-action pairs. The
controller uses learning rate 0.5 for the first 100,000 training transitions,
then per-pair harmonic decay. Its epsilon schedule starts at 1.0, decays by 0.995
per training episode, and has a floor of 0.1. All changing planning problems use
fresh zero Q-tables. The nominal baseline reuses its fixed policy.
Finite Q-learning approximates the robust optimum; inspect saved Bellman residuals
and unvisited pairs when selecting a common training budget across methods.

## Figures and what they establish

The separate `plot_online_comparison.py` reads the per-run files from
`saved_results/online/` and writes the R, ACP, and online policy comparison figures there.
Its paths are fixed relative to the script. It does not run the robot or retrain policies.

### R estimation: `r_estimation`

The first panel shows individual seed estimates, their mean, and the common true R-star.
The second shows each seed's absolute estimation error against its own cumulative
transition count tau_k, together with the simultaneous bound from Lemma 5:
`sqrt(log(2*tau_k*(tau_k+1)/delta_R)/(2*tau_k))`.
`plot_comparison(delta_R=0.05)` sets its confidence parameter. The bound
applies to each run; it is not a joint confidence statement across seeds.

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
tau_k = sum of actual episode lengths through episode k
R_hat_next = sum of observed mismatch indicators / tau_k
```

Under the default independent sampler and extended uniform Phi, the mismatch
mean is the same R-star at every counted step, conditional on previous observations
and the selected action. This satisfies Assumptions 3 and 5 and the martingale
condition used by Lemma 5. Ending an episode on a noisy terminal landing does not
discard that sample. R weights episodes by their actual transition counts, rather
than averaging per-episode mismatch rates. A custom sampler must independently
satisfy the paper's assumptions; accepting a callable does not verify them.

### ACP calibration: `acp_calibration`

The four panels show first-seed continuous episode scores and pre-episode radii,
cumulative episode coverage, trailing-window coverage, and the mean number of
states in localized neighborhoods compared with the 65-state global support.
The support-size diagnostic averages over nominal targets of nonterminal
state-action pairs. `plot_comparison(window=10)` controls the trailing
coverage window independently of the calibration window used during the run.

Coverage compares ACP with a rolling quantile at the fixed target failure
parameter and with a frozen initial pilot quantile. These two alternatives are
replayed on the same saved score stream. They isolate calibration behavior;
they do not simulate counterfactual policies or establish navigation benefits
for separate frozen-calibration controllers.

ACP uses rank `ceil((1-delta_k)*(N+1))` without interpolation. After observing
score `score_k=max_{0 <= t < episode_length} ||Delta_t||`, it updates:

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

### Online policy comparison: `policy_comparison`

The comparison uses the deployment episodes already recorded by Algorithm 2 and
the nominal and global baselines. The first panel shows cumulative mean observed
discounted reward, `G_k = sum_t gamma^t r(s_t,a_t)`, versus the online episode index.
Thin curves show individual trials; solid curves show their mean.
The second panel compares the fractions of those same episodes ending at the goal,
an obstacle, outside the workspace, or at the step limit. Bars show trial means
and dots show individual trials. These four outcomes are mutually exclusive.

Each episode supplies one observation of its current deployed policy. The figure
describes performance during online adaptation. It does not estimate Theorem 3's
regret, which compares robust optimal values at estimated and true R with the same C.
The comparison reads saved histories without running additional robot episodes.

The R and ACP figures use runs labeled `proposed`.
Single-run `main(save_plots=True)` saves `online_uncertainty_{test}.png` and
`final_policy_{test}.png`, where `{test}` is the method, R/C setting, and trial.

## Distribution shifts with a fixed true R

Assumption 5 allows the unknown distribution D-star-k and its bounded support
C-star-k to change between episodes, while retaining a common conditional
mismatch mean R-star. The distribution stays fixed within each episode.
The default comparison uses one fixed distribution, a special case of this setting.
Changing observed scores or ACP radii does not itself imply a distribution shift.

For a shifted experiment, change `controller.deployment_env.noise_radius` and
`noise_probability` between `run_episode()` calls, keeping the same R-star.
For a disk containing the whole nominal cell, use
`noise_probability = R_star / (1 - cell_area/(pi*noise_radius**2))`, provided this
probability is at most one. For example, the default unit-cell law at radius 1.2
and probability 0.3 has the same R-star as radius 2.4 and probability approximately
0.247354794. The learner receives neither parameter; its cumulative R estimate and
ACP history continue across the shift. The analytic R plot assumes this common
R-star; it does not infer an arbitrary custom sampler's reference.

Changing the radius alone generally changes R-star and violates Assumption 5.
When the true R varies, the cumulative estimator summarizes past transition
mismatch means; the paper's fixed-R concentration and regret statements no longer
apply. Theorem 3 compares robust planning at estimated R with an oracle at the same
calibrated C, rather than guaranteeing navigation success or finite-budget
Q-learning accuracy.

## Saved data and paths

All Algorithm 2 output uses the fixed folder `robot_2d/saved_results/online/`,
independently of the working directory. Each `main()` call saves one results dictionary
in `results_{method}_R{R}_C{C}_trial_{trial}.pkl`. For example:

- `results_proposed_RNone_CNone_trial_1.pkl`: online R estimation and ACP.
- `results_nominal_RNone_CNone_trial_1.pkl`: nominal baseline.
- `results_global_RNone_CNone_trial_1.pkl`: global R-contamination baseline.
- `results_proposed_R0.2_C0.7_trial_1.pkl`: fixed R/C experiment.

`None` in the filename denotes an unspecified planning parameter; method overrides
still determine nominal and global planning. The comparison plotter uses the fixed
pattern `results_{method}_RNone_CNone_trial_*.pkl` for each of the three methods,
including all matching trials. Edit those patterns in the plotter to select other tests.

Each run record contains settings, the configured MDP, initial and pilot scores,
final R and ACP parameter, full online histories, final Q/policy, state-action
visit counts.
Q has shape `(65, 4)` and policy has shape `(65,)` on the
default grid; the failure row is zero. Each history records pre-episode planning R/C, raw
conformal radius, score, miscoverage, ACP parameters before/after, cumulative
counts, training diagnostics, original deviation vectors, continuous positions,
extended-grid quantized positions, episode length, and terminal outcome.
`states` and `positions` contain the initial observation followed by each actual
landing, including the terminal landing; they have length `episode_length + 1`.
The disturbance, action, reward, and mismatch arrays have length `episode_length`.
Trajectory plots connect these successive positions directly.

Single-run figures use the same method, R/C, and trial suffix as the results file.
Comparison figures are `r_estimation`, `acp_calibration`, and `policy_comparison`,
each in PNG and PDF format. These figures share the same folder as the result files.

```python
import pickle
filename = 'robot_2d/saved_results/online/results_proposed_RNone_CNone_trial_1.pkl'
with open(filename, 'rb') as file:
    results = pickle.load(file)
history = results['history']
Q, policy = results['Q'], results['policy']
visit_counts = results['visit_counts']
```

Running the same method, R/C setting, and trial again replaces that file. Choose a
different trial number to retain another run. The script numbers trials from one,
independently of their seeds. The comparison plotter reads the matching files directly.
Results from previous projected, padded, or resetting deployments require rerunning
to produce the current terminal episodes, transition counts, and trajectory fields.

## Algorithm 1 alone and its learning curves

`robot_2d/main_robot_2d.py` trains on the nominal simulator at fixed R/C, without
ACP or online deployment. Its script entry point runs several fixed-R/C cases;
edit the calls at the bottom to select them. The `main()` function defaults to
15 trials and 5,000 training episodes per trial. For a single selected case:

```python
from robot_2d.main_robot_2d import main as train_fixed

curves = train_fixed(R=0.15, C=1.2, training_episodes=2000, n_trials=1)
```

Each call saves `robot_2d_R{R}_C{C}.pkl` under `robot_2d/saved_results/fixedRC/`, containing
one initial-state value curve per trial. Each starts at zero and records
`max_a Q(start,a)` after every training transition. Its index is cumulative
training steps; the agent attribute `evaluation_return` is this learned value,
not a sampled deployment return. General learning-curve and return utilities are
in `utils_tabular.py`.

The companion `robot_2d_R{R}_C{C}_agents.pkl` contains the MDP and trained agents.
Older curve files need retraining to produce this companion file. Use
`python -m robot_2d.plot_fixedRC_eval_returns` for the saved fixed-parameter learning curves
and `python -m robot_2d.plot_fixedRC_value_policy` for saved values and greedy policies.
The value/policy plotter defaults to R=0.2, C=1.0, trial 0; select another case with:

```python
from robot_2d.plot_fixedRC_value_policy import main as plot_policy

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
learning and reference solutions, cumulative accounting across unequal episode
lengths, continuous noise without projection, workspace-exit planning, terminal
transition recording, saved run files and policy plots,
script imports, and controller continuation with retained experiment settings.
