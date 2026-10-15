# Conformally robust 2-D robot motion

This repository implements Algorithms 1 and 2 for 2-D robot motion experiments.
Algorithm 1 learns a tabular robust Q-function from the nominal simulator.
Algorithm 2 deploys the policy, estimates the discrete mismatch probability R,
and uses adaptive conformal prediction (ACP) to calibrate a continuous deviation
radius C. Q-learning starts from zero; the transformer/LLM warm start is not implemented.
Deployment samples unprojected noisy endpoints and checks each straight transition
segment for obstacle or goal contact using grid traversal. The first contact ends
the episode and sets the recorded terminal state and position to the contact marker.
The sampled endpoint remains recoverable as nominal position plus sampled deviation;
it determines the discrete mismatch, including on the terminal transition.
Episode lengths may vary. Terminal transitions enter R estimation but not the ACP
score; a nonterminal transition at the episode step limit still enters the score.

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

The settings at the bottom of `robot_2d/main_robot_2d_online.py` currently run proposed
and nominal policies for seeds `(23,)`, with 40 outer episodes, 5,000 inner training
episodes per call, and 200 pilot episodes for ACP.
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
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP
from robot_2d.utils import randomize_obstacles
import numpy as np

# Run Algorithm 2 alone.
mdp = Robot2DMotionMDP(x_max=10, y_max=10, grid_shape=(10, 10), obstacles=[])
randomize_obstacles(mdp, np.random.default_rng(87))
controller = main(
    mdp=mdp, n_deployment_episodes=100, training_episodes=4000, pilot_episodes=100,
    R_star=0.2, shift_period=100, save_plots=False,
)

# Replot saved comparisons without training or deployment.
plot_comparison()
```

`main()` returns the controller. Its experiment settings persist when calling
`controller.run_episode()` or `controller.run(n_deployment_episodes)` to continue in memory.
These continuation calls do not rewrite the files previously saved by `main()`.

## Methods and settings

| Method | Planning R | Planning C | Training |
| --- | --- | --- | --- |
| `proposed` | Cumulative mismatch estimate | Pre-episode ACP radius | Fresh zero Q-table each episode |
| `nominal` | 0 | 0 | Train once per trial and reuse the policy |
| `R_contamination_0.1` | Fixed at 0.1 | Positive infinity | Train once per trial and reuse the policy |
| `R_contamination_0.3` | Fixed at 0.3 | Positive infinity | Train once per trial and reuse the policy |

The two R-contamination methods implement the globally supported comparison
associated with [Wang and Zou, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/3a4496776767aaa99f9804d0905fe584-Paper.pdf).
Using `np.inf` includes all 101 next states on the configured 10-by-10 grid:
100 grid states and the exit state. Their planning R stays fixed at 0.1 or 0.3
throughout deployment. They collect raw deployment observations without estimating R.

```python
controller = main(mdp=mdp, method='proposed', R=None, C=None, save_plots=False)
controller = main(mdp=mdp, method='nominal', save_plots=False)
controller = main(mdp=mdp, method='R_contamination_0.1', save_plots=False)
controller = main(mdp=mdp, method='R_contamination_0.3', save_plots=False)
controller = main(mdp=mdp, method='proposed', R=0.2, C=0.7, save_plots=False)
```

`main_robot_2d_online.py` owns the method settings and passes only R/C to the controller.
The controller has no method argument or method-specific behavior. `R=None` enables
cumulative mismatch counting and estimation; numeric R fixes the planning parameter
and disables that estimator. `C=None` enables pilots and ACP; numeric C, including
infinity, fixes the radius and disables all calibration. These choices are independent:

| R input | C input | R estimation | ACP | Training |
| --- | --- | --- | --- | --- |
| `None` | `None` | Enabled | Enabled | Retrain each episode |
| Numeric | `None` | Disabled | Enabled | Retrain each episode |
| `None` | Numeric | Enabled | Disabled | Retrain each episode |
| Numeric | Numeric | Disabled | Disabled | Train once and reuse on the fixed MDP |

Method overrides occur only in `main_robot_2d_online.py`:
nominal planning always uses R=C=0; the two R-contamination methods always use
their named fixed R and C=infinity, regardless of supplied R/C arguments.
Proposed runs with `C=None` initialize and update ACP. Baselines skip calibration pilots,
ignore calibration inputs, and save no ACP fields or uncertainty-history figure.

| Argument to `main()` | Default | Meaning |
| --- | --- | --- |
| `mdp` | Required | Robot geometry and rewards; this experiment requires unit cells |
| `n_deployment_episodes` | 30 | Outer deployment episodes |
| `training_episodes` | 4000 | Inner Algorithm 1 training episodes per call |
| `horizon` | 100 | Maximum deployment and nominal training episode length |
| `gamma` | 0.95 | Reward discount |
| `R_hat_init` | 0.1 | Estimate used before any online deployment |
| `delta` | 0.1 | Target long-run episode miscoverage |
| `delta_1` | `None` | Initial ACP parameter; `None` uses the target |
| `eta` | 0.01 | ACP update step size |
| `calibration_window` | 100 | Maximum number of stored calibration scores |
| `pilot_episodes` | 20 | Random-policy pilots when scores are not supplied |
| `calibration_scores` | `None` | Optional initial maxima of nonterminal deviations |
| `R_star` | 0.2 | Fixed true mismatch probability in the simulator |
| `shift_period` | 100 | Number of episodes in one support-bound cycle |
| `seed` | 7 | Random seed for a single run |
| `trial` | 1 | Trial number used in the saved filename |
| `save_plots` | `True` | Save the final policy figure and any active uncertainty history |
| `verbose` | `True` | Print episode progress and the saved path |

The script entry point selects methods and seeds at the bottom of the file.
It calls `main()` once per method and trial;
each call saves its own file. The script explicitly sets 200 pilots; calling `main()`
without that override uses the 20-pilot function default.

Within each comparison trial, proposed runs estimate R when `R=None` and collect
calibration scores when `C=None`. Baselines perform neither update.
Simulation draws ordinary random disturbances, with separate run seeds for the
methods. No disturbance tapes are used. The stored
run seed is `seed + 100000 * method_index` in the requested method order.
Each trial generates one obstacle map from its base seed before running any
method. The map stays fixed for all pilots, online episodes, and methods in that
trial. A new trial generates its own map. All methods share the same support
schedule and true R-star, while retaining distinct training/deployment seeds.
The script currently requests 20 obstacle cells; the map generator defaults to 10.
For separate calls to `main()`, pass the same prepared MDP and matching `shift_period`.
`main()` preserves the supplied map, and continuing a controller keeps that map.

`pilot_episodes` supplies the initial calibration dataset S_cal required by Algorithm 2.
When `calibration_scores` is absent, the robot runs this many random-policy episodes
under the deployment noise law and records each episode's maximum nonterminal
disturbance norm. A pilot that terminates on its first transition supplies no score
and is skipped. An observed zero deviation remains a valid calibration score.
Only these scores initialize ACP; pilot transitions never enter the online R estimate
or the policy performance comparison. Supplying `calibration_scores` skips pilots,
and `pilot_episodes=0` is allowed when those scores are supplied.
Calibration settings apply only when `C=None`, including when R is fixed.
Fixed C needs no calibration scores or pilots, including when R is adaptive.

Seeds control reproducibility of simulated randomness and are optional: `seed=None`
uses fresh randomness. The fixed-R/C runner seeds each trial's Q-learning agent;
its deterministic nominal MDP needs no seed. The online controller derives seeds for
the deployment MDP and each new training agent from the run's generator. Each generator
is initialized once; nominal training resets and deployment episodes do not reseed it.
Pilot actions use the controller's generator, exploration uses the agent's generator,
and disturbances use the MDP's generator. A separate generator initialized from the
base trial seed samples the obstacle map once in the trial loop. Plotting uses no randomness.

The MDP retains Gymnasium's optional `reset(seed=...)` argument. An explicit seed
restarts its disturbance generator for repeatable sampling; `reset()` preserves
the generator and continues its random stream. Resetting the robot's position itself
is deterministic. A run seed is stored once in the online results for reproducibility.

## Robot model and continuous deployment

The main scripts configure the closed workspace `[0, 10] x [0, 10]`, divided into
10-by-10 unit cells, with start `(0.5, 0.5)` and goal `(9.5, 9.5)`.
The uncertainty neighborhood uses closed grid cells. Discretization chooses the
upper-index cell at internal ties and an interior cell at the upper workspace boundary.
Every trial samples exactly ten distinct obstacle cells.
The Chebyshev distance in cell indices from every obstacle to both start and goal
must be at least two. Maps are resampled if they disconnect start from goal.
The same map is used for nominal training and physical deployment and stays fixed
throughout the trial. `set_obstacles()` rebuilds the MDP's reward
table, terminal mask, trainable states, and nominal transitions together.
The standalone fixed-R/C runner retains its fixed obstacles `(5, 1), (1, 3), (5, 6)`.
Cells are indexed by `(ix, iy)` with discrete state `s = ix + nx * iy`.
Actions `0, 1, 2, 3` command `+x, -x, +y, -y`, respectively.
The additional state `exit_state=100` represents workspace exit, giving 101 Q-table
rows. It aggregates outside cells for finite planning and has no cell center. The
nominal function f(s,a) is unchanged on the original grid states, including outward
commands that remain in the current cell. The failure row has zero value and reward.
Edit the direct `Robot2DMotionMDP` construction in the main scripts to change
bounds, grid, start/goal, or rewards. Online obstacle placement is handled in
`robot_2d/main_robot_2d_online.py`.

Section III defines the nominal dynamics as `F(x,a)=f(Phi(x),a)`:
the action targets the neighboring cell center even when the current continuous
position is away from its cell center. An outward command targets the current
cell center. At terminal states, the nominal model has self-loops, but no action is
taken after terminal arrival. Inside the closed workspace, `mdp.quantize(x)` returns
the nearest grid center, with the tie convention above:

```text
Phi(x) = argmin_{s in S} ||x - s||_2, x in X
Phi_plus(x) = Phi(x) if x in X else exit
```

`mdp.discretize(x)` returns the index of Phi_plus(x). It checks the original continuous
position against the closed workspace, so only points strictly outside X become exits.
Outside cell centers returned by `mdp.quantize(x)` are trajectory diagnostics;
they are not coordinates of the symbolic exit state. Deployment retains continuous x
and performs:

```text
s = discretize(x)
a = policy[s]
Delta = sample_from_unknown_distribution()
x_sampled = F(x,a) + Delta
s_sampled = discretize(x_sampled)
mismatch = (s_sampled != f(s,a))
contact = mdp._collision_check_segment(x, x_sampled)
s, x = (s_sampled, x_sampled) if contact is None else contact
# Record the transition, including its endpoint-based mismatch and original Delta.
# End at a contact, workspace exit, or step limit; exclude terminal Delta from ACP.
```

The online experiment uses the shifting axis-disturbance law described below.
The sampled vector is neither projected nor
resampled, including when it leaves the workspace. The final position marks the
first obstacle or goal contact when present. The original vector and norm remain
recorded even on this terminal transition, but are excluded from the ACP score. There are
no subsequent transitions from terminal states and no padding to the step limit.
R, C, and the policy stay fixed within each episode. Training uses a separate
noise-free episodic clone. The physical noise law is not supplied to the learner.

The exit outcome implements the symbolic `s_out` in Section III. Outside landings
without an earlier obstacle or goal contact retain their original continuous
coordinates and share this zero-valued planning state.
Episode resets reject goal, obstacle, and exit states, whether supplied as a state index
or a continuous position. Every episode begins at a nonterminal state.
A custom `noise_sampler(nominal_position, rng)` can return a bounded 2-D disturbance.
Online plots read the saved true R-star. The disk-intersection reference remains
available for earlier results using the MDP's zero/uniform-disk sampler.

Rewards remain the paper's fixed `r(s,a)` during noisy deployment: a nominal goal
command receives 1, a valid nominal free move receives 0.01, and obstacle or outward
commands receive 0. Terminal sources receive zero. Thus a nominal goal command can
earn reward without actual goal arrival. Record actual success, collision, and
workspace exit separately. When changing rewards, keep movement reward below
`(1-gamma)*goal_reward` for the nominal episodic goal-reaching task.

Goal, obstacle, and exit states have Q=V=0. The TD update uses these zero values for
the nominal successor and for terminal candidates in the worst-case minimum. The final
training transition receives its immediate reward before the episode ends. A step-limit
truncation at a nonterminal state preserves its continuation value in the TD update.

The localized uncertainty neighborhood follows Equations (4)-(5). It contains every
grid state whose closed cell has distance at most C from the nominal successor, including
obstacle and goal cells. For the rectangular workspace, the distances are:

```text
distance_to_cell = ||max(abs(f(s,a) - cell_center) - cell_widths/2, 0)||_2
distance_to_exit = min(f_x, f_y, x_max - f_x, y_max - f_y)
```

The exit state is included when `distance_to_exit <= C`, including equality on every
boundary. This is the infimum distance to `R^2 \ X`; equality includes the exit state
even though a disturbance ball tangent to the closed workspace has no outside point.
C uses physical distance, and rectangular grids account for both cell widths.

## Implementation and update order

`main_robot_2d_online.py` configures `ConformallyRobustController`, calls its
`run()`, then saves and plots. The controller's `run_episode()` is the
single implementation of training and R/ACP updates for all experiment settings.
The trial loop prepares the obstacle map before running the methods. An optional
episode-setup callback sets the disturbance law before pilots and before each online
training/deployment episode. It also runs during continuation.
`Robot2DMotionMDP.deploy_policy()` is the single continuous rollout routine shared
by random-policy pilots and online deployment.
Robot-specific dynamics and plotting remain under `robot_2d/`.
Method names, default R/C inputs, and display labels are defined together in
`METHOD_SETTINGS` in `main_robot_2d_online.py`; the comparison plotter reads that table.

Names follow the paper: `R_hat` is the current estimate, `R_hat_next` is its
post-episode update, `C_hat` is the ACP prediction, and `R_star`/`C_star_k` describe
simulator truth. `R_used` and `C_used` are the actual Algorithm 1 inputs after
runner configuration. `delta` is the fixed ACP target, `delta_1` its initial parameter,
and `delta_k`/`delta_next` its adaptive values before/after the episode update.
The MDP uses `start_state`, `goal_state`, and `exit_state`.

The MDP owns `reward(state, action)`, `terminal_mask`, `state_coordinates`, and
`localized_uncertainty_states(nominal_next_state, C)`. The last method returns
candidate state indices around the observed nominal successor and caches them in
the MDP. `Tabular_Agent` uses these definitions directly; it does not construct
geometry, copy terminal masks, or infer task definitions from Gym environments.
Training uses rewards returned by `env.step()` and the MDP's terminal mask;
Bellman diagnostics use `env.reward()`. Environments must supply this interface
explicitly, including when adapting a bare Gym environment such as FrozenLake.

For each online episode:

1. Set the episode's physical disturbance law while keeping the trial map fixed.
2. Use the R estimate when R is adaptive, otherwise fixed R. Use the ACP prediction
   when C is adaptive, otherwise fixed C.
3. Train Algorithm 1 from zero on the nominal simulator with the chosen R and C;
   runs with both inputs fixed reuse their policy after the first training call.
4. Deploy until a segment contacts an obstacle or goal, the endpoint exits the
   workspace, or the step limit is reached. Record every transition and disturbance;
   a contact changes the terminal marker without changing the sampled disturbance.
5. If R is adaptive, update it as cumulative mismatches divided by cumulative transitions.
6. If C is adaptive, score the maximum nonterminal continuous deviation against the
   pre-episode prediction and update ACP and its score window. If the first transition
   is terminal, save `score=None` and skip ACP: the calibration window and adaptive
   delta remain unchanged. A later termination retains the maximum from earlier steps.

Pilot transitions never enter the R estimator. With adaptive R, all online transitions enter it.
The calibration window applies only to C scores; R uses the entire online history.
The step limit ensures a finite episode length, as required by Assumption 6;
episodes contain at least one transition and need not have a common length.

Exploring starts cycle through shuffled nonterminal state-action pairs. The
controller uses learning rate 0.5 for the first 100,000 training transitions,
then per-pair harmonic decay. Its epsilon schedule starts at 1.0, decays by 0.995
per training episode, and has a floor of 0.1. All changing planning problems use
fresh zero Q-tables. All fixed-input runs train once on the fixed trial map.
Finite Q-learning approximates the robust optimum; inspect saved Bellman residuals
and unvisited pairs when selecting a common training budget across methods.

## Figures and what they establish

The separate `plot_online_comparison.py` reads the per-run files from
`saved_results/online/` and writes the R, ACP, online policy comparison, and trajectory
figures there.
Its paths are fixed relative to the script. It does not run the robot or retrain policies.

### R estimation: `r_estimation`

The single panel shows individual seed estimates, their mean, and the common true R-star
against deployment episode k.

The new experiment has true R-star exactly equal to `R_star`, default 0.2.
Every nonzero axis disturbance leaves the nominal unit cell, including for goal,
obstacle, edge, and corner nominal successors. An outside landing counts as a mismatch.
This reference is saved for plotting and is not supplied to Q-learning.

After k online episodes, the controller estimate is:

```text
tau_k = sum of actual episode lengths through episode k
R_hat_next = sum of observed mismatch indicators / tau_k
```

Under the default independent sampler and augmented discretization Phi_plus, the mismatch
mean is the same R-star at every counted step, conditional on previous observations
and the selected action. This satisfies Assumptions 3 and 5 and the martingale
condition used by Lemma 5. Ending an episode on a noisy terminal landing does not
discard that sample. R weights episodes by their actual transition counts, rather
than averaging per-episode mismatch rates. A custom sampler must independently
satisfy the paper's assumptions; accepting a callable does not verify them.

### ACP calibration: `acp_calibration`

The four panels show first-seed continuous episode scores, true support bounds,
and pre-episode radii,
cumulative episode coverage, trailing-window coverage, and the mean number of
states in localized neighborhoods compared with the 101-state global support.
The support-size diagnostic averages over nominal targets of nonterminal
state-action pairs using the fixed trial map.
`plot_comparison(window=10)` controls the trailing
coverage window independently of the calibration window used during the run.

Coverage compares ACP with a rolling quantile at the fixed target failure
parameter and with a frozen initial pilot quantile. These two alternatives are
replayed on the same saved score stream. They isolate calibration behavior;
they do not simulate counterfactual policies or establish navigation benefits
for separate frozen-calibration controllers.

ACP uses rank `ceil((1-delta_k)*(N+1))` without interpolation. The score is the
maximum disturbance norm over nonterminal transitions: indices
`0 <= t < episode_length - 1` when the episode terminates, or
`0 <= t < episode_length` at a nonterminal time limit. An episode with no nonterminal
transition has no score and skips ACP. Episodes with a score update:

```text
miss_k = 1{score_k > predicted_C_k}
delta_next = delta_k + eta*(delta - miss_k)
```

The adaptive failure parameter `delta_k` is not clipped. `ACP.quantile()` returns a
nonnegative `C_hat` directly: nonpositive ranks return zero, and ranks above N return
positive infinity. These tail-rank conventions are implementation choices not specified
in Algorithm 2 of the draft. A zero score is covered by a zero radius. Fixed C is
validated and converted when selected; there is no separate planning-radius function.
The plot marks unbounded predictions. With adaptive C, `C_used` equals `C_hat`,
so conformal coverage and planning-radius coverage are identical and recorded once.
Infinite radii are retained in saved data.
A calibrated radius need not equal that episode's true support bound.

For adaptive runs, the exact bookkeeping identity after N_scored ACP updates is:

```text
coverage(N_scored) = 1 - delta + (delta_next - delta_1)/(eta*N_scored)
```

Episodes without a score are excluded from coverage denominators. The coverage
target concerns the episode score computed from nonterminal disturbances.
It is not a goal-success guarantee, a collision bound, or a guarantee for every
short window. Assess coverage together with prediction efficiency and extended
prediction tails. Under a stationary law, a frozen quantile may already work well;
improved ACP performance must be demonstrated by the data.

### Online policy comparison: `policy_comparison`

The comparison uses the deployment episodes already recorded by Algorithm 2 and
the nominal and two fixed-R contamination baselines. The first panel shows cumulative
mean observed discounted reward, `G_k = sum_t gamma^t r(s_t,a_t)`, versus the online episode index.
Thin curves show individual trials; solid curves show their mean.
The second panel compares the fractions of those same episodes ending at the goal,
an obstacle, outside the workspace, or at the step limit. Bars show trial means
and dots show individual trials. These four outcomes are mutually exclusive.

Each episode supplies one observation of its current deployed policy. The figure
describes performance during online adaptation. It does not estimate Theorem 3's
regret, which compares robust optimal values at estimated and true R with the same C.
The comparison reads saved histories without running additional robot episodes.

For each trial and method, the plotter also overlays all `n_deployment_episodes`
trajectories on the saved obstacle map, marking the start and goal. Colors indicate
the deployment episode number. Each trajectory connects its recorded continuous
positions through its final landing; workspace exits remain visible outside the grid.
These figures are saved as `trajectories_{method}_R{R}_C{C}_trial_{trial}.png` and `.pdf`.

The R and ACP figures use runs with their corresponding component enabled.
Single-run `main(save_plots=True)` saves `final_policy_{test}.png` for every method
and `online_uncertainty_{test}.png` when either component is adaptive, where `{test}` is the method,
R/C setting, and trial.

## Distribution shifts with a fixed true R

Assumption 5 allows the unknown distribution D-star-k and its bounded support
C-star-k to change between episodes, while retaining a common conditional
mismatch mean R-star. The distribution stays fixed within each episode.
The online experiment implements this shift automatically. With zero-based online
episode index `k` and period `P=shift_period`, its true support bound is:

```text
C_star_k = 0.5 + 1.5*sin(pi*((k % P) + 0.5)/P)**2
```

With `P=100`, it moves smoothly between approximately 0.50037 and 1.99963,
with consecutive changes below 0.048. The half-episode phase keeps the radius
strictly above 0.5. At exactly 0.5, the current quantizer cannot provide a positive
common mismatch probability for every nominal target, including the corner goal.

At each transition, draw Delta=0 with probability `1-R_star`. Otherwise choose one
of the four signed coordinate directions uniformly and draw its length uniformly
between `(0.5+C_star_k)/2` and `C_star_k`. This length always exceeds half a unit cell,
so every nonzero draw is a mismatch. The law ignores the state, action, and obstacle
map; thus its conditional mismatch mean is exactly `R_star` at every counted step.
Its support reaches `C_star_k`, and its parameters stay fixed within the episode.
This satisfies the common-R requirement of Assumptions 3 and 5 without changing
the uncertainty-set construction in Section IV-A.

Pilots use the same periodic schedule at indices `-pilot_episodes, ..., -1`;
online episodes start at index zero. Shared calibration scores skip those pilots.
The cumulative R estimate and ACP history persist across distribution changes.

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
- `results_nominal_R0.0_C0.0_trial_1.pkl`: nominal baseline.
- `results_R_contamination_0.1_R0.1_Cinf_trial_1.pkl`: fixed-R=0.1 contamination baseline.
- `results_R_contamination_0.3_R0.3_Cinf_trial_1.pkl`: fixed-R=0.3 contamination baseline.
- `results_proposed_R0.2_C0.7_trial_1.pkl`: fixed R/C experiment.

Filenames and saved `fixed_R`/`fixed_C` use the actual inputs passed to the controller.
`None` identifies an adaptive component. The comparison plotter uses each method's
default R/C from `METHOD_SETTINGS` to locate matching trial files. Other parameter
experiments retain their own filenames.

Each run record contains settings, the configured MDP, full online histories,
final Q/policy, and state-action visit counts. Adaptive R saves `final_R`; adaptive C
saves ACP settings, initial and pilot scores, and the final adaptive parameter.
Q has shape `(101, 4)` and policy has shape `(101,)` on the
configured online grid; the exit row is zero. The fixed obstacle map is saved once
in the MDP. Each history records `C_star_k`, `R_used`, `C_used`,
the nonterminal deviation score, training diagnostics, original deviation vectors, continuous positions,
extended-grid quantized positions, episode length, and terminal outcome.
Adaptive R adds `R_hat_next`, `transition_count`, and `mismatch_count`.
Adaptive C adds `C_hat`, `miscoverage`, `delta_k`, and `delta_next`.
`states` and `positions` contain the initial observation followed by each landing
or the terminal contact marker; they have length `episode_length + 1`.
The original sampled endpoints are recovered from `nominal_positions + deviation_vectors`.
The disturbance, action, reward, and mismatch arrays have length `episode_length`.
Trajectory plots connect these successive positions directly.

Single-run figures use the same method, R/C, and trial suffix as the results file.
Comparison figures are `r_estimation`, `acp_calibration`, and `policy_comparison`,
plus one `trajectories_{method}_R{R}_C{C}_trial_{trial}` figure per loaded run.
Each is saved in PNG format in the same folder as the result files.

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
Runs saved before segment collision checks or terminal-score exclusion also require
rerunning; editing their scores alone cannot reproduce the resulting ACP and policies.
Runs from the former changing-map experiment must also be regenerated for this comparison.
Saved files using the previous attribute and result-key names must be regenerated.
Baseline files using the former `RNone_CNone` suffix must be regenerated under their
actual fixed R/C filenames for the updated comparison.

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
not a sampled deployment return. `utils_tabular.py` computes the mean and standard
deviation of saved learning curves for the fixed-R/C comparison plotter.

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
They also check randomized map protection and reachability, slow support shifts,
the common mismatch probability across nominal targets, including boundaries,
and a fixed trial map with paired support schedules across methods, pilots, and continuation.
