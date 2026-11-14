# Conformally robust reinforcement learning

This repository implements Algorithms 1 and 2 of the paper. Algorithm 1 trains
a robust Q-function using deterministic nominal transitions. Algorithm 2 deploys
the policy, estimates the discrete mismatch probability R, and calibrates the
continuous deviation radius C using adaptive conformal prediction (ACP).

## Run

From the repository root, create and activate a virtual environment, then install
all simulation, plotting, and transformer dependencies from the single requirements
file:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirement.txt
python -m robot_2d.main_robot_2d_online
python -m robot_2d.plot_robot_2d_online_comparison
```

On Windows, create the environment with `py -m venv .venv` and activate it with
`.\.venv\Scripts\Activate.ps1` in PowerShell before running the `python` commands.
Select `.venv/bin/python` (Windows: `.venv\Scripts\python.exe`) as the IDE interpreter.
The scripts also support direct execution. Configure experiments by editing the
settings under `if __name__ == "__main__"`; there is no command-line parser.

The online script runs five trials with seeds `(23, 67, 87, 101, 922)`, 100
deployment episodes per method, a 20,000-episode nominal training safety cap per
policy computation, and 100 random-policy pilot episodes for adaptive C. It uses discount
0.9, a 100-transition episode limit, initial R estimate 0.1, ACP target and initial
delta 0.1, learning rate eta 0.1, and a calibration window of 100 scores.

The map is fixed across all episodes, methods, and trials: 28 hard-coded obstacle
cells on a 10-by-10 grid, start center (1.5, 1.5), and goal center (7.5, 6.5).
The reward is 1 when a nonterminal state's nominal command targets the goal,
and 0 otherwise. Methods share the same episode-by-episode disturbance-support
schedule and use separate random draws.

The 3-D online experiment instead uses one map from `generate_grid_maps.training_maps`,
with seed 23 and obstacle-count range 250--400. It has 258 obstacles, start center
(1.5, 1.5, 1.5), and goal center (8.5, 8.5, 6.5). Its disturbance mixes bounded
quiet tracking errors with anisotropic Gaussian wind-gust displacements. The
gust component is conditioned to leave the nominal voxel, keeping R_star=0.3
as its support cycles between 0.7 and 1.3. The 2-D law and R_star=0.4 are unchanged.

Both online scripts use the saved Q-transformer and compiled nominal training.
Saved runs and figures stay in each example's `saved_results/online/` directory.

| Method | R used for training | C used for training |
| --- | --- | --- |
| `proposed` | Cumulative mismatch estimate | Pre-episode ACP radius |
| `nominal` | 0 | 0 |
| `R_contamination_0.1` | Fixed at 0.1 | Positive infinity |
| `R_contamination_0.5` | Fixed at 0.5 | Positive infinity |

The R-contamination baselines use prescribed R values as in Wang and Zou.
When both R and C are fixed, the controller trains once and reuses the policy.
Otherwise it recomputes the policy each deployment episode.
General statistics and plots (`plot_R`, `plot_acp`, `plot_performance`, and
`plot_conformal_history`) are shared in `utils.py`. Example-specific
geometry, disturbances, and trajectory plots live in `robot_2d/utils_2d.py` and
`robot_3d/utils_3d.py`. Each example owns its support schedule through
`C_star_shift_2d` or `C_star_shift_3d`; their current numerical schedules are unchanged.
Both robot MDPs inherit common nominal transitions, rewards, reset, collision traversal,
and deployment bookkeeping from `grid_robot_mdp.py`. Geometry and noise remain in each example.

Every `Robot2DMotionMDP` construction must explicitly specify `x_max`, `y_max`,
`grid_shape`, `start_cell`, `goal_cell`, `gamma`, `movement_reward`, and
`goal_reward`. These are required keyword arguments with no defaults. For example:

```python
mdp = Robot2DMotionMDP(
    x_max=10.0, y_max=10.0, grid_shape=(10, 10), start_cell=(0, 0), goal_cell=(9, 9),
    gamma=0.9, movement_reward=0.0, goal_reward=1.0,
)
```

The experiment scripts explicitly use a 10-by-10 workspace and grid, discount 0.9,
free-motion reward 0, and goal-command reward 1, with their own maps and endpoints.
Dataset generation randomizes maps and endpoints while preserving the template
MDP's rewards and discount. Clones explicitly copy all eight required settings.
The agent, controller, value-iteration labels, and deployment returns read the
discount from the MDP; these consumers have no separate `gamma` argument.

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
Construct an agent with `Tabular_Agent(mdp, R=R, C=C)`. R and C default to 0.0
and may also be passed positionally. These shared problem parameters are fixed
for that agent; each solver call supplies its own settings.
Q-learning receives lr_init, step_start_decay_lr, epsilon_init, epsilon_lb,
epsilon_decay_rate, and seed as method arguments. Its RNG, current epsilon,
learning rate, and working counters are local to that run. Value iteration needs
none of these Q-learning settings. Future solvers can follow the same pattern
without adding algorithm-specific constructor parameters.
`Tabular_Agent.initialize_q(initial_Q=None)` fills Q with zero by default, or copies
a supplied finite Q-table. Pass the table through
`agent.robust_q_learning(..., lr_init=learning_rate, initial_Q=predicted_Q)`.
Both solvers call `initialize_q(initial_Q)` directly, including when it is None.
Initialization clears previous diagnostics, curves, and visit counts.
Terminal rows are zeroed during initialization and never receive TD updates.
Initial cached values and diagnostics
are derived from the initialized table. The default case does not import transformer
libraries or require a checkpoint.

Both solvers stop when the full robust Bellman residual
`max(abs(T_R_C(Q) - Q))` meets `tolerance`, with terminal rows fixed at zero.
Q-learning checks after initialization, every `check_every` episodes, and at its
`max_training_episodes` safety cap. Its defaults are `tolerance=1e-6` and `check_every=100`.
The controller and robot entry points expose these as `training_tolerance` and
`training_check_every`; `max_training_episodes` is the maximum episode budget.
An accurate initial table can converge with zero training transitions. Small TD
updates alone do not establish convergence, because unvisited pairs and decaying
learning rates can hide large Bellman errors.

Value iteration retains `tolerance=1e-8` and `max_iterations=10000` by default.
It also accepts `initial_Q=predicted_Q`; omitting it preserves zero initialization.
Each solver records its algorithm, actual work (`global_steps`), residual,
tolerance, and `converged` status in `agent.diagnostics`. Value iteration also
records iterations; Q-learning records episodes and visit summaries and retains
its learning curve and visit-count table. These results always describe the
latest run, including a run that fails to converge. Reaching a safety cap raises an error;
the controller does not deploy that policy. Fixed-R,C learning-curve plots keep
each converged trial at its final value when averaging runs of different lengths.

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
first 100,000 training transitions by default, followed by per-pair harmonic decay, and an
epsilon schedule starting at 1.0 with decay 0.995 and floor 0.1.
Both online entry points extend the initial constant-learning-rate phase to
10,000,000 transitions so warm-start correction can meet the residual tolerance.
They set `accelerated_training=True`: `numba_grid_q_learning.py` compiles the nominal
rollout loop with Numba, retaining the same RNG, updates, episode boundaries,
and residual checks. Deployment continues through the original environment.
The ordinary solver remains the default for other callers; enable its compiled
path with `agent.robust_q_learning(..., accelerated=True)`.

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

The online plotting scripts read this fixed directory directly. Run
`python -m robot_2d.plot_robot_2d_online_comparison` or
`python -m robot_3d.plot_robot_3d_online_comparison` for the respective example.
It writes:

- `r_estimation.png`: cumulative R estimates and the true R_star.
- `acp_calibration.png`: first-trial scores/radii/support and delta history.
- `policy_comparison.png`: cumulative mean discounted returns and outcome rates.
- `trajectories_{method}_R{R}_C{C}_trial_{trial}.png`: all deployment trajectories.

Each figure is also saved as a PDF in that same directory.

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
python -m robot_2d.plot_robot_2d_fxRC_eval_returns
python -m robot_2d.plot_robot_2d_fxRC_value_policy
```

Edit the fixed-R/C calls at the bottom of `main_robot_2d.py`. Outputs go to
`robot_2d/saved_results/fixedRC/`. Each setting saves the initial-state value
curves in `robot_2d_R{R}_C{C}.pkl` and the MDP/trained agents in
`robot_2d_R{R}_C{C}_agents.pkl`. The curves show learned V(start) against
training transitions; they are not sampled deployment returns.

## Optional transformer warm start

Executable scripts keep their settings and workflow under `if __name__ == "__main__":`.
Importing a script does not run experiments. Transformer code is shared in
`q_transformer/`; it has no imports from the robot examples.

- `data.py` generates robust value-iteration Q labels, stores numeric MDP parameters,
  formats prompts, and provides seeded map-level splitting. `mdp_data(mdp)` exports
  parameters; `mdp_parameters(data, map_index)` supplies constructor arguments.
- `model.py` predicts Q-values directly from each cell's contextual BERT embedding.
  A non-affine LayerNorm and shared MLP map 768 to 128 to 128 to the action count,
  with GELU after the first two linear layers. There are no auxiliary feature
  heads, numerical feature inputs, or separate CLS/R/C readout branches. Terminal
  outputs remain exactly zero.
- `train.py` trains the encoder and Q head jointly from the original pretrained
  BERT weights and a new head. Existing Q-transformer checkpoints are never
  loaded. The 3-D encoder extends BERT's original 512 position embeddings to 2048,
  randomly initializes the extra positions, and trains all positions.

The prompt states R and C once as global parameters applying to every cell and
action. It notes that, for a fixed MDP, increasing either uncertainty parameter
with the other held fixed cannot increase any optimal Q(s,a). This is textual
guidance; the network has no additional monotonicity loss or hard constraint.
The prompt includes the full binary grid, column indices, explicit row/layer
indices, start/goal coordinates, cell widths, nominal motion, rewards, discount,
and the local-neighborhood rule. A fast tokenizer aligns each binary grid entry
with exactly one BERT token in Q-table row order, skipping coordinate labels.
The cell-to-token correspondence remains deterministic. Both examples reject
overlong prompts; neither truncates grid cells. Token capacities are 512 for 2-D
and 2048 for 3-D. Complete 10x10 and 10x10x10 prompts typically use approximately
400 and 1712 tokens, respectively, depending on numerical formatting.

Generate a dataset once, then train from its saved file:

```powershell
python -m robot_2d.generate_robot_2d_q_data
python -m robot_2d.train_robot_2d_transformer
```

For 3-D:

```powershell
python -m robot_3d.generate_robot_3d_q_data
python -m robot_3d.train_robot_3d_transformer
```

Datasets use NumPy arrays without pickle or saved prose descriptions:

| Scope | Fields |
|---|---|
| Shared by all maps | `bounds`, `grid_shape`, `gamma`, `movement_reward`, `goal_reward` |
| One entry per map | `start_states`, `goal_states`, `obstacle_masks` |
| One entry per Q-table | `map_index`, `R`, `C`, `Q`, `bellman_residual` |

Obstacle masks have shape `(maps, ny, nx)` or `(maps, nz, ny, nx)`. Start and goal
coordinates use `(x,y)` or `(x,y,z)` cell indices. Training constructs the example's
MDP class directly from these arrays, then calls the same `format_prompt(mdp,R,C)`
used at inference. Geometry, rewards, and discount must be consistent within a
dataset. Dataset generation does not import PyTorch or Transformers. Generation
uses robust value iteration on nominal deterministic transitions, with zero
terminal values and a Bellman-residual stopping tolerance.

Each generator uses 2,000 maps and all 20 R/C combinations per map, producing
40,000 target Q-tables. Both use `R_values=(0,0.2,0.4,0.5)`. The 2-D radii are
`(0,0.5,0.75,1.5,2)`; the 3-D radii are `(0,0.5,0.75,1,2)`.
The structured map generator fixes the start at (1,1) or (1,1,1), samples an
interior goal in the opposite half, protects endpoint obstacle clearance, and
provides both a wide route and a narrower route. The wide route admits a two-cell
cross-section in 2-D and a 2-by-2 cross-section in 3-D. Every free cell connects to
the start and goal; at least two obstacles intersect their straight segment.
Additional routes are allowed. There is no extra open region tied to C=3.
The seed controls map sampling. The generator scripts explicitly configure
geometry, obstacle counts, rewards, and discount.

Generation solves independent maps in CPU worker processes. Configure
`n_workers` in the generator script: one is serial; larger values use that many
processes. At most twice that many maps are pending. The parent generates maps,
updates progress, and writes the dataset in its original order.

Each training item is one map/R/C prompt and its target Q-table.
All cases belonging to a map stay together in the 90/10 map split; conditions of
a validation map cannot appear in training. One epoch visits all training
map/R/C cases. Batch size counts conditional prompts: 16 per GPU in 2-D and four
per GPU in 3-D, before gradient accumulation. The text is tokenized in bounded
per-map groups, and terminal masks are stored once per map. Each parameter pair
requires its own BERT encoding.

The shared defaults are:

| Setting | Value |
|---|---|
| Optimizer | AdamW; weight decay 0.01, excluding biases and normalization parameters |
| BERT peak learning rate | 1e-5 |
| Q head's peak learning rate | 3e-4 |
| 3-D position-embedding peak learning rate | 2e-5 |
| Schedule | 5% linear warmup, then cosine decay to zero |
| Epoch limit | 100 |
| Gradient accumulation / clipping | 1 update per batch / norm 1.0 |
| Split | 90% training maps, 10% validation maps; seed 42 |
| Validation / saving | Every epoch |
| Best-model metric | RMSE over nonterminal state-action entries |
| Early stopping | Patience 10; minimum improvement 1e-5 |
| Numbered checkpoints retained | Up to five |

The printed loss is Smooth L1 (Huber with beta=1), averaged over nonterminal
state-action entries. There is no auxiliary loss or feature-target generation.
Validation reports nonterminal Q RMSE, whole-table RMSE, nonterminal MAE, mean
per-table maximum error, and the largest Q error.
Q errors use terminal rows fixed at zero. The best Q-RMSE validation model is
restored before saving the final predictor.

CUDA uses BF16 when supported, otherwise FP16, with FP32 CPU fallback. On the
server, use distributed training so each GPU gets its own process:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc-per-node=2 \
  --module robot_3d.train_robot_3d_transformer
```

Replace `robot_3d` with `robot_2d` for the 2-D example. With the previously tested
Transformers/Accelerate versions, ordinary Python on multiple GPUs may use
DataParallel without activating mixed precision; torchrun activates distributed
mixed precision. A single GPU can run the ordinary Python module command.
Training entry points clean up the distributed process group when finished.

Data and new models are stored under each example's `saved_results/q_transformer/`:

```text
q_data_2d.npz or q_data_3d.npz  # Existing maps, R/C, Q labels, and Bellman residuals
checkpoint/                   # Best model.pt, encoder configuration, and tokenizer
checkpoint/checkpoint-*/       # Epoch weights, optimizer/scheduler, and Trainer state
```

Existing weights are not loaded during training. Every invocation starts fresh;
it does not resume an old optimizer, learning-rate schedule, or Q-transformer.
Previous numerical-input and auxiliary-feature checkpoints are incompatible with
the Q-only `text_cell_q` architecture and must be retrained. Existing checkpoint
files are not changed until a training run writes its outputs.
Settings are in the training scripts' main blocks. Native Trainer overrides are
available through the shared trainer's `training_options` argument.

The fixed-R/C and online scripts in both examples use their
`saved_results/q_transformer/checkpoint` directory by default. Set
`checkpoint_path = None` for zero initialization. The predictor loads `model.pt`
directly from the selected directory, or from its sole numbered `checkpoint-*`
subdirectory. If several numbered checkpoints exist without a root `model.pt`,
select the intended numbered directory explicitly. No weights are copied.
The fixed-R/C scripts predict once per map/R/C setting and reuse that initial
table across trials. Use the evaluation scripts for paired zero/warm-start
comparisons on the same map; the fixed-R/C scripts save the selected
initialization's results once per setting. Programmatic use is:

```python
from q_transformer.model import QTransformerPredictor

predictor = QTransformerPredictor("robot_2d/saved_results/q_transformer/checkpoint")
controller = ConformallyRobustController(mdp, q_initializer=predictor)
```

The predictor uses the same conditional prompt, token alignment, and terminal
mask as training. It returns an (n_states, n_actions) NumPy Q-table directly:
(100, 4) for the 2-D grid and (1000, 6) for the 3-D grid.
The controller supplies R_used and C_used before Algorithm 1. Omitting
`q_initializer` preserves zero initialization without importing transformer
dependencies. Match geometry, action ordering, rewards, and discount between
the dataset and the deployed MDP.

Run `python -m robot_2d.evaluate_robot_2d_transformer` or
`python -m robot_3d.evaluate_robot_3d_transformer` to evaluate the new checkpoint.
Both use the small comparison helper in `q_transformer/evaluate.py`.
Set R, C, algorithm (`"robust_value_iteration"` or `"robust_q_learning"`), and tolerance in the
script's main block. Each run generates one fresh map using the training map
family. All Q-learning settings are explicitly supplied in the script:
lr_init, step_start_decay_lr, epsilon_init, epsilon_lb, epsilon_decay_rate, and
seed are passed to robust_q_learning; the sampled MDP, R, and C construct the
agent. Both initializations use the same settings. It then
compares zero initialization against the transformer prediction on
that same map at the same Bellman-residual tolerance. Set seed to an integer to
repeat the map; the two Q-learning runs always share a learning RNG seed.
Only global time-step counts and final residuals are printed. For Q-learning,
steps are actual training transitions; for value iteration they are individual
nonterminal state-action updates (sweeps times nonterminal states times actions).
Residual checks and inference do not enter these counters. Q-learning uses the
same check_every setting for both runs. No dataset is loaded, no figures or
evaluation files are saved, and no checkpoint is modified.

The 3-D Q-learning comparison uses R=0.2, C=0.75, lr_init=0.9,
epsilon_init=0.1, epsilon_lb=0.01, epsilon_decay_rate=0.999, and
step_start_decay_lr=1000000, with tolerance=1e-3 and check_every=100.
With the previous numerical-input checkpoint, warm starts reduced training
transitions by 30% to 58% on map seeds 23, 67, and 91. Those historical results
must be re-evaluated after training the text-only model. Both initializations
reached the full
Bellman-residual threshold before learning-rate decay began. Exploring starts
remain enabled to cover all nonterminal state-action pairs. The improvement
depends on the sampled map and prediction quality.

## Validation

```powershell
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s robot_2d/tests -v
python -B -m unittest discover -s robot_3d/tests -v
```

Tests cover global discretization, distinct outside cells, cell-distance
neighborhoods, terminal values and initialization, nominal/reference learning,
terminal-inclusive calibration, mismatch accounting, controller continuation,
saved results, and plotting. Shared transformer checks cover 3-D state/action
shapes, sample splits, epoch/final checkpoint reload, and actual warm-start Q-learning
using a tiny local encoder. Model checks skip when optional dependencies are absent.
