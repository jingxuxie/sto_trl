# Fast Prototyping Plan: Stochastic Transitive RL

**Goal:** Quickly test whether a stochastic extension of Transitive RL (TRL) is promising, while avoiding misleading toy experiments that fail for the wrong reason.

**Prepared:** 2026-06-13

---

## 0. One-sentence project framing

Test whether **calibrated stochastic successor distances plus TRL-style divide-and-conquer path relaxation** improve long-horizon offline goal-conditioned RL under stochastic dynamics, especially in environments with risky shortcuts or stochastic teleporters.

The core research hypothesis is not merely “triangle inequality in stochastic RL.” Myers et al. already show that contrastive successor features can yield a temporal distance satisfying the triangle inequality even in stochastic settings. The interesting hypothesis is:

> A calibrated stochastic distance estimator can be made more horizon-efficient by adding a TRL-style recursive transitive backup, without inheriting the optimistic bias of deterministic TRL in stochastic environments.

---

## 1. What should count as early evidence?

A fast prototype is worth continuing only if it produces at least one of these signals:

1. **Bias signal:** original deterministic-style TRL overestimates risky stochastic paths, while a stochastic-calibrated variant is better calibrated.
2. **Horizon signal:** adding a transitive/log-space divide-and-conquer loss improves long-range goal prediction versus calibration-only baselines.
3. **Policy signal:** the resulting greedy policy avoids bad stochastic shortcuts and reaches goals more often or with lower regret.
4. **Scaling signal:** on small OGBench teleport tasks, the stochastic-TRL variant improves early-learning or final success over raw TRL and at least one stochastic-distance baseline.

A result is **not** persuasive if it only wins on a deterministic maze, only improves loss but not policy behavior, only works with oracle labels unavailable offline, or only wins because it has more parameters or more tuning.

---

## 2. Grounding in the current code and benchmarks

Use the existing TRL and OGBench implementations where possible:

- The official TRL repository uses JAX, depends on OGBench, and places the main implementation in `agents/trl.py`.
- TRL’s critic update currently computes two midpoint values, multiplies them as `target = first_q * second_q`, then applies expectile-weighted BCE. This is the smallest code location to fork for log-space and stochastic variants.
- OGBench provides 8 environment families, 85 datasets, state/pixel observations, and reference implementations for GCBC, GCIVL, GCIQL, QRL, CRL, and HIQL.
- OGBench teleport mazes are especially relevant because the teleporters are explicitly designed to test stochasticity: a black hole sends the agent to a random white hole, one of which is a dead end, so the agent must avoid optimistic bias from lucky outcomes.
- TMD is the most important related baseline because it is designed to learn optimal goal-reaching distances in stochastic dynamics using quasimetric structure plus transition/action invariance constraints.

Recommended repo layout:

```text
stoch-trl-proto/
  tabular/
    envs.py                 # tiny MDPs with exact transitions
    dp.py                   # exact value iteration and successor-distance computation
    datasets.py             # offline trajectory generation
    models.py               # tabular table, MLP, bilinear embeddings
    losses.py               # raw TRL, log-TRL, contrastive, TMD-like losses
    train.py
    eval.py
    plots.py
  ogbench_experiments/
    agents/trl_log.py       # fork of agents/trl.py
    agents/trl_stoch.py     # calibration + transitive variant
    configs_fast.sh
    eval_summary.py
  results/
    tabular_runs.csv
    ogbench_runs.csv
    figures/
```

---

## 3. Algorithm variants to prototype

Start with variants that answer distinct questions. Do not introduce too many changes at once.

### Variant 0: `TRL-raw`

This is the original product-backup idea:

\[
\hat U(s_i,a_i,g_j) \leftarrow \hat U(s_i,a_i,w_k)\,\hat U(w_k,a_k,g_j).
\]

Purpose: establish the failure mode under stochasticity.

Expected behavior: good on deterministic shortest paths; potentially overoptimistic on lucky stochastic paths.

---

### Variant 1: `TRL-log`

Represent log-reachability instead of probability:

\[
z_\theta(s,a,g) = \log U_\theta(s,a,g) \le 0.
\]

Then the product backup becomes an additive target:

\[
z_\theta(s_i,a_i,g_j) \leftarrow z_{\bar\theta}(s_i,a_i,w_k) + z_{\bar\theta}(w_k,a_k,g_j).
\]

Equivalently, with distance \(D=-z\):

\[
D_\theta(s_i,a_i,g_j) \leftarrow D_{\bar\theta}(s_i,a_i,w_k)+D_{\bar\theta}(w_k,a_k,g_j).
\]

Purpose: separate “numerical stability” from true stochastic calibration.

Implementation notes:

- Output `z = -softplus(raw)` so that `U = exp(z) <= 1`.
- Use Huber loss in `z` or `D`, not BCE on probabilities.
- Clamp `z` to something like `[-80, 0]` for safety.
- Keep the same midpoint sampler as TRL initially.

---

### Variant 2: `MC-cal + TRL-log`

Add an on-trajectory calibration term:

\[
\mathcal L_{\text{MC}} = \ell\left(z_\theta(s_i,a_i,s_j), (j-i)\log \gamma\right).
\]

Then add the transitive loss:

\[
\mathcal L_{\text{tr}} = \ell\left(z_\theta(s_i,a_i,s_j), z_{\bar\theta}(s_i,a_i,s_k)+z_{\bar\theta}(s_k,a_k,s_j)\right).
\]

Total:

\[
\mathcal L = \mathcal L_{\text{MC}} + \lambda_{\text{tr}}\mathcal L_{\text{tr}}.
\]

Purpose: test whether transitive recursion improves horizon generalization while an MC-like term anchors the scale.

Caveat: this is not a fully correct stochastic distance. It still labels lucky realized trajectories as if they were reliable. That is acceptable only as an intermediate diagnostic.

---

### Variant 3: `Successor-distance + TRL-log`

This is the first genuinely stochastic variant.

Learn a successor-style score \(M(s,g)\), then define a normalized distance roughly of the form:

\[
D(s,g) = \log M(g,g) - \log M(s,g).
\]

For state-action values:

\[
D(s,a,g) = \log M(g,g) - \log M(s,a,g).
\]

Add the transitive additive loss on this distance:

\[
\mathcal L_{\text{tr}} =
\ell\left(
D_\theta(s_i,a_i,g_j),
D_{\bar\theta}(s_i,a_i,w_k)+D_{\bar\theta}(w_k,a_k,g_j)
\right).
\]

Purpose: test whether stochastic successor-distance calibration plus TRL-style divide-and-conquer improves long-horizon generalization.

Practical minimal version:

- Use contrastive future-state prediction to learn successor scores.
- Use in-batch negatives.
- Compute normalized distances using a self-score correction.
- Add a small transitive loss weight first: `lambda_tr in {0.01, 0.03, 0.1, 0.3}`.

---

### Variant 4: `TMD + TRL-relax`

Start from TMD and add a TRL-like sampled path-relaxation term.

Purpose: strongest research baseline. If this variant helps, the project becomes:

> TRL-style recursive path relaxation improves horizon efficiency of stochastic quasimetric distance learning.

This is probably the most publishable direction if the signal is positive, but it is not the fastest place to start.

---

## 4. Experiment ladder

The ladder is designed to catch bugs and false positives early.

| Stage | Environment | Time cost | Main question | Advance only if |
|---|---:|---:|---|---|
| A | deterministic tabular | minutes | Does the implementation reproduce TRL-like shortest-path behavior? | raw/log variants recover shortest paths |
| B | stochastic tabular with exact DP | minutes | Does raw TRL overestimate stochastic lucky paths? | failure mode appears and metrics detect it |
| C | learned tabular, offline data | minutes to <1 hour | Does stochastic calibration + transitive loss help generalization? | held-out long-horizon value/policy improves |
| D | custom tiny continuous point maze | <1 hour per sweep | Does behavior survive function approximation and continuous states? | same qualitative signal as tabular |
| E | OGBench PointMaze teleport | hours | Does signal transfer to a real benchmark scaffold? | improvement over raw TRL or calibration-only |
| F | OGBench AntMaze teleport | hours to days | Is the direction worth serious compute? | survives at least 3 seeds and fair tuning |

Do not start with AntMaze. It is too expensive and too confounded for early iteration.

---

## 5. Stage A: deterministic tabular sanity check

### Purpose

Make sure the implementation can solve the deterministic case before adding stochasticity.

### Environment A1: deterministic chain

States: `0, 1, ..., N-1`.

Actions: `left`, `right`.

Transition:

```text
right: s -> min(s+1, N-1)
left:  s -> max(s-1, 0)
```

Goals: all states.

Use `N in {16, 32, 64}`.

### Environment A2: deterministic two-room maze

A tiny grid with a bottleneck. This tests whether the method discovers waypoint composition.

### Data

Generate offline trajectories using a mixture:

```text
70% noisy shortest-path controller
20% random walk
10% reverse shortest-path controller
```

Make sure every state-action appears at least a minimum number of times. For tabular experiments, also log a coverage matrix.

### Baselines

- Exact shortest path from graph search.
- Supervised MC distance: regress \((j-i)\log\gamma\) or \(\gamma^{j-i}\).
- `TRL-raw`.
- `TRL-log`.

### Metrics

- Distance MSE versus true shortest path.
- Long-horizon holdout MSE: train on pairs with separation `<= 8`, evaluate on `> 8`.
- Triangle violation rate:

\[
\frac{1}{|\mathcal T|}\sum_{s,w,g}\mathbf{1}[D(s,g) > D(s,w)+D(w,g)+\epsilon].
\]

- Greedy policy success rate.
- Median steps to goal.

### Pass criteria

Advance only if:

- `TRL-raw` and `TRL-log` recover near-shortest paths.
- Long-horizon estimates improve from transitive updates versus pure MC regression.
- Triangle violation metric is low for learned distances.

### Failure interpretation

If this stage fails, do not proceed. The bug is likely in indexing, midpoint sampling, discount convention, target networks, or policy extraction.

---

## 6. Stage B: stochastic tabular tests with exact ground truth

This is the most important stage. It tells you whether the research idea targets a real failure mode.

### Shared ground truth

For each goal \(g\), compute optimal discounted reachability by value iteration:

\[
V^*(s,g) = \max_a U^*(s,a,g),
\]

\[
U^*(s,a,g) = \gamma \; \mathbb E_{s'\sim P(\cdot|s,a)}
\left[
\mathbf 1[s'=g] + \mathbf 1[s'\ne g] V^*(s',g)
\right].
\]

Convert to a distance-like quantity only for diagnostics:

\[
D^*(s,a,g) = -\frac{\log(\max(U^*(s,a,g),\epsilon))}{-\log \gamma}.
\]

This `D*` is useful, but do not assume it automatically satisfies every stochastic triangle inequality. It is a ground-truth control objective.

---

### Environment B1: slip chain

Same as deterministic chain, but actions slip:

```text
with probability 1-p: intended move
with probability p/2: opposite move
with probability p/2: stay
```

Sweep:

```text
p_slip in {0.0, 0.05, 0.1, 0.2, 0.4}
N in {16, 32, 64}
```

Question: does each method stay calibrated as stochasticity increases?

Expected useful signal:

- Original TRL may remain okay for mild slip but become overconfident at high slip.
- Stochastic successor-distance variants should degrade smoothly.

---

### Environment B2: risky shortcut vs safe route

Construct an MDP with two routes from start to goal:

```text
safe route:   deterministic length L_safe
risky route:  length 2, succeeds with probability p_success, otherwise goes to trap/dead-end
```

Use:

```text
L_safe in {8, 16, 32}
p_success in {0.1, 0.2, 0.4, 0.6, 0.8}
trap_escape_length in {infinite, 32, 64}
gamma in {0.95, 0.98, 0.99}
```

This is the key diagnostic. It directly tests whether a method is fooled by lucky trajectories.

Design the offline data so that it contains:

- some lucky risky successes,
- some risky failures,
- enough safe-path trajectories,
- enough coverage near the start and goal.

Question: does the learned policy choose the route that is actually optimal under expected discounted reachability?

Metrics:

- Predicted \(U(start, risky, goal)\) versus exact \(U^*\).
- Predicted \(U(start, safe, goal)\) versus exact \(U^*\).
- Probability the greedy policy chooses risky action.
- Policy regret versus exact DP policy.
- Overestimation ratio:

\[
\frac{\hat U(start, risky, goal)}{U^*(start, risky, goal)}.
\]

Pass criteria:

- `TRL-raw` should show a detectable overoptimism region.
- Your stochastic variant should reduce overestimation without becoming overly pessimistic on the safe route.
- Transitive loss should help long-route value estimation, not merely suppress all values.

---

### Environment B3: stochastic teleporter maze

Small gridworld with:

- one black-hole teleporter,
- three white-hole destinations,
- one destination near the goal,
- one neutral destination,
- one dead-end destination.

This mirrors the qualitative structure of OGBench teleport mazes but is small enough for exact DP.

Sweep:

```text
p_good in {1/3, 1/2, 2/3}
p_dead in {1/3, 1/4, 0.1}
maze_size in {7x7, 11x11}
```

Metrics:

- Teleporter selection rate.
- Value calibration on states immediately before the teleporter.
- Triangle violation rate over sampled triples.
- Greedy policy success.
- Median steps conditional on success.

Pass criteria:

- The method should avoid the teleporter when DP says it is suboptimal.
- It should use the teleporter when DP says the risk is worth it.
- It should not simply learn “teleporters are bad.”

---

### Environment B4: stochastic stitching graph

Create two disconnected-looking trajectory modes in the dataset that can be stitched through a shared region:

```text
Mode 1 trajectories: A -> B -> C
Mode 2 trajectories: C -> D -> E
Evaluation: A -> E
```

Add stochastic transitions around `C` so the midpoint is not always reliable.

Question: does transitive learning still help stitching when the waypoint has stochastic outcomes?

Metrics:

- Held-out A-to-E value error.
- Success on A-to-E greedy policy.
- Predicted distance via best waypoint C.
- Overconfidence on bad stochastic C exits.

---

## 7. Stage C: learned tabular offline experiments

Stage B used exact ground truth for diagnosis. Stage C tests learning from offline trajectories.

### Model classes

Use three progressively harder function approximators:

1. **Table model:** one parameter per `(s,a,g)`.
2. **Embedding bilinear model:**

\[
score(s,a,g) = f_\theta(s,a)^\top h_\theta(g).
\]

3. **Tiny MLP:** one-hot or coordinate input, 2 layers of width 64.

The table model should work first. If the table works but the MLP fails, the issue is optimization/generalization, not the concept.

### Offline dataset sizes

For each environment, use:

```text
num_trajectories in {100, 1_000, 10_000}
max_episode_len = environment-specific horizon
behavior_noise in {0.05, 0.2, 0.5}
```

Hold out:

- long-horizon pairs,
- rare goals,
- starts near risky branch points,
- trajectories from one stochastic seed.

### Losses to compare

#### C0: MC supervised

\[
\mathcal L_{MC}=\ell(z_\theta(s_i,a_i,s_j), (j-i)\log\gamma).
\]

#### C1: raw TRL

\[
U_{target}=U_{\bar\theta}(s_i,a_i,s_k)U_{\bar\theta}(s_k,a_k,s_j).
\]

#### C2: log TRL

\[
z_{target}=z_{\bar\theta}(s_i,a_i,s_k)+z_{\bar\theta}(s_k,a_k,s_j).
\]

#### C3: MC + log TRL

\[
\mathcal L=\mathcal L_{MC}+\lambda_{tr}\mathcal L_{tr}.
\]

#### C4: contrastive successor distance

Train future-state contrastive scores, then use normalized successor distance.

#### C5: contrastive successor distance + log TRL

Same as C4, with additive transitive loss.

### Hyperparameter sweep

Keep this small:

```text
gamma:          {0.95, 0.98, 0.99}
lambda_tr:      {0.0, 0.01, 0.03, 0.1, 0.3}
expectile/tau:  {0.5, 0.7, 0.9}
midpoints:      {1, 4, 16 candidates per pair}
learning_rate:  {1e-3, 3e-4}
seeds:          {0, 1, 2}
```

Do not tune all methods equally deeply at first. Use a fixed small sweep to find gross signal. Once there is signal, rerun a fairer sweep.

### Core metrics

Use both value metrics and policy metrics.

Value metrics:

- MSE versus exact DP \(U^*\).
- MSE versus exact DP \(D^*\).
- Calibration error for discounted reachability.
- Overestimation error:

\[
\mathbb E[\max(0, \hat U-U^*)].
\]

- Underestimation error:

\[
\mathbb E[\max(0, U^*-\hat U)].
\]

- Triangle violation rate.
- Long-horizon holdout MSE.

Policy metrics:

- Success rate.
- Median steps to goal.
- Regret versus DP policy.
- Risky shortcut selection rate.
- Teleporter usage rate.
- Conditional success after choosing teleporter.

### Minimum result table

```text
method, env, p_stoch, dataset_size, seed,
value_mse, long_horizon_mse, calibration_ece,
overestimate_mean, triangle_violation_rate,
success_rate, median_steps, regret,
risky_action_rate, teleporter_rate
```

---

## 8. Stage D: tiny continuous control before OGBench

This catches the function-approximation and continuous-state issues without full OGBench cost.

### Environment D1: continuous point navigation with stochastic wind

State: `(x, y)`.

Action: clipped velocity `(dx, dy)`.

Transition:

\[
s_{t+1}=s_t+a_t+\epsilon_t,
\]

where \(\epsilon_t\sim\mathcal N(0,\sigma^2I)\), plus walls.

Goals: random points.

Reward/evaluation: success if within radius `r`.

Sweep:

```text
sigma in {0.0, 0.02, 0.05, 0.1}
maze: open, two-room, risky-teleporter
```

### Environment D2: continuous risky teleporter

Add a circular teleporter region. Entering it sends the point to one of several destinations, including a dead-end region.

This is a bridge between tabular B3 and OGBench teleport.

### Data

Generate offline data with a scripted noisy controller and random exploration.

Use small datasets first:

```text
10k, 50k, 100k transitions
```

### Metrics

Same as Stage C, except exact DP may require discretization. Use a fine grid approximation for approximate ground truth.

### Pass criteria

- The qualitative finding from tabular should persist.
- If it disappears, inspect whether the issue is representation, action coverage, or evaluation noise.

---

## 9. Stage E: OGBench PointMaze teleport experiments

Only start this once Stages B/C are clean.

### Why PointMaze first?

PointMaze is cheap and directly tests stochastic teleport behavior. OGBench’s teleport maze is explicitly designed to test stochasticity and optimism around lucky teleporter outcomes.

### Recommended datasets

Start with:

```text
pointmaze-teleport-navigate-v0
pointmaze-teleport-stitch-v0
```

Then move to:

```text
antmaze-teleport-navigate-v0
antmaze-teleport-stitch-v0
```

Avoid HumanoidMaze until the idea survives AntMaze.

### Fast-run settings

Use these only as *screening*, not as publishable results:

```text
training_steps:      100k, 250k, 500k
batch_size:          256 or 512
hidden_dims:         (256, 256) initially
seeds:               0, 1, 2
eval_episodes:       20 for screening, 50+ later
state observations:  yes
pixel observations:  no
```

### Baselines

Minimum screening baselines:

1. Original TRL.
2. CRL.
3. QRL.
4. TMD, once integration is available.
5. Your `TRL-log`.
6. Your best tabular stochastic variant.

Do not claim novelty over stochastic distance learning without TMD.

### Implementation fork

Start from the official TRL code:

```text
agents/trl.py        -> agents/trl_log.py
agents/trl.py        -> agents/trl_stoch.py
```

Change only the critic loss first.

#### `trl_log.py` changes

Replace probability target/BCE with log target/Huber:

```python
first_z = log_u(first_q_logits)
second_z = log_u(second_q_logits)
target_z = stop_gradient(first_z + second_z)
pred_z = log_u(q_logits)
q_loss = expectile_or_huber(pred_z, target_z, weights=dist_weight)
```

where:

```python
def log_u(raw):
    return -jax.nn.softplus(raw)  # <= 0
```

#### `trl_stoch.py` additions

Add one of:

1. contrastive successor calibration,
2. TMD-style temporal invariance term,
3. conservative penalty on high predicted reachability for stochastic branch states,
4. variance-aware target using repeated outcomes in tabular/custom envs.

For OGBench, prefer (1) or (2). The conservative penalty is useful as a diagnostic but less principled.

### Screening metrics

Beyond success rate, log:

- predicted value distribution near teleporter entrance,
- value assigned to goals behind dead-end teleport outcomes,
- action norms and BC loss,
- fraction of episodes entering teleporter,
- success conditional on entering teleporter,
- median episode length,
- seed variance.

### Pass criteria

A screening win is:

- `TRL-log` or stochastic-TRL improves over original TRL by at least 5 absolute success points on one PointMaze teleport task, or
- stochastic-TRL matches success but substantially reduces teleporter-overuse/overconfidence, or
- stochastic-TRL improves early learning at 100k/250k steps while matching final success.

A strong signal is:

- improvement appears on both `navigate` and `stitch`,
- at least 2 of 3 seeds improve,
- metrics show reduced stochastic optimism rather than arbitrary conservatism.

---

## 10. Stage F: OGBench AntMaze teleport

Run this only after PointMaze shows a signal.

### Datasets

```text
antmaze-teleport-navigate-v0
antmaze-teleport-stitch-v0
```

### Baselines

At minimum:

- TRL original,
- CRL,
- QRL,
- HIQL,
- TMD,
- your best stochastic-TRL variant.

### Run protocol

Use a fair budget:

```text
seeds:          3 initially, 5 if promising
training steps: match TRL/TMD conventions if possible
hyperparams:    tune alpha/lambda_tr/expectile comparably
```

### Pass criteria

Continue the project only if:

- stochastic-TRL is competitive with TMD/HIQL, or
- it is not final-SOTA but clearly improves horizon scaling, calibration, or early learning, or
- combining TRL-relaxation with TMD improves TMD.

If TMD dominates everything and the TRL loss does not improve it, the project may still be publishable only if the tabular theory reveals a new insight. Otherwise, pivot.

---

## 11. Ablation checklist

Run these ablations before believing the result.

| Ablation | Why it matters | Expected interpretation |
|---|---|---|
| `lambda_tr = 0` | isolates calibration-only baseline | if no difference, TRL relaxation is not helping |
| raw probability vs log-space | tests numerical stability | log should help long horizons if underflow mattered |
| expectile `{0.5,0.7,0.9}` | tests optimistic bias | high expectile may overvalue lucky paths |
| in-trajectory midpoints vs random midpoints | tests offline/OOD issue | random midpoints may overestimate or destabilize |
| number of midpoint candidates | tests divide-and-conquer strength | more candidates should help until overfitting/OOD |
| with/without self-normalization | tests stochastic distance validity | removing normalization should hurt calibration/triangle behavior |
| table vs MLP | separates concept from function approximation | table win + MLP fail means optimization issue |
| equal parameter count | prevents architecture confound | stronger method should not win only from size |
| equal tuning budget | prevents hyperparameter confound | especially important vs TMD/CRL/QRL |

---

## 12. Diagnostics that prevent flawed conclusions

### Coverage diagnostics

Log:

```text
state visitation histogram
action visitation histogram
goal visitation histogram
state-goal pair coverage
risky-success and risky-failure counts
teleporter outcome counts
```

A method cannot learn stochasticity if the dataset contains only lucky outcomes.

### Calibration diagnostics

For each bin of predicted \(\hat U\), estimate empirical discounted reachability from rollouts:

```text
bin: [0.0, 0.1), empirical E[gamma^tau], count
bin: [0.1, 0.2), empirical E[gamma^tau], count
...
```

Report expected calibration error:

\[
ECE = \sum_b \frac{n_b}{N}\left|\mathbb E[\hat U|b] - \mathbb E[\gamma^\tau|b]\right|.
\]

### Optimism diagnostics

Track separately:

```text
overestimation on safe states
overestimation on risky branch states
overestimation on teleporter entrance states
overestimation on long-horizon states
```

The expected failure mode is not uniform error; it is optimistic error at stochastic branch points.

### Policy diagnostics

Always evaluate:

```text
success rate
median steps
mean discounted return
risky branch choice rate
teleporter entry rate
success conditional on risky/teleporter choice
```

A method that avoids all risky branches may look calibrated but fail when risk is actually optimal.

### Robustness diagnostics

Run each final tabular experiment with:

```text
seeds: at least 5 for tabular
p_stoch sweep
small/medium/large datasets
behavior coverage sweep
```

The method should fail gracefully as data quality decreases.

---

## 13. Milestone plan

### Milestone 1: tabular harness and deterministic reproduction

**Deliverables**

- `envs.py` with deterministic chain and two-room grid.
- `dp.py` with graph shortest paths and discounted value iteration.
- `losses.py` with MC, raw TRL, and log TRL.
- `eval.py` with value, triangle, and policy metrics.

**Success criterion**

- Raw/log TRL recover deterministic shortest-path structure.
- Greedy policy succeeds on held-out goals.

**Stop condition**

- If raw/log TRL cannot solve deterministic chain, debug before proceeding.

---

### Milestone 2: risky stochastic MDP failure test

**Deliverables**

- Risky shortcut MDP.
- Exact DP policy/value.
- Dataset generator with lucky and unlucky risky outcomes.
- Results table over `p_success`, `L_safe`, and `gamma`.

**Success criterion**

- Original TRL shows measurable overoptimism in at least part of the sweep.
- Metrics identify the failure before policy rollouts.

**Stop condition**

- If raw TRL does not fail anywhere, the proposed project may lack a clear target. Try a more adversarial teleporter before giving up.

---

### Milestone 3: first stochastic-calibrated variant

**Deliverables**

- Contrastive successor-distance or self-normalized successor score.
- `Successor-distance + TRL-log` loss.
- Comparison against calibration-only.

**Success criterion**

- Lower overestimation on risky branch states.
- Similar or better long-horizon value MSE than calibration-only.
- Better policy regret than raw TRL.

**Stop condition**

- If calibration-only dominates and transitive loss consistently hurts, reduce `lambda_tr`, use lower expectile, or make transitive loss one-sided/conservative.

---

### Milestone 4: stochastic teleporter gridworld

**Deliverables**

- Small teleporter grid with exact DP.
- Teleporter usage diagnostics.
- Sweep over good/dead-end probabilities.

**Success criterion**

- Method uses teleporter when optimal and avoids it when suboptimal.
- Raw TRL is more sensitive to lucky samples.
- `stochastic + transitive` improves long-horizon estimates over `stochastic only`.

**Stop condition**

- If the method simply avoids all teleporters, it is conservative, not intelligent.

---

### Milestone 5: continuous tiny point maze

**Deliverables**

- Continuous point maze with wind and/or teleporter.
- Approximate grid DP for evaluation.
- MLP implementation.

**Success criterion**

- Same qualitative pattern as tabular under function approximation.
- No catastrophic instability from log-space targets.

**Stop condition**

- If table works but MLP fails, focus on architecture/normalization before OGBench.

---

### Milestone 6: OGBench PointMaze teleport screening

**Deliverables**

- `agents/trl_log.py` fork.
- `agents/trl_stoch.py` fork.
- Fast commands for `pointmaze-teleport-navigate-v0` and `pointmaze-teleport-stitch-v0`.
- Eval summary with teleporter usage.

**Success criterion**

- Positive signal on at least one task and no obvious calibration regression.

**Stop condition**

- If no signal appears, inspect whether the tabular success depends on oracle quantities unavailable in OGBench.

---

### Milestone 7: TMD comparison/integration

**Deliverables**

- Reproduce a small TMD run from `tmd-release`.
- Add TRL-relaxation term to TMD or compare directly.
- Run PointMaze teleport and, if promising, AntMaze teleport.

**Success criterion**

- TRL-relaxation improves TMD or provides a clear efficiency/calibration tradeoff.

**Stop condition**

- If TMD dominates with equal compute and the TRL term adds nothing, pivot to theory or drop the project.

---

## 14. Practical commands and scaffolding

### TRL setup

```bash
git clone https://github.com/aoberai/trl.git
cd trl
pip install -r requirements.txt
```

Fast baseline smoke test:

```bash
python main.py \
  --env_name=pointmaze-teleport-navigate-v0 \
  --agent=agents/trl.py \
  --eval_episodes=20
```

If running headless MuJoCo:

```bash
MUJOCO_GL=egl python main.py \
  --env_name=pointmaze-teleport-navigate-v0 \
  --agent=agents/trl.py \
  --eval_episodes=20
```

### OGBench direct API smoke test

```python
import ogbench

env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
    'pointmaze-teleport-navigate-v0',
    compact_dataset=False,
)

print(train_dataset.keys())
print(train_dataset['observations'].shape)
```

### Suggested tabular run commands

```bash
python -m tabular.train \
  --env risky_shortcut \
  --method trl_raw \
  --n_states 32 \
  --p_success 0.3 \
  --gamma 0.98 \
  --num_trajectories 1000 \
  --seed 0

python -m tabular.train \
  --env risky_shortcut \
  --method succdist_trl_log \
  --lambda_tr 0.1 \
  --expectile 0.7 \
  --p_success 0.3 \
  --gamma 0.98 \
  --num_trajectories 1000 \
  --seed 0
```

### Suggested result aggregation

```bash
python -m tabular.eval_summary \
  --input results/tabular_runs.csv \
  --groupby env,method,p_success,gamma,dataset_size \
  --metrics value_mse,long_horizon_mse,overestimate_mean,success_rate,regret,risky_action_rate
```

---

## 15. Concrete “week 1” checklist

### Day 1

- Implement deterministic chain, slip chain, risky shortcut.
- Implement exact DP.
- Implement table model and greedy policy extraction.
- Verify graph shortest path and DP agree when `p_stoch = 0`.

### Day 2

- Implement MC, raw TRL, log TRL losses.
- Run deterministic chain and two-room grid.
- Confirm long-horizon holdout improvement from transitive loss.

### Day 3

- Run risky shortcut sweep.
- Plot raw TRL overestimation vs `p_success`.
- Confirm metrics detect risky overoptimism.

### Day 4

- Implement successor-distance/contrastive calibration.
- Add transitive log-space loss.
- Compare calibration-only vs calibration+transitive.

### Day 5

- Implement teleporter gridworld.
- Run 3 seeds for top 3 variants.
- Decide whether to proceed to continuous/OGBench.

End-of-week decision:

```text
Continue if:
  stochastic+transitive improves long-horizon MSE or policy regret
  AND it reduces risky overestimation versus raw TRL
  AND it does not merely avoid all stochastic choices.

Pause/pivot if:
  calibration-only dominates everywhere,
  or transitive loss only adds optimism,
  or improvements vanish outside deterministic tasks.
```

---

## 16. What to plot first

Make these plots before running bigger experiments:

1. **Risky action value calibration**
   - x-axis: `p_success`
   - y-axis: predicted \(U(start, risky, goal)\) and exact \(U^*\)
   - methods: raw TRL, log TRL, stochastic-only, stochastic+TRL

2. **Policy regret vs stochasticity**
   - x-axis: slip/teleporter probability
   - y-axis: regret versus DP policy

3. **Long-horizon MSE vs distance**
   - x-axis: true shortest/effective distance bin
   - y-axis: MSE

4. **Triangle violation histogram**
   - x-axis: violation magnitude
   - y-axis: count

5. **Teleporter usage vs optimal teleporter usage**
   - x-axis: `p_good` or `p_dead`
   - y-axis: usage rate

6. **Effect of lambda_tr**
   - x-axis: `lambda_tr`
   - y-axis: long-horizon MSE, overestimation, success rate

The most compelling early figure is likely:

> A risky-shortcut plot showing raw TRL overestimates lucky paths, stochastic-only is calibrated but weak at long horizons, and stochastic+TRL keeps calibration while improving long-horizon value/policy.

---

## 17. Kill criteria and pivots

### Kill criteria

Seriously consider stopping if all are true:

- The stochastic-calibrated method does not improve over calibration-only in tabular long-horizon tests.
- The transitive term consistently increases overestimation at risky states.
- TMD dominates your method on PointMaze teleport with equal compute.
- The only wins are from deterministic tasks or numerical log-space stability.

### Pivot options

If results are mixed:

1. **Conservative stochastic TRL**
   - Make the transitive update one-sided so it cannot increase reachability beyond calibrated evidence.

2. **TRL as a regularizer for TMD**
   - Treat the contribution as faster path-relaxation for existing stochastic quasimetric methods.

3. **Uncertainty-aware TRL**
   - Penalize transitive compositions through high-variance waypoints.

4. **Subgoal selection under stochasticity**
   - Focus on choosing reliable waypoints rather than changing the value representation.

5. **Theory-only result**
   - Characterize when product-style TRL backups are biased under stochastic dynamics and when normalized successor distances fix the bias.

---

## 18. Most likely outcome and recommended first bet

The most likely useful path is:

```text
MC/contrastive stochastic calibration
+ log-space TRL transitive regularization
+ conservative/low-expectile backup
+ in-trajectory midpoint sampling
```

The first serious prototype should be:

```text
Successor-distance + TRL-log on risky shortcut and teleporter gridworld
```

The first serious benchmark should be:

```text
pointmaze-teleport-navigate-v0
pointmaze-teleport-stitch-v0
```

The first serious baseline to integrate should be:

```text
TMD
```

If you can show that TRL-style recursion improves stochastic successor-distance learning without adding risky optimism, the direction is worth pursuing.

---

## 19. Reference links

- Transitive RL paper: https://arxiv.org/abs/2510.22512
- TRL code: https://github.com/aoberai/trl
- Learning Temporal Distances: https://arxiv.org/abs/2406.17098
- Contrastive temporal-distance code: https://github.com/vivekmyers/contrastive_metrics
- OGBench project page: https://seohong.me/projects/ogbench/
- OGBench code: https://github.com/seohongpark/ogbench
- TMD paper / project: https://tmd-website.github.io/
- TMD code: https://github.com/vivekmyers/tmd-release/
