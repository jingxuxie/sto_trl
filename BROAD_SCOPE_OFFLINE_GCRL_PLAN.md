# Broad-Scope Plan: Stochastic TRL as a Learned Offline GCRL Value Algorithm

**Date:** 2026-06-19  
**Goal:** move from “stochastic transitive planning / topology diagnostics” to a broader, high-impact paper in the same spirit and scope as the original TRL paper: **a new value-learning algorithm for offline goal-conditioned RL (GCRL)** that handles stochastic dynamics.

---

## 1. Executive summary

Yes, we should broaden the work.

The current table/model/topology results are valuable and should remain in the paper, but they should become the **operator validation layer**, not the endpoint. A high-impact stochastic TRL paper should ultimately show that the idea can be trained from offline data as a learned goal-conditioned value function, not only solved by exact table/topology planning.

The right expanded thesis is:

> Stochastic TRL is a Bellman-calibrated transitive **value-learning algorithm** for offline GCRL. It reduces long-horizon value propagation like TRL, but replaces deterministic realized-path composition with calibrated stochastic reachability composition, preventing lucky stochastic outcomes from being over-composed into overconfident shortcuts.

The current paper should be reorganized around three layers:

1. **Exact operator layer:** tabular and topology MDPs prove the stochastic operator is correct and diagnose failure modes.
2. **Learned empirical-MDP layer:** transitions or high-level modules are learned from offline data, showing the result is not just hand-built planning.
3. **Neural offline GCRL layer:** a learned critic/value algorithm is trained from offline data, compared against modern offline GCRL baselines.

The biggest upgrade is to add a **neural stochastic TRL critic** that can be trained from offline transitions with relabeled goals and sampled transitive waypoints.

---

## 2. Target paper identity

### Recommended title direction

- **Stochastic Transitive RL: Calibrated Divide-and-Conquer Value Learning for Offline GCRL**
- **Stochastic Transitive Value Learning for Offline Goal-Conditioned RL**
- **Calibrated Transitive Value Learning in Stochastic Goal-Conditioned MDPs**

### Target contribution statement

A strong version of the contribution list:

1. We identify a stochastic failure mode of deterministic TRL: realized-path or support-based transitive composition can overvalue lucky stochastic outcomes.
2. We introduce a Bellman-calibrated stochastic transitive value operator that composes stochastic reachability estimates rather than binary realized support.
3. We derive lower-bound / monotonicity properties and show deterministic reduction to the original TRL-style transitive propagation in deterministic MDPs.
4. We instantiate the operator as a learned offline GCRL critic trained from offline transitions, relabeled goals, Bellman calibration targets, and sampled transitive waypoint targets.
5. We show controlled stochastic shortcut, grid, and topology evidence that the operator fixes stochastic over-composition and preserves fast horizon propagation.
6. We evaluate learned stochastic TRL against TRL and offline GCRL baselines on stochastic and long-horizon OGBench tasks.

### What must change from the current draft

The current draft is strongest as a planning/operator paper. To reach the broader “new offline GCRL value algorithm” scope, it needs:

- a neural or at least function-approximation critic trained from offline data;
- standard offline GCRL baselines;
- deterministic reduction results showing it does not hurt deterministic TRL-style tasks;
- stochastic calibration diagnostics;
- compute-normalized comparisons;
- less reliance on table Q at execution time.

---

## 3. What should remain from the current work

Do **not** discard the current table/topology work.

It should become the clean foundation of the paper:

| Existing component | Role in broader paper |
|---|---|
| Tabular risky shortcut | Minimal counterexample and calibration proof-of-concept. |
| 2D stochastic grid shortcut | Long-horizon stochastic propagation demonstration. |
| PointMaze topology diagnostics | OGBench-relevant stochastic planning diagnostic. |
| AntMaze topology + BC executor | Continuous-control diagnostic isolating high-level value propagation. |
| Support TRL ablation | Shows transitive composition alone is insufficient. |
| Learned transition tables/MLPs | Bridge from hand-built model to learned empirical MDP. |
| Tie-policy / previous-action heads | Bridge from table Q to learned high-level execution. |
| Raw scalar value-head failure | Important limitation: low MSE is not enough; action ranking/ties matter. |

These are not weaknesses if framed correctly. They show why the learned stochastic critic should include calibration and action-ranking structure.

---

## 4. Main algorithm to add: learned stochastic TRL critic

## 4.1 Goal-conditioned critic

Train a critic:

```text
Q_theta(s, a, g) in [0, 1]
```

or a log-distance critic:

```text
d_theta(s, a, g) >= 0
Q_theta(s, a, g) = exp(-d_theta(s, a, g))
```

Recommended initial implementation:

- use `Q_theta` with sigmoid output for fast prototyping;
- also implement log-distance as an ablation because long horizons can underflow in probability space.

## 4.2 Bellman calibration loss

For offline transition `(s, a, s')` and relabeled goal `g`:

```text
y_B = 1                         if s reaches g immediately / is goal
    = gamma * max_a' Q_bar(s', a', g) otherwise
```

Loss:

```text
L_B = loss(Q_theta(s,a,g), stopgrad(y_B))
```

This is the stochastic calibration term. In stochastic environments, repeated or similar `(s,a)` contexts with different outcomes should average to the correct expected reachability.

## 4.3 Transitive waypoint loss

Sample waypoints `w` from a candidate set:

```text
W(s,g) = future states from trajectory
       + batch states
       + high-Q retrieved states
       + optional nearest/topological candidates
```

Candidate target:

```text
y_T(w) = Q_bar(s,a,w) * max_a' Q_bar(w,a',g)
y_T = max_w y_T(w)
```

Combined target:

```text
y = max(y_B, y_T)
```

Start with conservative variants:

```text
L_T = expectile_loss(Q_theta(s,a,g), stopgrad(y_T))
```

or:

```text
L_T = max(0, stopgrad(y_T) - Q_theta(s,a,g))^2
```

The conservative hinge version says: “if the composed route provides a higher reliable lower bound, raise the value,” but it avoids forcing equality for every noisy waypoint.

## 4.4 Ranking / policy loss

Your diagnostics show that low value MSE can fail if the learned head gets action ranking or tie semantics wrong. Add an action-ranking or policy-set loss from the start.

For each `(s,g)`, define near-optimal actions under the current target or table/reference:

```text
A_good(s,g) = {a : Q_target(s,a,g) >= max_a Q_target(s,a,g) - eps}
```

Use either:

```text
L_policy_set = binary_cross_entropy(policy_head(s,g), A_good)
```

or pairwise ranking:

```text
L_rank = max(0, margin - Q_theta(s,a_good,g) + Q_theta(s,a_bad,g))
```

Recommended total critic objective:

```text
L = L_B + lambda_T * L_T + lambda_rank * L_rank + lambda_cql * L_support
```

where `L_support` is optional but useful for offline OOD action conservatism.

## 4.5 Actor / policy extraction

Two options:

### Option A: actor-free discrete/high-level policy

For tabular, grid, and topology tasks:

```text
pi(s,g) = argmax_a Q_theta(s,a,g)
```

This is simplest and best for debugging the value-learning claim.

### Option B: goal-conditioned actor for continuous control

Train actor:

```text
pi_psi(s,g)
```

with advantage-weighted BC:

```text
weight = exp(beta * A_theta(s,a,g))
L_actor = weight * ||pi_psi(s,g) - a||^2
```

or use the same actor setup as the original TRL codebase for fair comparison.

Initial recommendation:

- first use high-level discrete/topology actions to validate the learned critic;
- then integrate into the original TRL/OGBench actor pipeline.

---

## 5. Algorithm variants to implement

Implement these in a single code path with flags.

| Variant | Bellman calibration | Transitive composition | Learned function approx | Purpose |
|---|---:|---:|---:|---|
| `bellman_td` | yes | no | yes | TD baseline. |
| `realized_trl` | no / weak | yes, trajectory positives | yes | original deterministic-style TRL baseline. |
| `support_trl` | no | support composition | table/topology | stochastic overoptimism baseline. |
| `sto_trl_critic` | yes | yes | yes | main method. |
| `sto_trl_log_critic` | yes | yes in distance space | yes | numerical/horizon ablation. |
| `sto_trl_no_rank` | yes | yes | yes | shows ranking loss matters. |
| `sto_trl_no_cal` | no | yes | yes | shows calibration matters. |
| `sto_trl_no_trans` | yes | no | yes | shows transitivity matters. |
| `table_sto_trl` | exact | exact | no | oracle/operator reference. |

---

## 6. Milestone roadmap

## M0. Refactor current paper claims

### Objective

Reframe the existing manuscript as the foundation for a learned offline GCRL algorithm.

### Actions

- Keep table/topology experiments as “operator diagnostics.”
- Add a section: “From stochastic transitive planning to learned value functions.”
- State that the current topology experiments isolate value propagation, following TRL’s value-learning motivation.
- Explicitly say the new goal is to train `Q_theta` from offline data.

### Deliverable

Revised intro and experiment overview.

---

## M1. Deterministic reduction and parity with original TRL

### Objective

Show that stochastic TRL does not harm deterministic TRL-style value learning.

### Environments

- deterministic chain;
- deterministic grid/corridor;
- deterministic PointMaze or OGBench non-teleport PointMaze;
- optionally deterministic AntMaze topology.

### Methods

- original / realized TRL;
- stochastic TRL critic;
- Bellman TD;
- table stochastic TRL;
- original TRL repo baseline if feasible.

### Metrics

```text
success_rate
value_mse_to_full_bellman
policy_agreement_with_full_bellman
number_of_sweeps_or_gradient_steps_to_threshold
overestimation/underestimation
```

### Acceptance criterion

Stochastic TRL should match original TRL/support TRL on deterministic tasks, or any gap must be explained by implementation/function approximation rather than the stochastic objective.

### Main paper use

One small table or appendix figure:

> In deterministic tasks, stochastic TRL reduces to deterministic transitive propagation.

---

## M2. Learned critic on tabular and small grid tasks

### Objective

Move beyond table value iteration while keeping ground truth available.

### Setup

Use small finite MDPs, but train an MLP critic from sampled offline transitions rather than direct table updates.

Inputs:

```text
state one-hot or coordinates
action one-hot
goal one-hot or coordinates
```

Offline data:

```text
mixed behavior policy
safe route trajectories
risky shortcut trajectories
random local data
held-out start-goal pairs
```

### Methods

- table full Bellman;
- table stochastic TRL;
- neural Bellman TD;
- neural realized TRL;
- neural stochastic TRL;
- neural stochastic TRL without ranking loss;
- neural stochastic TRL without Bellman calibration.

### Metrics

```text
success_rate
risky_action_rate
regret_to_DP_oracle
value_calibration_error
action_rank_accuracy
stitch-only pair error
long-horizon pair error
```

### Acceptance criterion

The neural stochastic TRL critic should recover the table stochastic TRL decision boundary and outperform neural realized TRL/support-style composition on stochastic shortcuts.

---

## M3. Shortcut phase diagram with learned critic

### Objective

Make the stochastic calibration story visually and quantitatively undeniable.

### Sweep

```text
safe_length H in {8, 16, 32, 64, 128}
p_success in {0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80}
gamma in {0.95, 0.98, 0.995}
```

### Plot

- x-axis: shortcut success probability;
- y-axis: safe path length;
- color: learned risky-action rate;
- overlay: true optimal boundary `gamma^H = p * gamma^d`.

### Methods

- neural Bellman TD;
- neural realized TRL;
- neural stochastic TRL;
- table full Bellman;
- table stochastic TRL.

### Acceptance criterion

Neural stochastic TRL should follow the true safe/risky boundary more closely than realized TRL and MC-positive baselines.

---

## M4. Cost-normalized propagation curves

### Objective

Defend against the objection that transitive sweeps are more expensive than Bellman sweeps.

### Environments

- tabular shortcut length 128;
- 2D grid shortcut length 127;
- PointMaze topology MDP;
- learned critic version if feasible.

### X-axes

```text
sweeps / gradient updates
wall-clock time
number of sampled waypoint candidates
estimated primitive backup operations
```

### Variants

```text
all-waypoint transitive backup
top-K waypoint backup, K in {4, 8, 16, 32, 64}
random-K waypoint backup
batch-waypoint backup
retrieval-based waypoint backup
```

### Acceptance criterion

Stochastic TRL should retain a meaningful advantage under at least one fair compute normalization. If not, the paper should claim horizon-depth efficiency rather than wall-clock superiority.

---

## M5. OGBench PointMaze with learned stochastic TRL critic

### Objective

First standard offline GCRL benchmark result for the learned algorithm.

### Environments

Start with state-based PointMaze:

- teleport navigate;
- teleport stitch;
- non-teleport navigate/stitch if available;
- deterministic PointMaze sanity tasks.

### Implementation path

Fork the TRL training stack and add:

```text
agents/sto_trl.py
agents/sto_trl_log.py
agents/sto_trl_ablate_no_cal.py
agents/sto_trl_ablate_no_trans.py
```

Use the same actor/controller interface as TRL where possible.

### Training data

Use OGBench offline datasets and relabel goals using the same convention as the baselines.

### Baselines

Minimum:

- GCBC;
- GCIQL;
- GCIVL;
- CRL;
- QRL;
- HIQL;
- original TRL;
- stochastic TRL critic.

Stretch:

- TMD;
- MQE;
- TTGS on top of learned values;
- LAVL if code is practical.

### Metrics

```text
official success_rate
sample/gradient steps
wall-clock training time
value calibration curves
stitch-task success
teleport/risky transition action rate
policy agreement with table/topology planner if available
```

### Acceptance criterion

On PointMaze teleport/stitch, learned stochastic TRL should beat original TRL/realized TRL and Bellman-style TD, and should be competitive with the strongest relevant offline GCRL baselines.

---

## M6. OGBench AntMaze with learned stochastic TRL value module

### Objective

Show the algorithm scales beyond point-mass control.

### Two tracks

#### Track A: high-level learned value module + shared BC executor

This is the safer bridge from current results.

- Use learned stochastic TRL high-level critic/policy head.
- Use the same learned BC executor across methods.
- Report as learned high-level value/control diagnostics.

#### Track B: original TRL-style end-to-end actor/critic integration

This is the broader-scope goal.

- Use standard OGBench training loop.
- Train `Q_theta` and actor jointly/offline.
- Compare against TRL and OGBench baselines.

### Environments

- antmaze teleport navigate;
- antmaze teleport stitch;
- antmaze non-teleport long-horizon tasks if feasible;
- eventually humanoidmaze or other long-horizon OGBench tasks.

### Acceptance criterion

A high-impact paper does not need to solve every AntMaze task, but it should show at least one setting where the learned stochastic value algorithm beats original TRL specifically because of stochastic calibration.

---

## M7. External stochastic/quasimetric baselines

### Objective

Position the work relative to modern offline GCRL and stochastic temporal-distance methods.

### Required baselines for serious submission

1. Original TRL.
2. CRL.
3. QRL/CMD-style quasimetric method.
4. TMD.
5. OGBench reference baselines: GCBC, GCIVL, GCIQL, HIQL.

### Strong additional baselines

- MQE, for multistep quasimetric long-horizon comparison.
- TTGS, because it is a test-time planning method and may be viewed as close to topology/graph planning.
- LAVL, because recent OGBench work claims strong value generalization performance.
- Prioritized sweeping / Gauss-Seidel Bellman for planning efficiency.

### Where to run baselines first

1. small stochastic shortcut/grid tasks;
2. PointMaze teleport state-based tasks;
3. AntMaze teleport hard slices;
4. broader OGBench tasks only after the setup is stable.

### Acceptance criterion

Stochastic TRL does not have to beat every baseline everywhere. It must show a unique advantage in **stochastic long-horizon value propagation**, especially where original TRL over-composes lucky outcomes or TD is too slow.

---

## 7. Neural stochastic TRL implementation details

## 7.1 Data sampling

Each batch should include:

```text
(s, a, s_next)
goal g_future from same trajectory
goal g_random from dataset
goal g_task if available
waypoints w_future
waypoints w_batch
waypoints w_retrieved
```

Use a mixture:

```text
50% future relabeled goals
25% random dataset goals
25% task/eval-style goals or hard negative goals
```

For stochastic calibration, make sure stochastic rows include failed outcomes, not just successful future relabels.

## 7.2 Avoiding positive-only bias

Do not train only on successful future goals. That reproduces the realized-path TRL failure mode.

Add:

- random goals not reached in the trajectory;
- Bellman bootstrapped targets for arbitrary goals;
- negative or low-value stochastic outcomes;
- calibration bins by predicted value.

## 7.3 Target networks and double critics

Use:

```text
Q1_theta, Q2_theta
Q_bar = min(Q1_bar, Q2_bar)
```

for target computation, especially in stochastic environments where max-over-waypoints can overestimate.

## 7.4 Conservative transitive target

Start with:

```text
y = max(y_B, stopgrad(y_T))
```

Then test conservative forms:

```text
y = max(y_B, min(y_T, y_cap))
y_T = quantile_or_expectile_over_waypoints(y_T_candidates)
```

Avoid aggressive max over too many out-of-distribution waypoints early.

## 7.5 Candidate waypoint selection

Test:

```text
future trajectory waypoint
batch state waypoint
nearest latent waypoint
top-K by Q(s,a,w)
top-K by V(w,g)
product top-K
random dataset waypoint
```

A publishable method should not rely on oracle midpoints.

## 7.6 Actor learning

Use the original TRL actor pipeline where possible. Add ablations:

- actor from stochastic TRL critic;
- actor from original TRL critic;
- same actor, different critic values;
- BC-only actor;
- advantage-weighted actor.

This isolates whether gains come from value learning rather than controller differences.

---

## 8. Experiment matrix for the final paper

## Main paper experiments

### Experiment 1: stochastic shortcut counterexample

Purpose:

- show original/realized TRL over-composes lucky outcomes;
- show stochastic TRL chooses the safe route when optimal.

Methods:

- MC-positive;
- realized TRL;
- support TRL;
- Bellman TD;
- stochastic TRL;
- full Bellman oracle.

### Experiment 2: learned critic phase diagram

Purpose:

- show neural stochastic TRL learns the correct stochastic decision boundary.

### Experiment 3: long-horizon propagation and compute normalization

Purpose:

- show stochastic TRL keeps the divide-and-conquer advantage fairly.

### Experiment 4: OGBench PointMaze stochastic tasks

Purpose:

- first standard benchmark evidence for learned stochastic TRL.

### Experiment 5: OGBench AntMaze stochastic tasks

Purpose:

- show scalability to locomotion and learned controllers.

### Experiment 6: ablations

Ablate:

- no Bellman calibration;
- no transitive composition;
- realized-path positive-only targets;
- support-only composition;
- no ranking loss;
- all-waypoint vs sampled-waypoint;
- probability-space vs log-space.

## Appendix experiments

- deterministic reduction;
- topology operator results;
- learned transition robustness;
- tie-policy / previous-action head diagnostics;
- negative raw scalar value-head diagnostic;
- extra seeds and hyperparameters.

---

## 9. Go/no-go thresholds

## Continue toward broad high-impact paper if

1. Neural stochastic TRL matches table stochastic TRL on small stochastic tasks.
2. Neural stochastic TRL follows the safe/risky phase boundary better than realized TRL.
3. On deterministic tasks, stochastic TRL matches original TRL within noise.
4. On PointMaze teleport/stitch, learned stochastic TRL beats original TRL or a deterministic/realized TRL baseline.
5. Gains remain after cost normalization or candidate-waypoint normalization.
6. At least one strong external baseline is included and does not erase the main story.

## Keep as operator/planning paper if

1. The neural critic remains unstable or fails on PointMaze.
2. Gains only appear with table/topology planning.
3. The learned critic works only with oracle topology or labels.
4. Original TRL already performs equally well on stochastic OGBench once tuned.

## Pivot if

1. Bellman calibration + transitive composition cannot be made stable under function approximation.
2. The method consistently overestimates stochastic shortcuts despite calibration.
3. External stochastic baselines such as TMD solve the same tasks with simpler objectives and stronger performance.

---

## 10. Recommended immediate execution order

### Step 1: implement neural stochastic TRL on tiny MDPs

Deliverable:

```text
agents/sto_trl_neural_toy.py
results/paper_tables/neural_shortcut_phase.csv
results/figures/neural_shortcut_phase.pdf
```

### Step 2: deterministic parity check

Deliverable:

```text
results/paper_tables/deterministic_reduction.csv
```

Show stochastic TRL matches original/realized TRL on deterministic tasks.

### Step 3: PointMaze learned critic screen

Deliverable:

```text
agents/sto_trl.py
results/paper_tables/pointmaze_learned_critic.csv
```

Use OGBench state observations first.

### Step 4: ablation grid

Deliverable:

```text
results/paper_tables/sto_trl_ablation.csv
```

Ablate calibration, transitivity, log-space, ranking loss, waypoint count.

### Step 5: external baseline comparison

Deliverable:

```text
results/paper_tables/external_baselines_pointmaze.csv
```

Start with original TRL, CRL, QRL, and TMD if available.

### Step 6: AntMaze learned-module or learned-critic scaling

Deliverable:

```text
results/paper_tables/antmaze_learned_sto_trl.csv
```

Keep controller fixed across methods.

### Step 7: write final paper around learned value algorithm

Only after Steps 1-6 should the paper be framed as broad offline GCRL value learning.

---

## 11. How to position against related work

### Original TRL

Position:

- original TRL gives deterministic divide-and-conquer value learning;
- stochastic TRL preserves the deterministic benefit but fixes stochastic realized-path over-composition.

Need to show:

- deterministic parity;
- stochastic advantage.

### OGBench baselines

Position:

- OGBench is the standard benchmark context for offline GCRL and includes tasks probing stitching, long-horizon reasoning, high-dimensional inputs, and stochasticity.

Need to show:

- standard baseline comparisons on at least PointMaze and ideally AntMaze.

### TMD / stochastic quasimetric methods

Position:

- TMD addresses stochastic optimal goal-reaching distances with quasimetric representations;
- stochastic TRL is different because it is a calibrated transitive **backup/operator** for divide-and-conquer value propagation.

Need to show:

- TMD baseline if possible;
- or clear conceptual distinction plus experiments where transitive propagation budget matters.

### MQE / multistep quasimetric methods

Position:

- MQE is a strong long-horizon quasimetric method;
- compare if feasible, especially on long-horizon tasks.

### TTGS / graph search

Position:

- TTGS is relevant because it adds test-time graph planning on top of learned GCRL values;
- your method learns the stochastic transitive value itself, but topology diagnostics may look similar to graph planning.

Need to show:

- no privileged graph reliance in main learned-critic claims;
- include TTGS if reviewers are likely to view your topology layer as graph search.

### LAVL and newer value-generalization methods

Position:

- if submitting in 2026, be aware of recent OGBench methods claiming strong value generalization;
- compare if feasible or clearly scope the paper around stochastic transitive calibration.

---

## 12. Suggested final paper structure

1. **Introduction**
   - deterministic TRL is fast but stochastic realized paths are dangerous;
   - stochastic GCRL needs calibrated divide-and-conquer value learning.

2. **Background**
   - offline GCRL;
   - discounted hitting values;
   - original TRL transitive composition;
   - stochastic failure mode.

3. **Stochastic Transitive Value Operator**
   - Bellman calibration;
   - transitive composition;
   - lower-bound/monotonicity;
   - deterministic reduction.

4. **Learned Stochastic TRL**
   - critic parameterization;
   - Bellman loss;
   - transitive waypoint loss;
   - ranking/conservative losses;
   - actor extraction.

5. **Controlled Experiments**
   - shortcut phase diagram;
   - grid propagation;
   - calibration curves;
   - compute normalization.

6. **Offline GCRL Experiments**
   - PointMaze;
   - AntMaze;
   - stochastic teleport/stitch tasks;
   - standard baselines.

7. **Ablations and Diagnostics**
   - no calibration;
   - no transitivity;
   - deterministic reduction;
   - learned transition robustness;
   - topology/operator appendix.

8. **Related Work**
   - TRL;
   - temporal distances and successor features;
   - quasimetric GCRL;
   - offline GCRL benchmarks;
   - graph/test-time planning.

9. **Limitations**
   - learned critic still not fully end-to-end on every environment if applicable;
   - stochastic transition estimation;
   - computational cost of waypoint max;
   - controller dependence.

---

## 13. Minimal high-impact result package

If time is limited, the smallest package that could justify the broader claim is:

1. deterministic parity with original TRL;
2. learned neural stochastic TRL on risky shortcut/grid phase diagrams;
3. learned stochastic TRL on OGBench PointMaze teleport/stitch;
4. original TRL and OGBench baseline comparisons on PointMaze;
5. topology/table OGBench AntMaze as supporting diagnostic;
6. cost-normalized propagation figure;
7. calibration/overestimation figure.

That would support the statement:

> We introduce stochastic TRL as a value-learning algorithm, validate its operator properties exactly, and show that a learned offline critic instantiation improves stochastic long-horizon GCRL over deterministic transitive and TD-style alternatives.

---

## 14. Key risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Neural critic fails despite table success | Would limit scope to planner/operator paper | Add ranking loss, log-space critic, double critic, conservative waypoint sampling. |
| Original TRL works after tuning | Weakens novelty | Create controlled stochastic tasks where realized-path over-composition is mathematically visible. |
| TMD/MQE dominate | Review risk | Emphasize divide-and-conquer backup and compute-normalized propagation; include calibration failure modes. |
| Transitive sweep too expensive | Review risk | Add sampled/top-K waypoint and cost-normalized curves. |
| OGBench controller confounds value learning | Review risk | Keep controller fixed; show value/policy agreement and topology diagnostics. |
| Topology appears privileged | Review risk | Promote learned critic and least-privileged settings; keep env-topology in appendix. |

---

## 15. Bottom line

To reach the impact/scope of the original TRL paper, the project should move from:

```text
table/topology stochastic transitive planning
```

to:

```text
learned stochastic transitive value learning from offline data
```

The current results are still essential: they prove the operator and expose the stochastic failure mode. But the main paper should add a neural offline critic trained with Bellman calibration plus sampled transitive waypoint composition, then compare against original TRL and modern offline GCRL baselines.

The desired final claim is:

> Stochastic TRL generalizes TRL from deterministic reachability composition to stochastic offline GCRL by learning calibrated reachability values and composing them transitively. It retains TRL’s fast long-horizon propagation while avoiding overoptimistic composition of lucky stochastic transitions.
