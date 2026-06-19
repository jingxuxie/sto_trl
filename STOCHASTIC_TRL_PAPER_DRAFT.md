# Stochastic Transitive RL: Draft Paper Skeleton

## Working Title

Stochastic Transitive RL: Divide-and-Conquer Goal Reaching Under Stochastic Dynamics

## Core Claim

Transitive RL gives a logarithmic-horizon divide-and-conquer backup for
deterministic goal-conditioned RL. In stochastic environments, the same realized
trajectory product backup can become over-optimistic: lucky stochastic
shortcuts are treated as reliable paths. The key idea here is to keep the
divide-and-conquer transitive backup, but compose calibrated stochastic
reachability estimates rather than realized path labels.

The resulting stochastic transitive backup propagates long-horizon safe routes
in `O(log H)` sweeps while avoiding optimistic bias from stochastic shortcuts.

## Abstract Draft

Long-horizon offline goal-conditioned reinforcement learning is typically
limited by a tradeoff between temporal-difference bias accumulation and
Monte-Carlo variance. Transitive RL recently showed that deterministic
goal-reaching tasks contain a triangle-inequality structure that can be turned
into a divide-and-conquer value backup, reducing the number of value recursions
needed to propagate length-`H` paths. However, deterministic transitive backups
can fail in stochastic environments: a lucky realized trajectory through a
risky shortcut may be composed as if it were a reliable path. We introduce
Stochastic Transitive RL, a calibrated transitive value iteration algorithm
that combines empirical Bellman expectation backups with transitive composition
of stochastic reachability probabilities. In tabular stochastic shortcut tasks,
the method learns the safe long-horizon policy with only logarithmic matched
sweeps, while realized-trajectory TRL, Monte-Carlo positive labels, and matched
Bellman backups choose the risky shortcut. Across five seeds and safe path
lengths 16 to 128, stochastic TRL achieves 1.000 success with zero regret,
where matched Bellman succeeds only at the shortcut probability. On an empirical
2D stochastic grid-shortcut benchmark with safe path lengths 31 to 127,
stochastic TRL also achieves 1.000 success with logarithmic matched sweeps,
while matched Bellman and MC-positive choose the risky portal. On an empirical
OGBench PointMaze teleport topology diagnostic inferred from the offline
dataset, stochastic TRL reaches 0.901 mean success with a 6-sweep budget and
matches a 180-sweep Bellman reference, while matched Bellman reaches 0.343. On
the harder PointMaze teleport-stitch diagnostic, stochastic TRL again reaches
0.901 and matches full Bellman, while matched Bellman reaches 0.343. On a fully
empirical PointMaze graph, stochastic TRL reaches 0.355 mean success with an
11-sweep budget, compared with 0.216 for matched Bellman and 0.402 for a
220-sweep Bellman reference. These results suggest that calibrated stochastic
transitive backups preserve TRL's horizon-efficiency benefits while addressing
its deterministic assumption. As learned-controller evidence on OGBench
AntMaze teleport, pairing the same topology planner with a learned
goal-conditioned BC controller and body-state-compatible waypoint goals reaches
0.947 mean success on navigate and 0.960 on the harder stitch variant across
three evaluation seeds and 20 episodes per task, matching the 180-sweep Bellman
reference while matched Bellman remains near 0.31. The AntMaze stitch result is
robust across three independently trained controllers, with stochastic TRL
staying in the 0.950-0.963 success range while matched Bellman remains near
0.32; the navigate result is also robust across three independently trained
controllers. As learned-module screen evidence, raw-observation transition and
tie-preserving policy heads preserve the AntMaze hard-task signal: with
transition seed 0 fixed, stochastic TRL reaches 0.933 on navigate and 0.967 on
stitch across three evaluation seeds, matching full Bellman while matched
Bellman reaches 0.283.

## Contributions

1. We identify a stochastic failure mode of deterministic/realized TRL: lucky
   stochastic trajectories are over-composed into high-value shortcuts.
2. We propose a stochastic transitive backup that composes calibrated
   reachability probabilities and takes a Bellman expectation backup at each
   sweep.
3. We show a long-horizon scaling result in stochastic shortcut MDPs: matched
   stochastic TRL succeeds across horizons up to 128 with `ceil(log2 H)+1`
   sweeps, while matched Bellman and MC-positive baselines choose the risky
   shortcut.
4. We add a 2D stochastic grid-shortcut benchmark that separates horizon
   propagation from lucky stochastic shortcut bias: stochastic TRL matches full
   Bellman across path lengths up to 127 while matched Bellman and MC-positive
   fail under the matched sweep budget.
5. We provide continuous-control evidence on OGBench PointMaze teleport: a
   dataset-inferred topology diagnostic reaches 0.901 mean success with the
   matched stochastic TRL budget on both navigate and stitch, and a fully
   empirical graph planner improves over matched Bellman under the same
   11-sweep budget.
6. We give an AntMaze learned-controller diagnostic: under the same 6-sweep
   inferred-topology budget and learned BC executor, stochastic TRL matches the
   180-sweep Bellman reference at 0.947 success on navigate and 0.960 on the
   harder stitch screen across three evaluation seeds and 20 episodes per task;
   both navigate and stitch remain high across three controller seeds.
7. We show that learned high-level raw-observation transition and
   tie-preserving policy heads preserve the AntMaze hard-task advantage over
   matched Bellman, while keeping the remaining cell-abstraction boundary
   explicit.

## Main Hard-Task Results

| env | executor | eval setting | Bellman matched | Stochastic TRL | Bellman full |
| --- | --- | --- | ---: | ---: | ---: |
| PointMaze teleport navigate | dataset topology scaffold | 5 seeds, 50 eps/task | 0.343 | 0.901 | 0.901 |
| PointMaze teleport stitch | dataset topology scaffold | 5 seeds, 50 eps/task | 0.343 | 0.901 | 0.901 |
| AntMaze teleport navigate | full-goal BC + body-nearest k16 | 3 seeds, 20 eps/task | 0.310 | 0.947 | 0.947 |
| AntMaze teleport stitch | full-goal BC + body-nearest k16 | 3 seeds, 20 eps/task | 0.317 | 0.960 | 0.960 |

These rows compare the same 6-sweep planning budget against the same
environment, executor, and rollout seeds. The AntMaze rows use one saved
task-specific controller per environment, so they are multi-evaluation-seed
screens rather than fully independent controller-training seeds. The paired
AntMaze improvement over matched Bellman is +0.637 on navigate with 95% CI
[0.525, 0.749] and +0.643 on stitch with 95% CI [0.494, 0.793]; the mean
gap from stochastic TRL to the 180-sweep Bellman reference is 0.000 in both
tasks.

## Relation to Deterministic TRL

Transitive RL starts from the deterministic GCRL identity that shortest-path
temporal distances satisfy a triangle inequality,
`d*(s,g) <= d*(s,w) + d*(w,g)`, or equivalently
`V*(s,g) >= V*(s,w)V*(w,g)` for discounted hitting values. The original TRL
paper turns this structure into a practical divide-and-conquer update by
training on in-trajectory behavioral subgoals with an expectile version of the
product target.

Our setting changes the object being composed. In stochastic dynamics, an
observed successful transition or trajectory segment is not a reliable edge:
it is one sample from a transition distribution. A deterministic transitive
closure over transition support can therefore over-compose lucky teleport or
portal outcomes. Stochastic TRL keeps the divide-and-conquer composition but
requires every composed edge to first be calibrated by an empirical stochastic
Bellman expectation. This makes the method a stochastic reachability planner,
not a deterministic shortest-path closure.

This distinction is visible in the PointMaze teleport-stitch ablation:
support-based deterministic TRL reaches only 0.449 mean success with the
6-sweep budget, whereas calibrated stochastic TRL reaches 0.901 and matches
180-sweep Bellman.

## Method

For a transition model `P`, define the discounted hitting value

```text
Q*(s,a,g) = sup_pi E[ gamma^{tau_g} 1{tau_g < infinity} | s0=s, a0=a ],
```

where `tau_g` is the first hitting time of goal `g` under the policy after the
first action. The learned table `Q(s,a,g)` estimates this quantity, with
`Q(g,a,g)=1`. Let `V(s,g)=max_a Q(s,a,g)`.

### Bellman Calibration

Given an empirical transition model `P_hat(s' | s,a)`, the stochastic Bellman
backup is

```text
(B Q)(s,a,g) = gamma * E_{s' ~ P_hat(.|s,a)} [ V(s',g) ],
(B Q)(g,a,g) = 1.
```

This is the calibration term that prevents stochastic outcomes from being
treated as deterministic successes.

### Transitive Composition

The stochastic transitive backup composes calibrated reachability:

```text
(T Q)(s,a,g) = max_w Q(s,a,w) V(w,g).
```

Intuition: reaching an intermediate subgoal `w` and then reaching `g` is a
valid policy class, so the product is a lower-bound style candidate path value
when `Q` is calibrated. Unlike realized TRL, `Q(s,a,w)` is not a binary
trajectory label; it is a stochastic reachability estimate.

For action values, the implementation excludes the tautological intermediate
`w=s` when `s != g`; otherwise `Q(s,a,s)=1` would copy `V(s,g)` into every first
action regardless of the action's actual transition. The implemented operator is

```text
(T Q)(s,a,g) = 1,                                  if s = g
(T Q)(s,a,g) = max_{w != s} Q(s,a,w) V(w,g),       otherwise.
```

### Stochastic TRL Update

The tabular prototype uses

```text
Q_{k+1} = max( B Q_k, T Q_k )
```

initialized from one-step empirical transitions. In deterministic environments,
this reduces to the usual transitive closure intuition. In stochastic
environments, the Bellman term calibrates stochastic branches while the
transitive term propagates long reliable routes quickly.

Algorithmically, the model-based diagnostic is:

```text
Input: empirical transition model P_hat, discount gamma, sweep budget K.
Initialize Q_0(s,a,g) = gamma P_hat(g | s,a), and Q_0(g,a,g)=1.
for k = 0, ..., K-1:
    V_k(s,g) = max_a Q_k(s,a,g), with V_k(g,g)=1
    Bellman(s,a,g) = gamma E_{s'~P_hat(.|s,a)} V_k(s',g)
    Transitive(s,a,g) = max_{w != s} Q_k(s,a,w) V_k(w,g)
    Q_{k+1}(s,a,g) = max(Bellman(s,a,g), Transitive(s,a,g))
    Q_{k+1}(g,a,g) = 1
Return greedy policy pi(s,g)=argmax_a Q_K(s,a,g).
```

The current continuous-control experiments use this exact high-level
stochastic planner on an empirical topology or graph, then isolate execution
with either a topology-level PointMaze controller or a shared AntMaze BC
controller. This is intentionally a clean algorithmic diagnostic before
claiming a full end-to-end neural TRL replacement.

## Proposition Candidates

These are draft statements to formalize before manuscript submission.

### Proposition 1: Lower-Bound Preservation

Fix a transition model `P`, let `Q*` be the optimal discounted hitting value for
that model, and define `T` with the `w != s` exclusion above. If
`0 <= Q <= Q*` pointwise, then both `B Q <= Q*` and `T Q <= Q*`. Therefore
`max(BQ,TQ)` preserves pointwise lower bounds.

Proof sketch: Bellman monotonicity gives `B Q <= B Q* = Q*`. For `T`, since
`Q <= Q*`, we have `Q(s,a,w)V(w,g) <= Q*(s,a,w)V*(w,g)`. The right-hand side is
the discounted value of a feasible composite policy that first tries to hit
`w`, then switches to a goal-reaching policy from `w` to `g`; if `g` is reached
before `w`, the true hitting value is only larger. Hence
`Q*(s,a,w)V*(w,g) <= Q*(s,a,g)`, and maximizing over `w` preserves the bound.

The one-step initialization `Q_0(s,a,g)=gamma P(g|s,a)` with
`Q_0(g,a,g)=1` is also a lower bound, so repeated stochastic TRL updates remain
lower-bounded by the optimal model value.

### Proposition 2: Logarithmic Safe-Path Propagation

In a deterministic path initialized with one-step reachability edges, after
`k` transitive sweeps the table is exact for path segments of length at most
`2^k`. Thus a length-`H` reliable route is represented after
`ceil(log2 H)` sweeps, whereas Bellman-only updates need `H-1` or `H` local
propagation sweeps depending on indexing.

Proof sketch: the base table contains all length-one edges. If all segments up
to length `m` are represented, then for any segment of length at most `2m`,
choosing the midpoint `w` composes two represented subsegments. Induction gives
coverage of length `2^k` after `k` sweeps. In the 16x8 grid budget curve, the
safe path length is 127 and the empirical switch occurs at 7 sweeps; the main
matched benchmarks use a conservative `ceil(log2 H)+1` schedule.

### Proposition 3: Risky Shortcut Calibration

In the stochastic shortcut MDP, suppose the safe route reaches the goal after
`H` reliable transitions and a risky route reaches the goal after `d`
transitions with probability `p`, otherwise entering a trap. If the empirical
transition model accurately estimates `p`, then the Bellman-calibrated risky
value is approximately `gamma^d p`, while the safe path value is `gamma^H`.
When `gamma^H > gamma^d p`, stochastic TRL selects the safe path after
logarithmic transitive propagation. Realized TRL and MC-positive labels can
instead condition on successful risky samples and estimate the shortcut around
`gamma^d`, overestimating by roughly a factor of `1/p`.

## Current Experimental Evidence

### Tabular Risky Shortcut Scaling

Source files:

- `results/tabular_safe_horizon_L16_128_5seed.csv`
- `results/paper_tables/tabular_safe_horizon.csv`
- `results/figures/tabular_horizon_success.svg`

Result:

- `L in {16, 32, 64, 128}`.
- `p_success in {0.02, 0.05}`.
- Five seeds.
- Stochastic TRL uses `ceil(log2 L)+1` sweeps: `5, 6, 7, 8`.
- Stochastic TRL success: `1.000` for every horizon and probability.
- Matched Bellman success: `0.018` to `0.054`, with risky-action rate `1.000`.
- MC-positive has the same risky shortcut failure mode.
- Full Bellman success: `1.000`.

Paper message: stochastic TRL preserves TRL's horizon efficiency while
calibrating stochastic shortcuts.

### Tabular Aggregate

Source files:

- `results/fast_tabular_compact.csv`
- `results/paper_tables/tabular_risky_aggregate.csv`

Key aggregate:

- Stochastic TRL: mean success `0.955`, regret `0.000`.
- Full Bellman: mean success `0.955`, regret `0.000`.
- Matched Bellman: mean success `0.436`, regret `0.315`.
- Realized TRL and MC-positive: mean success `0.436`, regret `0.315`, risky
  overestimate ratio about `4.58`.

### 2D Stochastic Grid Shortcut

Source files:

- `results/grid_shortcut_2d_5seed.csv`
- `results/paper_tables/grid_shortcut_2d.csv`
- `results/paper_tables/grid_realized_diagnostic.csv`
- `results/paper_tables/grid_budget_curve.csv`
- `results/figures/grid_shortcut_success.svg`
- `results/figures/grid_budget_curve.svg`

Result:

- Grids: `8x4`, `16x4`, `16x8`.
- Safe path lengths: `31`, `63`, `127`.
- Portal success probabilities: `0.02`, `0.05`.
- Five seeds.
- Stochastic TRL matched sweeps: `6`, `7`, `8`.
- Stochastic TRL success: `1.000` for every grid and probability.
- Matched Bellman and MC-positive choose the risky portal and reach mean
  success `0.036`.
- Full Bellman success: `1.000`.
- Reduced realized-TRL diagnostic: raw realized TRL and log realized TRL choose
  the risky portal and reach success `0.051`, matching the MC-positive failure
  mode; stochastic TRL reaches `1.000`.
- Planning-budget curve on the hardest `16x8` grid: Bellman stays at the
  portal success rate through `120` sweeps and first reaches success `1.000`
  at `126` sweeps; stochastic TRL reaches success `1.000` at `7` sweeps.

Paper message: the stochastic transitive backup gives logarithmic propagation
on a 2D long-horizon task while remaining calibrated against risky stochastic
portals.

### OGBench PointMaze Teleport Graph

Source files:

- `results/pointmaze_topology_dataset_5seed_ep50.csv`
- `results/paper_tables/pointmaze_topology_5seed.csv`
- `results/paper_tables/pointmaze_topology_paired_stats.csv`
- `results/pointmaze_graph_persistent_waypoint_matched_independent_seeded_paired50.csv`
- `results/paper_tables/pointmaze_graph_5seed.csv`
- `results/paper_tables/pointmaze_graph_paired_stats.csv`
- `results/paper_tables/pointmaze_graph_task_deltas.csv`
- `results/figures/pointmaze_graph_success.svg`

Dataset-inferred topology diagnostic:

| method | sweeps | mean success |
| --- | ---: | ---: |
| Bellman matched | 6 | 0.343 |
| Stochastic TRL | 6 | 0.901 |
| Bellman full | 180 | 0.901 |

This result infers free cells and stochastic teleport jumps from the offline
dataset, then uses that coarse topology as a routing scaffold. It is best
framed as a high-success algorithmic diagnostic rather than a fully neural
controller result. It shows that the stochastic transitive backup can match the
long Bellman reference with a logarithmic planning budget when low-level
execution is not the bottleneck. Paired seed-level improvement over matched
Bellman is `+0.558`, 95% CI `[0.501, 0.614]`.

Controlled evaluation detail: the teleport environment samples random exits
from global NumPy state, so the evaluator seeds `np.random` per rollout. The
paper-facing table also uses disjoint episode seed ranges across nominal seed
ids.

Teleport-stitch dataset-inferred topology diagnostic:

| method | sweeps | mean success |
| --- | ---: | ---: |
| Bellman matched | 6 | 0.343 |
| Stochastic TRL | 6 | 0.901 |
| Bellman full | 180 | 0.901 |

This uses `pointmaze-teleport-stitch-v0`, five evaluation seeds, and 50
episodes per task. It shows that the same stochastic transitive backup result
holds on the harder stitch variant: stochastic TRL again matches full Bellman
with the 6-sweep budget while matched Bellman remains local. The paired
seed-level improvement over matched Bellman is also `+0.558`, 95% CI
`[0.501, 0.614]`.

Deterministic-support ablation on PointMaze teleport stitch:

| method | sweeps | mean success |
| --- | ---: | ---: |
| Bellman matched | 6 | 0.343 |
| Support TRL | 6 | 0.449 |
| Stochastic TRL | 6 | 0.901 |
| Bellman full | 180 | 0.901 |

Support TRL treats every observed stochastic teleport outcome as a reliable
transition and then applies deterministic transitive composition. It improves
slightly over matched Bellman but remains far below stochastic TRL, supporting
the claim that stochastic calibration is necessary rather than transitive
composition alone.

Fully empirical graph result:

| method | sweeps | mean success |
| --- | ---: | ---: |
| Bellman matched | 11 | 0.216 |
| Stochastic TRL | 11 | 0.355 |
| Bellman full | 220 | 0.402 |

Paired seed-level statistics:

- Stochastic TRL minus matched Bellman: `+0.139`, 95% CI `[0.112, 0.166]`.
- The improvement is positive for all five seeds.
- Stochastic TRL recovers `0.750` of the gap between matched Bellman and the
  220-sweep Bellman reference.
- Task deltas show the matched-budget gain is concentrated on tasks 1 and 2,
  where matched Bellman has zero success and stochastic TRL reaches `0.384`
  and `0.228`.

Paper message: under the same small planning budget, stochastic transitive
composition can reach high success when the topology/executor is reliable, and
it still improves over matched Bellman in the fully empirical graph setting.
The remaining high-impact gap is to make the topology/executor learned rather
than privileged, then carry the same idea to AntMaze teleport.

### OGBench AntMaze Teleport Learned Controller

Source files:

- `scripts/run_antmaze_bc_topology_planner.py`
- `results/antmaze_bc_topology_summary.md`
- `results/antmaze_bc_topology_fullgoal_20k_ep3_seed0.csv`
- `results/antmaze_bc_topology_fullgoal_20k_ep3_seed0.json`
- `results/paper_tables/antmaze_bc_topology_20k_ep3_seed0.csv`

This diagnostic uses the same inferred-topology planner as the PointMaze
topology result, but executes subgoals in `antmaze-teleport-navigate-v0` with a
learned goal-conditioned BC controller. The controller is shared across all
planning methods.

Current best setting:

- Controller: deterministic BC MLP, hidden dims `256,256,256`.
- Goal representation: full future observation; `xy`-only goals were weaker.
- Training budget: 20k optimizer steps, batch size 2048, JAX on `cuda:0`.
- Planner budget: 6 matched sweeps on a 45-state inferred topology.
- Evaluation: seed 0, 3 episodes per task.

| method | sweeps | mean success | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bellman matched | 6 | 0.333 | 0.000 | 0.000 | 0.333 | 0.667 | 0.667 |
| Stochastic TRL | 6 | 0.933 | 1.000 | 1.000 | 0.667 | 1.000 | 1.000 |

Single-episode full-Bellman check: with the same 20k-step controller,
stochastic TRL and 180-sweep Bellman full both reach 0.800 success, while
matched Bellman reaches 0.200. This supports the interpretation that stochastic
TRL recovers the long-horizon full-Bellman plan, while the remaining misses are
mostly local-controller execution errors.

Paper message: AntMaze is no longer only a negative result. With a learned
controller that matches the TRL codebase's goal-conditioned setup more closely
than KNN, stochastic TRL shows a high-success preliminary result. This still
needs a multi-seed controller sweep before becoming a main paper claim.

### OGBench Teleport Stitch Screens

Source files:

- `results/teleport_stitch_screen_summary.md`
- `results/paper_tables/teleport_stitch_screen_seed0.csv`
- `results/pointmaze_topology_stitch_seed0_ep10.csv`
- `results/antmaze_bc_topology_stitch_fullgoal_20k_seed0.csv`
- `results/policies/antmaze_stitch_fullgoal_bc_20k.pkl`
- `results/figures/antmaze_bodyk16_multiseed.svg`
- `results/figures/antmaze_budget.svg`

PointMaze stitch, dataset-inferred topology, seed 0, 10 episodes per task:

| method | sweeps | mean success |
| --- | ---: | ---: |
| Bellman matched | 6 | 0.380 |
| Stochastic TRL | 6 | 0.980 |
| Bellman full | 180 | 0.980 |

AntMaze stitch, shared 20k-step full-observation BC controller, seed 0, one
episode per task:

| method | sweeps | mean success |
| --- | ---: | ---: |
| Bellman matched | 6 | 0.200 |
| Stochastic TRL | 6 | 1.000 |
| Bellman full | 180 | 1.000 |

Repeated AntMaze stitch check, same controller with 16 body-nearest candidate
waypoint goals, seed 0, 10 episodes per task:

| method | sweeps | mean success |
| --- | ---: | ---: |
| Bellman matched | 6 | 0.380 |
| Stochastic TRL | 6 | 0.940 |
| Bellman full | 180 | 0.960 |

Compact multi-seed AntMaze body-candidate screen, three evaluation seeds, five
episodes per task:

| env | Bellman matched | Stochastic TRL | Bellman full |
| --- | ---: | ---: | ---: |
| Navigate | 0.347 | 0.893 | 0.893 |
| Stitch | 0.333 | 0.907 | 0.907 |

Independent controller-seed AntMaze screens, `bc_seed=1`, three evaluation
seeds, 20 episodes per task:

| env | Bellman matched | Stochastic TRL | Bellman full |
| --- | ---: | ---: | ---: |
| Navigate | 0.320 | 0.933 | 0.933 |
| Stitch | 0.323 | 0.950 | 0.950 |

These robustness checks retrain the full-observation BC controllers with a
different seed and use the same 20-episode protocol as the headline AntMaze
table. They preserve the same algorithmic pattern: stochastic TRL matches the
180-sweep Bellman reference under the 6-sweep matched budget, while matched
Bellman remains local.

AntMaze stitch controller-seed aggregate, three controller seeds, three
evaluation seeds per controller, 20 episodes per task:

| controller seed | Bellman matched | Stochastic TRL | Bellman full |
| ---: | ---: | ---: | ---: |
| 0 | 0.317 | 0.960 | 0.960 |
| 1 | 0.323 | 0.950 | 0.950 |
| 2 | 0.323 | 0.963 | 0.967 |

Across stitch controller seeds 0, 1, and 2, stochastic TRL stays in the
0.950-0.963 success range at the 6-sweep budget, while matched Bellman remains
near 0.32. The seed-2 run is a near-match to full Bellman, trailing by 0.003
absolute success.

AntMaze navigate controller-seed aggregate, three controller seeds, three
evaluation seeds per controller, 20 episodes per task:

| controller seed | Bellman matched | Stochastic TRL | Bellman full |
| ---: | ---: | ---: | ---: |
| 0 | 0.310 | 0.947 | 0.947 |
| 1 | 0.320 | 0.933 | 0.933 |
| 2 | 0.310 | 0.940 | 0.940 |

Across navigate controller seeds 0, 1, and 2, stochastic TRL stays in the
0.933-0.947 success range at the 6-sweep budget, matches full Bellman, and
keeps an improvement of at least 0.613 over matched Bellman.

20-episode AntMaze hard-task check, three evaluation seeds:

| env | Bellman matched | Stochastic TRL | Bellman full |
| --- | ---: | ---: | ---: |
| Navigate | 0.310 | 0.947 | 0.947 |
| Stitch | 0.317 | 0.960 | 0.960 |

Paired evaluation-seed improvement over matched Bellman is +0.637 on navigate
with 95% CI [0.525, 0.749] and +0.643 on stitch with 95% CI [0.494, 0.793].
The mean full-Bellman minus stochastic-TRL gap is 0.000 on both tasks.

AntMaze planning-budget screens, seed 0, three episodes per task:

| env | Bellman 6 | Stochastic TRL 6 | Bellman 24 | Bellman 45 | Bellman full |
| --- | ---: | ---: | ---: | ---: | ---: |
| Navigate | 0.333 | 1.000 | 0.933 | 1.000 | 1.000 |
| Stitch | 0.333 | 1.000 | 0.933 | 1.000 | 1.000 |

AntMaze stitch executor ablation, seed 0, three episodes per task:

| executor | Stochastic TRL | Bellman full |
| --- | ---: | ---: |
| nearest-xy waypoint observation | 0.867 | 0.867 |
| 16 body-nearest waypoint candidates | 1.000 | 1.000 |

Paper message: the harder stitch variants strengthen the performance story.
Stochastic TRL reaches the full-Bellman long-horizon plan under the matched
small planning budget on both PointMaze and AntMaze stitch screens. With the
same body-nearest waypoint-goal rule, the compact multi-seed AntMaze navigate
and stitch checks both show stochastic TRL matching full Bellman and
substantially outperforming matched Bellman.

Learned transition-module screens:

- PointMaze teleport navigate/stitch, tabular-softmax transition model trained
  from collapsed offline cell changes, seed 0, 50 episodes per task:
  stochastic TRL reaches `0.916` on both environments, matching full Bellman,
  while matched Bellman reaches `0.408`.
- PointMaze teleport navigate/stitch, shared raw-observation MLP transition
  head trained from collapsed offline cell changes, three transition seeds,
  seed 0, 50 episodes per task: stochastic TRL reaches `0.916` on both
  environments, matching full Bellman, while matched Bellman averages `0.512`
  on navigate and `0.483` on stitch.
- AntMaze teleport navigate/stitch hard tasks, same learned-transition
  diagnostic plus learned BC executor, seed 0, 20 episodes per task:
  stochastic TRL matches full Bellman at `0.925` on navigate and `0.950` on
  stitch, while matched Bellman reaches `0.325`.
- A three-transition-seed robustness check at 10 episodes per hard task keeps
  stochastic TRL at `0.950` on AntMaze navigate and `1.000` on AntMaze stitch,
  always matching full Bellman.
- PointMaze teleport navigate/stitch, shared raw-observation MLP transition
  head plus a tie-preserving raw-observation policy head, seed 0, 20 episodes
  per task: stochastic TRL reaches `0.980` on both, matching full Bellman,
  while matched Bellman reaches `0.530`.
- AntMaze teleport navigate/stitch hard tasks, learned BC executor plus
  topology transitions and a tie-preserving raw-observation high-level policy
  head, seed 0, 20 episodes per task: stochastic TRL reaches `0.925` on
  navigate and `0.950` on stitch, matching full Bellman, while matched Bellman
  reaches `0.350` and `0.400`.
- AntMaze teleport navigate/stitch hard tasks, learned BC executor plus a
  raw-observation MLP jump-change transition head and tie-preserving
  raw-observation high-level policy head, three transition seeds, seed 0, 10
  episodes per task: stochastic TRL reaches `0.950` on navigate and `1.000` on
  stitch, matching full Bellman, while matched Bellman reaches `0.300` on both.
- With transition seed 0 fixed, the same combined AntMaze learned-module screen
  over three evaluation seeds reaches `0.933` on navigate and `0.967` on
  stitch, matching full Bellman, while matched Bellman reaches `0.283`.

These screens support the learned-module story, but they are not yet a fully
end-to-end neural TRL result.

## What Not To Claim

- Do not claim the neural TRL variants are solved. The tie-preserving
  PointMaze and AntMaze policy-head diagnostics are promising, but current
  continuous-control evidence still uses the cell abstraction/topology scaffold
  and isolated executors. The newest AntMaze screen combines raw-observation
  transition and policy heads, but still at the high-level cell abstraction.
- Do not use the older unseeded PointMaze mean-action comparison as a main
  result. It is superseded by the controlled teleport seeding fix.
- Do not use the older overlapping-seed PointMaze table as a main result. It is
  superseded by `pointmaze_graph_persistent_waypoint_matched_independent_seeded_paired50.csv`.
- Do not use the stopped `L=256, p in {0.02,0.05}` sweep for safe-optimal
  claims; those probabilities make the risky shortcut optimal under
  `gamma=0.98`.
- Do not claim stochastic TRL beats 220-sweep Bellman on PointMaze. The current
  claim is horizon efficiency: near-full-Bellman success with a small budget,
  and much better than matched Bellman.
- Do not claim graph-resolution tuning improved the PointMaze result. A
  two-seed screen over `cell_size in {0.75, 1.0, 1.25, 1.5}` found the existing
  `cell_size=1.0` configuration was still best.
- Do not claim the AntMaze learned-controller result is an end-to-end neural
  TRL result. It is now a three-controller topology-planner diagnostic for
  both navigate and stitch, but still uses task-specific BC executors.
- Do not claim the learned-transition screens solve full raw observation-level
  dynamics learning. The strongest PointMaze row now uses a shared
  raw-observation MLP transition head over the current cell abstraction, while
  AntMaze still learns high-level transition probabilities from sampled or
  jump-change outcomes.
- Do not claim the raw-observation value/control head is generally solved. A
  PointMaze diagnostic with the successful raw-observation transition MLP drops
  to 0.000 success for scalar value and single-label policy heads; the
  tie-preserving policy head is the positive exception and now has a compact
  AntMaze hard-task confirmation; see
  `results/raw_obs_value_policy_diagnostic.md`.
- The TRL-codebase GCFBC controller screen is a controller ablation, not main
  evidence. With body-nearest waypoint candidates it improves from 0.0 to 0.6
  success on AntMaze navigate task 3 after 20k updates, but remains well below
  the current MSE BC executor; see `results/gcfbc_controller_screen.md`.
- The official TRL neural-codepath PointMaze screens are also not main
  evidence. Clean direct-actor screens in `scripts/run_pointmaze_screen.py`
  peak at 0.080 for GCFBC and 0.120 for deterministic MSEBC; existing
  TRL/log/relax screens stay in the 0.000-0.100 range. Treat this as an
  actor/controller bottleneck diagnostic; see
  `results/neural_codepath_diagnostic.md`. The learned waypoint-executor
  screen reaches 1.000 on both PointMaze teleport variants, so the boundary is
  direct actor execution without explicit high-level waypoints, not stochastic
  TRL planning.

## Next Experiments

Completed since this draft note:

- Added the focused PointMaze teleport stitch task-5 table:
  `results/paper_tables/pointmaze_stitch_task5_ep100_seed01234.csv`.
- Promoted lower-bound preservation and logarithmic propagation into the
  compiled LaTeX draft.
- Added executable claim and LaTeX consistency verifiers.
- Added PointMaze learned-BC-controller screens: stochastic TRL reaches 1.000
  on both teleport navigate and stitch with saved 5k-step full-goal BC
  controllers across three evaluation seeds; these are now generated and
  source-verified appendix rows.
- Added learned transition-module screens: PointMaze reaches 0.916 on navigate
  and stitch with both table-softmax and raw-observation MLP transition heads
  (the raw MLP rows are now verified across three transition seeds), and
  AntMaze hard tasks reach 0.950 on navigate and 1.000 on stitch in the
  verified transition-seed robustness table, each matching full Bellman under
  the learned transition model.
- Promoted the learned-transition and PointMaze learned-controller screens into
  generated paper tables and source-backed verifiers.
- Added the combined AntMaze raw-observation transition plus tie-policy-head
  screen to generated tables, source-backed verifiers, and the main LaTeX
  experiment narrative.

Remaining experiments:

1. Extend the raw-observation MLP transition screens beyond the current
   cell-abstraction setting, or add a differentiable empirical-transition
   auxiliary loss for the official TRL code.
2. If compute allows, run a longer but still bounded neural experiment only
   after a short screen clears a plausible path to 0.90+ success.

## Paper Structure

1. Introduction: horizon curse, deterministic TRL, stochastic failure mode.
2. Background: offline GCRL, hitting reachability, TRL divide-and-conquer.
3. Failure of realized transitive backups in stochastic environments.
4. Stochastic Transitive RL: Bellman calibration plus transitive composition.
5. Theory: lower-bound preservation and logarithmic propagation.
6. Experiments:
   - Tabular risky shortcut bias.
   - Horizon scaling.
   - 2D stochastic grid shortcut.
   - OGBench PointMaze teleport empirical graph planner.
   - OGBench AntMaze teleport learned-controller diagnostic.
   - AntMaze learned high-level module diagnostic.
   - Negative neural/ablation results in appendix.
7. Discussion: toward learned stochastic transition modules and stochastic
   quasimetric/TR-style neural algorithms.
