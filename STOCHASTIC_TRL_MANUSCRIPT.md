# Stochastic Transitive Reinforcement Learning

## Abstract

Long-horizon offline goal-conditioned reinforcement learning is limited by the
number of value-propagation steps needed to transmit sparse reachability
signals. Transitive RL shows that deterministic goal-reaching tasks contain a
divide-and-conquer structure: if a policy can reach an intermediate subgoal and
then reach the final goal, long paths can be propagated in logarithmically many
transitive sweeps. This deterministic structure breaks under stochastic
dynamics. A lucky trajectory through a risky shortcut or teleporter is only one
sample from a transition distribution, but realized-path transitive backups can
compose it as if it were reliable. We introduce Stochastic Transitive RL, a
calibrated transitive value-iteration algorithm that composes stochastic
reachability probabilities rather than binary realized path labels. Each update
combines an empirical Bellman expectation backup, which calibrates stochastic
outcomes, with a transitive composition backup, which propagates reliable
long-horizon reachability. In tabular stochastic shortcut tasks and 2D
stochastic grid shortcuts, stochastic TRL learns safe long-horizon policies
with logarithmic matched sweep budgets while matched Bellman and MC-positive
baselines choose risky shortcuts. On OGBench PointMaze teleport navigate and
stitch topology diagnostics, stochastic TRL reaches 0.901 mean success with a
6-sweep budget, matching a 180-sweep Bellman reference while matched Bellman
reaches 0.343. On OGBench AntMaze teleport with learned BC executors,
stochastic TRL reaches 0.947 success on navigate and 0.960 on stitch, again
matching the long-sweep Bellman reference while matched Bellman remains near
0.31. The harder AntMaze stitch result remains above 0.95 success across three
independently trained controllers.

## 1. Introduction

Goal-conditioned reinforcement learning often needs to propagate sparse
success information across long horizons. A Bellman backup propagates this
information locally: a path of length `H` needs roughly `H` value-iteration
sweeps before the start state sees the goal. Transitive RL addresses this in
deterministic environments by exploiting a shortest-path triangle inequality.
If reaching `w` from `s` and reaching `g` from `w` are both possible, then a
divide-and-conquer backup can compose them and propagate length-`H` structure
in `O(log H)` sweeps.

The same idea is not directly valid in stochastic environments. A successful
rollout segment through a stochastic teleporter or risky shortcut is not a
reliable edge. It is one observed outcome from a transition distribution.
Composing such outcomes can turn lucky samples into overconfident plans. This
creates a tension: we want TRL's horizon efficiency, but we need stochastic
calibration.

This paper studies that tension and proposes Stochastic Transitive RL. The
central change is simple: compose calibrated stochastic reachability estimates,
not realized path labels. The resulting operator uses a Bellman expectation
backup to estimate the probability of reaching subgoals under stochastic
dynamics, and a transitive product backup to propagate long-horizon routes.

### Contributions

1. We identify a stochastic failure mode of deterministic or realized-path TRL:
   lucky stochastic outcomes are over-composed into high-value shortcuts.
2. We introduce a calibrated stochastic transitive backup for discounted
   goal-reaching values.
3. We give proof sketches for lower-bound preservation and logarithmic
   propagation on reliable paths.
4. We show controlled tabular and grid evidence that stochastic TRL preserves
   TRL-style horizon efficiency while avoiding risky shortcut optimism.
5. We provide OGBench PointMaze and AntMaze teleport diagnostics showing high
   long-horizon success under a matched 6-sweep budget, including AntMaze
   controller-seed robustness.

## 2. Problem Setting

We consider offline goal-conditioned control with a finite empirical transition
model or topology abstraction. For state `s`, action `a`, and goal `g`, define
the discounted hitting value

```text
Q*(s,a,g) = sup_pi E[ gamma^{tau_g} 1{tau_g < infinity} | s0=s, a0=a ],
```

where `tau_g` is the first hitting time of goal `g`, and the policy acts after
the first action. Let

```text
V(s,g) = max_a Q(s,a,g).
```

The experiments in this prototype use empirical transition/topology models for
planning and separate low-level executors for continuous control. This isolates
the value-propagation question: can calibrated transitive backups learn useful
long-horizon plans under stochastic dynamics faster than matched Bellman
updates?

## 3. Method

### Bellman Calibration

Given an empirical transition model `P_hat(s' | s,a)`, define

```text
(B Q)(s,a,g) = gamma E_{s' ~ P_hat(.|s,a)} [ V(s',g) ],
(B Q)(g,a,g) = 1.
```

This term calibrates stochastic outcomes. A transition into a teleporter does
not become a deterministic edge to every observed exit; it receives the
expected reachability value under the empirical exit distribution.

### Transitive Composition

The transitive candidate composes calibrated reachability through an
intermediate subgoal:

```text
(T Q)(s,a,g) = max_{w != s} Q(s,a,w) V(w,g),     if s != g,
(T Q)(g,a,g) = 1.
```

The exclusion `w != s` prevents the tautology `Q(s,a,s)=1` from copying
`V(s,g)` into every first action.

### Stochastic TRL Update

The algorithm starts from one-step empirical reachability,

```text
Q_0(s,a,g) = gamma P_hat(g | s,a),    Q_0(g,a,g)=1,
```

then iterates

```text
Q_{k+1} = max(B Q_k, T Q_k).
```

Pseudocode:

```text
Input: empirical transition model P_hat, discount gamma, sweep budget K
Initialize Q_0(s,a,g) = gamma P_hat(g | s,a), and Q_0(g,a,g)=1
for k = 0, ..., K-1:
    V_k(s,g) = max_a Q_k(s,a,g), with V_k(g,g)=1
    Bellman(s,a,g) = gamma E_{s'~P_hat(.|s,a)} V_k(s',g)
    Transitive(s,a,g) = max_{w != s} Q_k(s,a,w) V_k(w,g)
    Q_{k+1}(s,a,g) = max(Bellman(s,a,g), Transitive(s,a,g))
    Q_{k+1}(g,a,g) = 1
Return pi(s,g) = argmax_a Q_K(s,a,g)
```

## 4. Theory Sketch

### Proposition 1: Lower-Bound Preservation

Fix a transition model `P` and let `Q*` be the optimal discounted hitting value
for that model. If `0 <= Q <= Q*` pointwise, then `B Q <= Q*` and `T Q <= Q*`.
Therefore `max(BQ,TQ)` preserves pointwise lower bounds.

Proof sketch: Bellman monotonicity gives `B Q <= B Q* = Q*`. For the
transitive term, `Q(s,a,w) V(w,g)` is the value of a feasible composite policy:
first try to hit `w`, then switch to a policy for `g`. Since the optimal policy
for `g` can do at least as well as that composite policy, the product is at
most `Q*(s,a,g)`.

### Proposition 2: Logarithmic Propagation

In a deterministic path initialized with one-step edges, after `k` transitive
sweeps the table represents path segments of length at most `2^k`. Therefore a
reliable route of length `H` is propagated after `ceil(log2 H)` sweeps, while
Bellman-only updates require linear propagation.

Proof sketch: the base table contains all length-one edges. If all segments up
to length `m` are represented, then any segment of length at most `2m` can be
split at an intermediate point and composed. Induction gives the result.

### Proposition 3: Shortcut Calibration

In a stochastic shortcut MDP, suppose a safe route has value `gamma^H` and a
risky shortcut succeeds after `d` steps with probability `p`, otherwise
entering a trap. A calibrated Bellman model estimates the shortcut value near
`gamma^d p`. Realized-path TRL or MC-positive labels can condition on
successful shortcut samples and overestimate the shortcut near `gamma^d`. When
`gamma^H > gamma^d p`, stochastic TRL propagates the safe route while avoiding
the optimistic shortcut.

## 5. Experiments

### 5.1 Tabular Stochastic Shortcuts

Safe path lengths are 16, 32, 64, and 128. Stochastic TRL reaches `1.000`
success with matched logarithmic sweeps. Matched Bellman and MC-positive
baselines choose the risky shortcut and succeed only near the shortcut
probability.

Source: `results/paper_tables/tabular_safe_horizon.csv`.

### 5.2 2D Stochastic Grid Shortcuts

The hardest grid has safe path length 127. Stochastic TRL reaches `1.000`
success at 7 sweeps. Bellman remains at portal-rate success through 120 sweeps
and first reaches `1.000` success at 126 sweeps.

Source: `results/paper_tables/grid_budget_curve.csv`.

### 5.3 OGBench PointMaze Teleport

The PointMaze diagnostic uses a dataset-inferred topology scaffold and a
simple low-level topology executor. It tests whether the stochastic transitive
planner can select long-horizon routes through stochastic teleport dynamics.

| env | Bellman matched | Stochastic TRL | Bellman full |
| --- | ---: | ---: | ---: |
| teleport navigate | 0.343 | 0.901 | 0.901 |
| teleport stitch | 0.343 | 0.901 | 0.901 |

Both rows use five evaluation seeds, 50 episodes per task, 6 matched sweeps,
and a 180-sweep Bellman reference.

### 5.4 Calibration Ablation

On PointMaze teleport stitch, deterministic support TRL composes every observed
stochastic transition support edge as reliable. It improves over matched
Bellman but remains far below calibrated stochastic TRL.

| method | sweeps | success |
| --- | ---: | ---: |
| Bellman matched | 6 | 0.343 |
| support TRL | 6 | 0.449 |
| stochastic TRL | 6 | 0.901 |
| Bellman full | 180 | 0.901 |

Source:
`results/paper_tables/pointmaze_topology_stitch_support_baseline_5seed.csv`.

### 5.5 PointMaze Learned-Controller Screen

As an executor check, we replace the PointMaze proportional topology executor
with saved 5k-step full-goal BC controllers and 16 body-nearest waypoint
candidates. The high-level planner is unchanged.

| env | Bellman matched | Stochastic TRL | Bellman full |
| --- | ---: | ---: | ---: |
| teleport navigate | 0.323 | 1.000 | 1.000 |
| teleport stitch | 0.223 | 1.000 | 1.000 |

Both rows use three evaluation seeds and 20 episodes per task. Source:
`results/paper_tables/pointmaze_learned_controller_ep20_seed012.csv`.

### 5.6 OGBench AntMaze Teleport

AntMaze uses the same topology-level stochastic planner with a learned
goal-conditioned BC executor. Waypoint goals use full future observations and
16 body-nearest candidates so the target waypoint is compatible with the
current ant body state.

| env | Bellman matched | Stochastic TRL | Bellman full |
| --- | ---: | ---: | ---: |
| teleport navigate | 0.310 | 0.947 | 0.947 |
| teleport stitch | 0.317 | 0.960 | 0.960 |

Both rows use three evaluation seeds and 20 episodes per task. Stochastic TRL
uses the same 6-sweep budget as matched Bellman; full Bellman uses 180 sweeps.

### 5.7 AntMaze Controller Robustness

Both AntMaze teleport variants were evaluated across three independently
trained task-specific BC controllers under the same 20-episode protocol.

Navigate uses 50k-step BC controllers:

| controller seed | Bellman matched | Stochastic TRL | Bellman full |
| ---: | ---: | ---: | ---: |
| 0 | 0.310 | 0.947 | 0.947 |
| 1 | 0.320 | 0.933 | 0.933 |
| 2 | 0.310 | 0.940 | 0.940 |

Stitch uses 20k-step BC controllers:

| controller seed | Bellman matched | Stochastic TRL | Bellman full |
| ---: | ---: | ---: | ---: |
| 0 | 0.317 | 0.960 | 0.960 |
| 1 | 0.323 | 0.950 | 0.950 |
| 2 | 0.323 | 0.963 | 0.967 |

Across navigate controller seeds, stochastic TRL remains in the `0.933-0.947`
success range and matches full Bellman. Across stitch controller seeds,
stochastic TRL remains in the `0.950-0.963` success range; the seed-2 stitch
run trails full Bellman by `0.003` absolute success.

Source:
`results/paper_tables/antmaze_navigate_controller_seeds_ep20_seed012.csv` and
`results/paper_tables/antmaze_stitch_controller_seeds_ep20_seed012.csv`.

## 6. Discussion

The results support the intended paper claim: stochastic TRL recovers TRL's
horizon-efficiency benefits in stochastic environments by composing calibrated
reachability probabilities. The main empirical pattern is consistent across
tabular shortcuts, 2D stochastic grids, PointMaze teleport, and AntMaze
teleport diagnostics: matched Bellman remains local under small sweep budgets,
while stochastic TRL reaches or nearly reaches the long-sweep Bellman
reference.

The support-TRL ablation is important. It shows that transitive composition
alone is not sufficient: composing observed stochastic support helps only
moderately, whereas Bellman-calibrated stochastic composition reaches the
high-success solution.

## 7. Limitations

The continuous-control results are learned-controller topology diagnostics, not
yet a complete end-to-end neural implementation of stochastic TRL. The planner
uses an empirical topology or graph, and execution is isolated through either a
simple topology controller or a learned BC controller. This is deliberate for
algorithmic diagnosis, but the final paper should not claim that the neural
TRL replacement is solved unless an end-to-end learned value implementation is
added.

AntMaze navigate and stitch now both have three-controller coverage, but the
controllers remain task-specific BC executors rather than a single end-to-end
neural stochastic TRL agent.

## 8. Reproducibility

The primary reproduction entry point is `REPRODUCE_MAIN_RESULTS.md`. The main
claim verifier is:

```bash
/home/eston/anaconda3/envs/autoresearcher_sto_trl/bin/python scripts/verify_main_claims.py
```

The current verifier checks headline hard-task rows, paired seed improvements,
the deterministic-support ablation, AntMaze `bc_seed=1` robustness, and the
three-controller AntMaze navigate and stitch aggregates.

## 9. Remaining Manuscript Work

1. Convert this markdown draft into the target conference LaTeX template.
2. Turn the proposition sketches into polished theorem/proof statements.
3. Add final figures from `results/figures/`.
4. Write related work around deterministic TRL, stochastic shortest paths,
   goal-conditioned offline RL, and successor/reachability methods.
5. Keep limitation language explicit: current continuous-control results are
   topology-planner diagnostics with learned executors.
