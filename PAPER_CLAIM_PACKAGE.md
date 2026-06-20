# Stochastic TRL Paper Claim Package

This file is the concise source of truth for paper claims, evidence, and
non-claims. It is grounded in the generated tables and
`results/main_claim_verification.md`.

## Core Thesis

Stochastic Transitive RL keeps TRL's divide-and-conquer horizon propagation but
replaces deterministic realized-path composition with calibrated stochastic
reachability composition. This preserves logarithmic propagation on reliable
long-horizon routes while avoiding optimistic composition of lucky stochastic
shortcut outcomes.

## Main Algorithm Claim

For an empirical transition model `P_hat`, Stochastic TRL iterates

```text
Q_{k+1}(s,a,g) = max(
    gamma E_{s'~P_hat(.|s,a)} max_{a'} Q_k(s',a',g),
    max_{w != s} Q_k(s,a,w) max_{a'} Q_k(w,a',g)
)
```

with `Q(g,a,g)=1`. The Bellman term calibrates stochastic transition
probabilities; the transitive term propagates long-horizon reachability.

## Headline Evidence

Verified by `scripts/verify_main_claims.py`.

| setting | matched Bellman | stochastic TRL | full Bellman | checked source |
| --- | ---: | ---: | ---: | --- |
| PointMaze teleport navigate | 0.343 | 0.901 | 0.901 | `results/paper_tables/main_hard_task_results.csv` |
| PointMaze teleport stitch | 0.343 | 0.901 | 0.901 | `results/paper_tables/main_hard_task_results.csv` |
| AntMaze teleport navigate | 0.310 | 0.947 | 0.947 | `results/paper_tables/main_hard_task_results.csv` |
| AntMaze teleport stitch | 0.317 | 0.960 | 0.960 | `results/paper_tables/main_hard_task_results.csv` |

All headline rows compare the same 6-sweep planning budget against a
180-sweep Bellman reference. PointMaze rows use five evaluation seeds and
50 episodes per task. AntMaze rows use three evaluation seeds and
20 episodes per task.

## Learned Neural Shortcut-Critic Evidence

Generated from `results/paper_tables/neural_shortcut_phase.csv`; the figure is
`results/figures/neural_shortcut_phase.pdf` and the detailed note is
`results/neural_shortcut_phase.md`.

| method | exact success | correct decision | risky action rate |
| --- | ---: | ---: | ---: |
| neural Bellman TD | 0.027 | 0.583 | 0.417 |
| neural stochastic TRL | 1.000 | 1.000 | 0.000 |
| neural support TRL | 0.093 | 0.000 | 1.000 |
| table stochastic TRL | 1.000 | 1.000 | 0.000 |
| table full Bellman | 1.000 | 1.000 | 0.000 |

This is the first learned value-function evidence: a small MLP critic trained
from empirical offline transitions on stochastic risky-shortcut MDPs with safe
lengths 16, 32, and 64. Neural stochastic TRL matches the table stochastic-TRL
and full-Bellman decisions under the logarithmic matched sweep budget, while
neural Bellman TD is too slow to learn the full safe route and neural
support-TRL over-composes the lucky risky shortcut. This supports the learned
operator story, but it is still a controlled finite-MDP screen rather than an
OGBench end-to-end actor result.

## PointMaze Joint-Action Critic Evidence

Generated from `results/pointmaze_qhead_target_fit_navigate_all_exact.csv`,
`results/pointmaze_qhead_target_fit_stitch_all_exact.csv`,
`results/pointmaze_qhead_target_fit_navigate_all_env_ep10_seed012.csv`, and
`results/pointmaze_qhead_target_fit_stitch_all_env_ep10_seed012.csv`, with
focused verification in `results/qhead_critic_claim_verification.md` and
`results/qhead_multiseed_env_claim_verification.md`.

| setting | qhead Bellman TD | qhead generated stochastic target | table matched Bellman | table stochastic TRL | qhead action agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| PointMaze teleport navigate | 0.066 | 1.000 | 0.476 | 1.000 | 0.966 |
| PointMaze teleport stitch | 0.072 | 1.000 | 0.481 | 1.000 | 0.959 |

Real environment rollout with the same PointMaze proportional controller:

| setting | task scope | eval seeds | qhead Bellman TD | qhead generated stochastic target | table matched Bellman | table stochastic TRL |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| PointMaze teleport navigate | all tasks, 10 eps/task | 0,1,2 | 0.000 | 0.933 | 0.393 | 0.933 |
| PointMaze teleport stitch | all tasks, 10 eps/task | 0,1,2 | 0.027 | 0.933 | 0.393 | 0.933 |
| PointMaze teleport navigate | tasks 4,5, 20 eps/task | 0 | 0.000 | 1.000 | 0.500 | 1.000 |
| PointMaze teleport stitch | tasks 4,5, 20 eps/task | 0 | 0.125 | 1.000 | 0.500 | 1.000 |

This is the next learned-critic bridge after the finite-MDP shortcut screen. A
joint-action critic `Q_theta(s,g) -> R^4`, trained on raw-observation
state-goal features plus state/goal identifiers with the learned
raw-observation transition head fixed, can carry a stochastic transitive target
generated internally by matched-budget stochastic target iteration. It reaches
1.000 success on PointMaze teleport navigate and stitch in exact high-level
model evaluation, 0.933 success across three evaluation seeds in all-task real
rollouts, and 1.000 success on hard tasks 4 and 5 in single-seed real
rollouts. The
matched-budget Bellman TD q-head and the raw self-bootstrapped stochastic q-head
remain low. This should be claimed as a two-phase generated-target learned
critic over the learned high-level MDP, not as a solved fully self-bootstrapped
neural fitted-iteration algorithm.
The self-bootstrap boundary is now verifier-covered by
`results/qhead_stabilization_boundary_verification.md`: monotone targets,
self-buffering, fresh projected iteration, high rank loss, and generated-target
action-label guidance all remain far below the generated-target q-head on the
PointMaze navigate exact screen. A reset-final target-buffer variant is a new
positive stabilization: after explicit target-buffer generation, resetting the
q-head and optimizer for final consolidation reaches 1.000 exact success on
both PointMaze navigate and stitch. In a three-evaluation-seed real-environment
support check, the same reset-final q-head reaches 0.933 on both variants,
matching generated-target q-head, table stochastic TRL, and full Bellman.
A top-4 bridge waypoint variant preserves these results while composing over
only four retrieved waypoints per state-goal pair.
A sampled target-buffer variant is also positive in a bounded support screen:
using 64 sampled offline next states per row and 32 sampled bridge candidates
per state-goal pair, while retaining four bridge waypoints per backup, it
reaches 1.000 exact success and 0.980 seed-0 real-environment success on both
PointMaze teleport variants, matching table stochastic TRL and full Bellman in
those rows. This is verified by
`results/qhead_sampled_buffered_verification.md`.

## Fast Hard-Task Stress Evidence

Generated from `results/paper_tables/hard_task_stress_seed0.csv` and verified
against raw CSVs by `scripts/verify_main_claims.py`.

| setting | task scope | matched Bellman | support TRL | stochastic TRL | full Bellman |
| --- | --- | ---: | ---: | ---: | ---: |
| PointMaze teleport stitch | tasks 4,5 | 0.420 | 0.480 | 0.930 | 0.930 |
| AntMaze teleport navigate | tasks 4,5 | 0.300 |  | 0.950 | 0.950 |
| AntMaze teleport stitch | tasks 4,5 | 0.350 |  | 1.000 | 1.000 |

These are single-evaluation-seed stress checks for fast iteration on harder
task slices. They should support discussion and debugging, not replace the
main multi-seed evidence.

The current fast-hard iteration table
`results/paper_tables/current_fast_hard_screen.csv` is also generated and
verified by `scripts/verify_main_claims.py`. It records the optimized exact
PointMaze model proxy and current real-environment hard-task checks: PointMaze
stitch tasks 4,5 reach 1.000 stochastic-TRL success in the exact proxy,
PointMaze navigate/stitch all-task seed-0 real-env checks reach 1.000, and
AntMaze navigate/stitch hard-task checks reach 0.900/1.000 while matching the
180-sweep Bellman reference. This is iteration evidence, not a replacement for
the headline multi-seed table.

## Focused PointMaze Single-Task Evidence

Generated from `results/paper_tables/pointmaze_stitch_task5_ep100_seed01234.csv`
and verified against raw CSVs by `scripts/verify_main_claims.py`.

| setting | task | matched Bellman | stochastic TRL | full Bellman |
| --- | ---: | ---: | ---: | ---: |
| PointMaze teleport stitch | 5 | 0.380 | 0.908 | 0.908 |

This is a five-seed, 100-episode-per-seed appendix check. It supports the
absolute-success story on a single long-horizon PointMaze task, but it does not
replace the all-task headline table.

## Learned-Transition Appendix Evidence

Generated from `results/paper_tables/pointmaze_learned_transition.csv` and
`results/paper_tables/antmaze_learned_transition_robustness.csv`, and verified
against raw CSVs by `scripts/verify_main_claims.py`.

| setting | transition fit | matched Bellman | stochastic TRL | full Bellman |
| --- | --- | ---: | ---: | ---: |
| PointMaze teleport navigate | offline cell changes, seed 0 | 0.408 | 0.916 | 0.916 |
| PointMaze teleport stitch | offline cell changes, seed 0 | 0.408 | 0.916 | 0.916 |
| PointMaze teleport navigate | raw-observation MLP cell changes, 3 transition seeds | 0.512 | 0.916 | 0.916 |
| PointMaze teleport stitch | raw-observation MLP cell changes, 3 transition seeds | 0.483 | 0.916 | 0.916 |
| AntMaze teleport navigate tasks 4,5 | 20 samples/row, 3 transition seeds | 0.300 | 0.950 | 0.950 |
| AntMaze teleport stitch tasks 4,5 | 20 samples/row, 3 transition seeds | 0.367 | 1.000 | 1.000 |
| AntMaze teleport navigate tasks 4,5 | raw-observation MLP jump changes, 3 transition seeds | 0.300 | 0.950 | 0.950 |
| AntMaze teleport stitch tasks 4,5 | raw-observation MLP jump changes, 3 transition seeds | 0.300 | 1.000 | 1.000 |

These screens show that the high-level stochastic transition model can be
learned from collapsed offline cell changes on PointMaze, including with a
shared raw-observation MLP transition head over XY/action features, and from
finite sampled outcomes or raw-observation jump-change transition heads on
AntMaze while preserving the matched-budget stochastic TRL advantage. They are
appendix evidence, not a replacement for the multi-evaluation-seed headline
topology table or a fully end-to-end neural transition/value algorithm.

## PointMaze Learned-Controller Appendix Evidence

Generated from
`results/paper_tables/pointmaze_learned_controller_ep20_seed012.csv` and
verified against raw CSVs by `scripts/verify_main_claims.py`.

| setting | controller | matched Bellman | stochastic TRL | full Bellman |
| --- | --- | ---: | ---: | ---: |
| PointMaze teleport navigate | 5k full-goal BC + body-nearest k16, 3 eval seeds | 0.323 | 1.000 | 1.000 |
| PointMaze teleport stitch | 5k full-goal BC + body-nearest k16, 3 eval seeds | 0.223 | 1.000 | 1.000 |

This screen shows that the PointMaze result is not tied to the simple topology
executor: a learned full-observation BC controller preserves the matched-budget
stochastic TRL advantage and reaches the full-Bellman reference on both
teleport variants.

The controller-isolation table
`results/paper_tables/controller_execution_isolation.csv` contrasts this
waypoint execution path against short direct final-goal actor screens in the
official loop. In that diagnostic, GCFBC peaks at 0.080 success and MSEBC peaks
at 0.120 on PointMaze teleport navigate, while stochastic TRL plus the learned
waypoint BC executor reaches 1.000 on PointMaze navigate and stitch.

## Tie-Preserving Policy-Head Appendix Evidence

Generated from `results/paper_tables/pointmaze_tie_policy_head_ep20_seed0.csv`,
`results/paper_tables/pointmaze_rawobs_transition_prev_policy_head_ep20_seed0.csv`,
`results/paper_tables/pointmaze_tie_policy_head_ep20_evalseed012.csv`,
`results/paper_tables/pointmaze_tie_policy_head_ep20_tseed012.csv`,
`results/paper_tables/antmaze_tie_policy_head_hard_tasks_ep20_seed0.csv`,
`results/paper_tables/antmaze_rawobs_transition_tie_policy_head_ep10_tseed012.csv`,
and
`results/paper_tables/antmaze_rawobs_transition_tie_policy_head_ep10_evalseed012.csv`,
then verified against raw CSVs by `scripts/verify_main_claims.py`.

| setting | control head | matched Bellman | stochastic TRL | full Bellman | action agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| PointMaze teleport navigate | raw-observation tie-policy MLP | 0.530 | 0.980 | 0.980 | 0.982 |
| PointMaze teleport stitch | raw-observation tie-policy MLP | 0.530 | 0.980 | 0.980 | 0.978 |
| PointMaze navigate, 3 eval seeds | raw-observation transition + tie-policy MLP | 0.417 | 0.927 | 0.927 | 0.982 |
| PointMaze stitch, 3 eval seeds | raw-observation transition + tie-policy MLP | 0.417 | 0.927 | 0.927 | 0.978 |
| PointMaze navigate, 3 transition seeds | raw-observation transition + tie-policy MLP | 0.530 | 0.980 | 0.980 | 0.978 |
| PointMaze stitch, 3 transition seeds | raw-observation transition + tie-policy MLP | 0.530 | 0.980 | 0.980 | 0.978 |
| PointMaze navigate, previous-action head | raw-observation transition + prev-action policy MLP | 0.530 | 0.980 | 0.980 | 1.000 |
| PointMaze stitch, previous-action head | raw-observation transition + prev-action policy MLP | 0.530 | 0.980 | 0.980 | 1.000 |
| AntMaze teleport navigate tasks 4,5 | raw-observation tie-policy MLP + 50k BC | 0.350 | 0.925 | 0.925 | 1.000 |
| AntMaze teleport stitch tasks 4,5 | raw-observation tie-policy MLP + 20k BC | 0.400 | 0.950 | 0.950 | 1.000 |
| AntMaze navigate, raw-observation transition | raw-observation tie-policy MLP + 50k BC | 0.300 | 0.950 | 0.950 | 0.999 |
| AntMaze stitch, raw-observation transition | raw-observation tie-policy MLP + 20k BC | 0.300 | 1.000 | 1.000 | 0.999 |
| AntMaze navigate, 3 eval seeds | raw-observation transition + tie-policy MLP + 50k BC | 0.283 | 0.933 | 0.933 | 0.999 |
| AntMaze stitch, 3 eval seeds | raw-observation transition + tie-policy MLP + 20k BC | 0.283 | 0.967 | 0.967 | 0.999 |
| AntMaze navigate, previous-action head, 3 eval seeds | raw-observation transition + prev-action policy MLP + 50k BC | 0.350 | 0.933 | 0.933 | 1.000 |
| AntMaze stitch, previous-action head, 3 eval seeds | raw-observation transition + prev-action policy MLP + 20k BC | 0.283 | 0.967 | 0.950 | 1.000 |
| AntMaze navigate, previous-action all tasks, 3 eps/task | raw-observation transition + prev-action policy MLP + 50k BC | 0.200 | 0.933 | 0.933 | 1.000 |
| AntMaze stitch, previous-action all tasks, 3 eps/task | raw-observation transition + prev-action policy MLP + 20k BC | 0.200 | 0.933 | 0.933 | 1.000 |
| AntMaze navigate, previous-action all tasks, 5 eps/task | raw-observation transition + prev-action policy MLP + 50k BC | 0.360 | 0.920 | 0.920 | 1.000 |
| AntMaze stitch, previous-action all tasks, 5 eps/task | raw-observation transition + prev-action policy MLP + 20k BC | 0.360 | 0.960 | 0.960 | 1.000 |

PointMaze also uses the learned raw-observation transition head; the three
evaluation-seed rows fix transition seed 0 and aggregate eval seeds 0, 1, and
2 over all tasks, while the three-transition-seed rows fix eval seed 0 and
aggregate transition seeds 0, 1, and 2. AntMaze uses the topology transition
model in the first two AntMaze rows, isolating the high-level policy-head
representation. The final two AntMaze rows combine the
raw-observation jump-change transition head with the tie-policy head across
three transition seeds and use 10 episodes per hard task. The final two rows
fix transition seed 0 and check three evaluation seeds on the same hard-task
scope.

The PointMaze previous-action rows are single-evaluation-seed all-task screens
with 20 episodes per task and 1000 value-head steps. The AntMaze
previous-action rows fix transition seed 0 and aggregate evaluation seeds 0, 1,
and 2 over tasks 4 and 5 with 10 episodes per task. Together they show that a
single-label neural high-level policy can recover the sticky tie-breaking
behavior when the previous high-level action is explicit, staying within 0.02
of the full-Bellman rollout reference in these diagnostics.
The AntMaze all-task previous-action rows are bounded seed-0 support screens
with 3 and 5 episodes per task, verified by
`scripts/verify_antmaze_alltask_prev_policy.py` and summarized in
`results/antmaze_alltask_prev_policy_summary.md`; they should not replace the
multi-eval-seed hard-task rows.

Additional AntMaze screen: fitting only observed offline jump/teleport rows
with `dataset_jump_changes` and keeping local moves at the topology scaffold
also preserves the hard-task result on one eval seed. On tasks 4 and 5 with
10 episodes per task, stochastic TRL reaches 0.950 on navigate and 1.000 on
stitch, matching full Bellman, while matched Bellman reaches 0.300 and 0.400.
The direct all-cell-change AntMaze target is a negative boundary: it corrupts
local rows and drops stochastic TRL/full Bellman to 0.500 in 5-episode screens.

## Controller Robustness Evidence

AntMaze stitch has three independently trained 20k-step BC controllers under
the same 20-episode protocol:

| controller seed | matched Bellman | stochastic TRL | full Bellman |
| ---: | ---: | ---: | ---: |
| 0 | 0.317 | 0.960 | 0.960 |
| 1 | 0.323 | 0.950 | 0.950 |
| 2 | 0.323 | 0.963 | 0.967 |

Source: `results/paper_tables/antmaze_stitch_controller_seeds_ep20_seed012.csv`.
The verifier checks stochastic TRL success at least 0.90, matched Bellman at
most 0.40, improvement at least 0.50, and full-reference gap at most 0.02.

AntMaze navigate now also has three independently trained 50k-step BC
controllers under the same 20-episode protocol:

| controller seed | matched Bellman | stochastic TRL | full Bellman |
| ---: | ---: | ---: | ---: |
| 0 | 0.310 | 0.947 | 0.947 |
| 1 | 0.320 | 0.933 | 0.933 |
| 2 | 0.310 | 0.940 | 0.940 |

Source: `results/paper_tables/antmaze_navigate_controller_seeds_ep20_seed012.csv`.
The verifier checks the same thresholds as the stitch aggregate.

## Stochastic Calibration Ablation

PointMaze teleport stitch, five seeds, 50 episodes per task:

| method | sweeps | success |
| --- | ---: | ---: |
| Bellman matched | 6 | 0.343 |
| deterministic support TRL | 6 | 0.449 |
| stochastic TRL | 6 | 0.901 |
| full Bellman | 180 | 0.901 |

Source: `results/paper_tables/pointmaze_topology_stitch_support_baseline_5seed.csv`.
This supports the claim that stochastic calibration matters beyond transitive
composition over observed support.

## Long-Horizon Scaling Evidence

Tabular safe-optimal shortcut tasks:

- Safe path lengths 16, 32, 64, and 128.
- Stochastic TRL reaches `1.000` success with matched logarithmic sweeps.
- Matched Bellman and MC-positive choose the risky shortcut and succeed only
  near the shortcut probability.

2D stochastic grid budget curve:

- Hardest 16x8 grid safe path length is 127.
- Stochastic TRL switches to `1.000` success at 7 sweeps.
- Bellman remains at portal-rate success through 120 sweeps and first reaches
  `1.000` success at 126 sweeps.

Sources:

- `results/paper_tables/tabular_safe_horizon.csv`
- `results/paper_tables/grid_budget_curve.csv`

## Paper-Ready Claims

- Stochastic TRL substantially improves matched-budget long-horizon success in
  stochastic shortcut/teleport settings.
- The improvement comes from calibrated stochastic reachability composition,
  not merely deterministic transitive closure over observed support.
- On PointMaze teleport navigate/stitch and AntMaze teleport navigate/stitch,
  stochastic TRL reaches or nearly reaches the long-sweep Bellman reference at
  the matched 6-sweep budget.
- On controlled stochastic shortcut MDPs, a learned MLP stochastic-TRL critic
  reaches 1.000 exact success across safe lengths 16, 32, and 64, while neural
  Bellman TD and neural support-TRL fail under the same matched budget.
- On PointMaze, the result also survives replacing the simple topology executor
  with saved 5k-step full-goal BC controllers across three evaluation seeds.
- On both AntMaze navigate and stitch, high success is robust across three
  independently trained BC controllers.
- Finite-sample high-level transition fitting preserves the stochastic TRL
  advantage on PointMaze and AntMaze learned-controller screens.
- On PointMaze, a shared raw-observation MLP transition head trained from
  offline cell-change targets preserves 0.916 stochastic-TRL success on both
  teleport navigate and stitch across three transition seeds.
- On PointMaze teleport navigate and stitch, a raw-observation MLP transition
  head plus a tie-preserving raw-observation policy head reaches 0.980
  stochastic-TRL success on one evaluation seed, matching full Bellman while
  matched Bellman reaches 0.530.
- On PointMaze teleport navigate and stitch, replacing the tie mask with a
  previous-action-conditioned single-label policy head also reaches 0.980
  stochastic-TRL success on one evaluation seed, matching full Bellman while
  matched Bellman reaches 0.530.
- On AntMaze hard tasks, a raw-observation tie-preserving high-level policy
  head with the learned BC executor keeps stochastic TRL high on one evaluation
  seed with 20 episodes per task: 0.925 on navigate and 0.950 on stitch.
- Combining AntMaze raw-observation MLP jump-change transitions with the
  raw-observation tie-policy head reaches 0.950 on navigate and 1.000 on
  stitch across three transition seeds with 10 episodes per hard task, matching
  full Bellman while matched Bellman is 0.300.
- With transition seed 0 fixed, the same combined AntMaze learned-module screen
  reaches 0.933 on navigate and 0.967 on stitch across three evaluation seeds,
  matching full Bellman while matched Bellman is 0.283.
- The previous-action version of the same AntMaze learned-module screen also
  remains positive across all five tasks in seed-0 support checks: with 5
  episodes per task it reaches 0.920 on navigate and 0.960 on stitch, matching
  full Bellman while matched Bellman reaches 0.360.
- A raw-observation joint-action q-head diagnostic is positive on AntMaze
  navigate hard tasks, reaching 1.000 stochastic-TRL success and matching the
  learned full-Bellman q-head while the matched Bellman q-head reaches 0.333.
  The same q-head is a boundary on AntMaze stitch: stochastic and full-Bellman
  q-heads both reach only 0.667, and 10000 fitting steps do not improve task 5.
  A previous-action-conditioned q-head also fails on stitch, with stochastic
  and full-Bellman q-head success both at 0.000 in the current hard-task screen.
  This is verified by `results/antmaze_qhead_hard_task_verification.md`.
- A focused path-following diagnostic shows persistent waypoint tracking is a
  non-promoted executor ablation: on AntMaze navigate hard tasks 4 and 5 it
  reaches 0.917 stochastic-TRL success across three evaluation seeds while
  matched Bellman remains 0.000, but the existing greedy hard-task aggregate is
  slightly stronger and persistent pathing hurts AntMaze stitch task 5.
- On AntMaze hard tasks, a shared raw-observation MLP transition head trained
  from offline jump-change targets preserves 0.950 success on navigate and
  1.000 on stitch across three transition seeds with the learned BC executor.
- A less privileged AntMaze jump-transition screen suggests the important
  stochastic rows can be recovered from offline jump changes, although local
  AntMaze rows still rely on the topology scaffold.

## Non-Claims

- Do not claim this is already a complete end-to-end neural TRL algorithm.
  The neural shortcut critic is learned, but current continuous-control
  results use a model/topology-level stochastic planner plus a learned BC
  executor.
- Do not claim the raw-observation value/control story is generally solved.
  The scalar value MLP and previous-action-free single-label policy MLP
  diagnostics in `results/raw_obs_value_policy_diagnostic.md` drop to 0.000
  success. The tie-preserving policy head and previous-action policy head are
  positive diagnostics, but they are still learned high-level heads over the
  cell abstraction/topology scaffold rather than complete end-to-end neural TRL
  agents.
- Do not claim a fully self-bootstrapped PointMaze q-head critic is solved.
  Fresh projected iteration reaches 0.000 exact success, and a guided-rank
  variant with generated stochastic-TRL action labels reaches only 0.119 even
  with `rank_ce_weight=10.0`. The reset-final target-buffer q-head is positive
  in exact PointMaze model evaluation and in three-evaluation-seed PointMaze
  real-environment support checks, but it still relies on explicit
  target-buffer generation over the learned high-level transition model. The
  top-4 bridge variant reduces all-waypoint composition but does not remove the
  learned high-level transition matrix. The sampled target-buffer variant
  reduces explicit matrix-backup dependence and reaches high seed-0 success, but
  it is still a high-level sampled-target diagnostic rather than a projected
  target-network self-bootstrap algorithm. A direct sampled target-network
  projection with the same sample counts remains low on PointMaze navigate
  exact evaluation, even with heavier fitting. Removing the final reset-fit from
  the sampled target-buffer variant, or replacing it with reset-each-iteration
  target replay, also fails in the current navigate exact screens.
- Do not claim the AntMaze q-head replaces the previous-action policy head yet.
  It solves navigate hard tasks, but the learned q-head full-Bellman reference
  itself reaches only 0.667 on stitch tasks 4 and 5 in the current screen. The
  previous-action-conditioned q-head variant is worse on stitch, with
  stochastic and full-Bellman q-head success both at 0.000.
- Do not claim stochastic TRL beats full Bellman. The claim is horizon
  efficiency: matching or nearly matching the long-sweep Bellman reference with
  far fewer sweeps.
- Do not claim the empirical graph planner solves PointMaze teleport stitch.
  The less-privileged graph check on `pointmaze-teleport-stitch-v0` reaches
  only 0.144 stochastic-TRL success, 0.168 matched Bellman success, and 0.078
  full-Bellman success over five independent evaluation seeds with 50 episodes
  per task. Since the 220-sweep full Bellman policy also fails under this
  executor, this is a graph/execution boundary rather than a contradiction of
  the verified high-success topology and learned-controller stitch results. A
  task-4 executor sweep over the available graph decoders found `transition_value`
  was best but still only 0.133 mean success over three seeds; see
  `results/pointmaze_graph_summary.md`. A follow-up learned-BC graph waypoint
  executor also scored 0.000 on a 10-episode seed-0 task-4 check, which points
  to the empirical graph path/representation as the current bottleneck.
- Do not use the older weak KNN/snippet controller screens as main evidence.
- Do not use current GCFBC controller screens as main evidence. After adding
  body-compatible waypoint candidates, GCFBC improves on AntMaze navigate task
  3 but reaches only 0.600 success after 20k updates in the latest short screen;
  on harder AntMaze navigate tasks 4 and 5 it reaches 0.000 success after 10k
  GPU updates with the same body-nearest waypoint setup;
  see `results/gcfbc_controller_screen.md`.
- Do not use the official TRL neural-codepath PointMaze screens as main
  evidence. Clean direct-actor diagnostics peak at 0.080 for GCFBC and 0.120
  for deterministic MSEBC, while existing TRL/log/relax screens stay in the
  0.000-0.100 range; see `results/neural_codepath_diagnostic.md`.
  The corresponding learned waypoint-executor screen reaches 1.000 on both
  PointMaze teleport variants, so the boundary should be described as direct
  actor execution being weak, not as a failure of stochastic TRL planning.

## Current Paper Package

- `paper/stochastic_trl/main.tex` is a compiled LaTeX draft with method,
  theorem statements, proof sketches, experiments, a main-section learned
  high-level module diagnostic, limitations, and appendix tables.
- `scripts/verify_main_claims.py` verifies headline tables, hard-task stress
  rows, the focused PointMaze task-5 row, paired seed/eval-seed checks,
  learned-transition appendix screens, the PointMaze learned-controller
  appendix screen, support ablations, and AntMaze controller-seed aggregates.
- `scripts/verify_latex_claims.py` verifies that the LaTeX table rows match the
  generated paper-table CSVs.
- AntMaze wording should remain "learned-controller topology diagnostic" unless
  an end-to-end neural implementation is added.

## Remaining High-Impact Gaps

1. Replace the remaining target-buffer generation in the reset-final q-head path
   with a stronger projected update. The sampled offline target-buffer screen is
   positive on PointMaze, but it still forms an explicit final target buffer
   before consolidation; direct sampled target-network, no-final buffer, and
   reset-each-iteration replay variants are negative boundaries, not the
   solution.
2. Extend the raw-observation transition/value story beyond the current
   high-level cell-abstraction screens, ideally to a differentiable auxiliary
   loss in the TRL codebase that reproduces the topology-level stochastic TRL
   signal without a discrete planner.
