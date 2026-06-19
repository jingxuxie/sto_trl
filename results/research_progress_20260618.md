# Stochastic TRL Progress - 2026-06-18

## Upstream Code

- Official TRL repo cloned at `external/trl`.
- Remote: `https://github.com/aoberai/trl`.
- Base commit checked during this run: `adfec6f`.

## Tabular Evidence

The toy stochastic MDP is a two-route long-horizon task: a deterministic safe
chain competes with a short stochastic risky shortcut. Realized-trajectory TRL
overestimates lucky risky trajectories, while the stochastic transitive backup
combines a sampled transition/Bellman update with transitive composition.

Key long-horizon files:

- `results/fast_tabular_compact.csv`
- `results/tabular_horizon_sweep.csv`
- `results/tabular_L128_fast.csv`

At `L=128` with three seeds:

| p_success | optimal first action | method | risky action rate | success | mean regret |
| --- | --- | --- | --- | --- | --- |
| 0.02 | safe | matched Bellman, 8 sweeps | 1.000 | 0.017 | 0.056 |
| 0.02 | safe | MC positive | 1.000 | 0.017 | 0.056 |
| 0.02 | safe | stochastic TRL, 8 sweeps | 0.000 | 1.000 | 0.000 |
| 0.05 | safe | matched Bellman, 8 sweeps | 1.000 | 0.038 | 0.027 |
| 0.05 | safe | MC positive | 1.000 | 0.038 | 0.027 |
| 0.05 | safe | stochastic TRL, 8 sweeps | 0.000 | 1.000 | 0.000 |
| 0.10 | risky | matched Bellman, 8 sweeps | 1.000 | 0.080 | 0.000 |
| 0.10 | risky | stochastic TRL, 8 sweeps | 0.667 | 0.388 | 0.007 |
| 0.20 | risky | matched Bellman, 8 sweeps | 1.000 | 0.192 | 0.000 |
| 0.20 | risky | stochastic TRL, 8 sweeps | 1.000 | 0.192 | 0.000 |

Interpretation: in safe-optimal long-horizon stochastic cases, stochastic TRL
finds the safe policy with few transitive sweeps, while realized positive
baselines and matched Bellman sweeps prefer the risky shortcut. This is the
strongest current result.

Added a corrected five-seed horizon-scaling sweep:

- File: `results/tabular_safe_horizon_L16_128_5seed.csv`.
- Summary artifact: `results/paper_tables/tabular_safe_horizon.csv`.
- Figure: `results/figures/tabular_horizon_success.svg`.
- Horizons: `L in {16, 32, 64, 128}`.
- Risk probabilities: `p_success in {0.02, 0.05}`.
- Methods: MC-positive, MC-all-goals, matched Bellman, stochastic TRL, full
  Bellman.

Across all tested safe-optimal horizons and probabilities:

- Stochastic TRL reaches success `1.000`, risky-action rate `0.000`, and regret
  `0.000` using `ceil(log2(L)) + 1` matched sweeps.
- Matched Bellman keeps choosing the risky shortcut with risky-action rate
  `1.000`; success stays near the shortcut probability (`0.018` to `0.054`).
- MC-positive has the same risky-shortcut failure mode as matched Bellman.
- Full Bellman reaches success `1.000`, serving as the high-sweep reference.

An attempted `L=256` sweep with `p_success in {0.02, 0.05}` was stopped and is
not used for claims because those probabilities make the risky shortcut
optimal under `gamma=0.98`; that setting does not test the intended
safe-optimal stochastic failure mode.

## OGBench PointMaze Screens

Environment: `pointmaze-teleport-navigate-v0`, seed 0, FRS actor, hidden dims
`(128, 128)`, batch size 256, W&B disabled.

Existing signals before the TD variant:

- Raw `trl`, 10k FRS: overall success `0.10`.
- `trl_log_relax_mc`, 10k FRS: overall success `0.10`.
- Raw `trl`, 30k FRS: overall success `0.00`.
- `trl_log_relax_mc`, 30k FRS: overall success `0.10`.
- `trl_log_relax_mc`, partial 100k run stopped around 60k: 50k eval still `0.10`.

New TD variant:

- Agent: `trl_log_relax_td`.
- Change: one-sided log transitive relaxation plus symmetric q-space one-step
  TD calibration using dataset `next_observations` and `next_actions`.
- Run dir: `results/ogbench_screen_exp/dummy/pointmaze_teleport_10k_frs_td/sd000_20260618_210313`.
- 5k eval: overall success `0.00`.
- 10k eval: task2 success `0.20`, overall success `0.04`.

Interpretation: the TD variant is stable but does not improve the continuous
control result. It is weaker than `trl_log_relax_mc` on this short screen.

## Current Decision

The tabular result supports the paper idea, but the neural continuous-control
port is not yet strong enough. The next algorithmic step should be closer to
the tabular stochastic backup: estimate or sample a local transition
distribution and apply a Bellman expectation/max target, rather than relying
only on realized future goals, MC anchors, random negatives, or behavior-action
TD.

## Empirical OGBench Graph Planner

Added `scripts/run_pointmaze_graph_planner.py`, a model-based continuous
prototype that applies the successful tabular backup directly to an empirical
PointMaze graph:

- Quantize xy observations into cells.
- Bin continuous actions by direction.
- Estimate `P(next_cell | cell, action_bin)` from the offline dataset.
- Solve reachability with Bellman only, Bellman plus stochastic transitive
  backup, or transitive warm-start followed by Bellman polish sweeps.
- Evaluate by nearest-cell action lookup or persistent graph-waypoint execution
  in the real OGBench environment.

Important correction: the teleport environment uses global NumPy randomness
for teleport exits. `evaluate_agent` now seeds `np.random` before each rollout,
so all methods see matched teleport randomness. The earlier mean-action
three-seed result is superseded; under controlled teleport randomness, the
mean-action executor ties Bellman-polished, stochastic-TRL-polished, and full
Bellman exactly at mean success `0.413`.

Additional correction: the evaluator now uses disjoint episode seed ranges,
`seed * 1_000_000 + task_id * 10_000 + episode_idx`. Earlier five-seed
PointMaze tables used overlapping rollout randomness across nominal seed ids
and are now historical diagnostics only.

Current paper-facing paired 50-episode, five-seed result:

- Summary: `results/pointmaze_graph_summary.md`.
- File:
  - `results/pointmaze_graph_persistent_waypoint_matched_independent_seeded_paired50.csv`
- Env: `pointmaze-teleport-navigate-v0`.
- Graph: `cell_size=1.0`, `action_bin_levels=3`, `554` cells, `4197`
  state-action bins.
- Discount: `gamma=0.995`.
- Matched budget: `ceil(log2(554)) + 1 = 11` sweeps.
- Executor: persistent graph waypoint controller.

| method | sweeps | seed successes | mean success |
| --- | ---: | --- | ---: |
| Bellman matched | 11 | 0.220, 0.200, 0.200, 0.264, 0.196 | 0.216 |
| Stochastic TRL matched | 11 | 0.392, 0.324, 0.324, 0.388, 0.348 | 0.355 |
| Bellman full | 220 | 0.452, 0.360, 0.360, 0.436, 0.400 | 0.402 |

Interpretation: with the same 11-sweep planning budget, the stochastic
transitive backup improves success by about 64% relative to matched Bellman
and approaches the 220-sweep Bellman reference. This is the cleanest current
continuous evidence for the horizon-efficiency claim, but the wording should
be more conservative than the superseded overlapping-seed table.

Added `scripts/sweep_pointmaze_graph.py` for incremental graph sweeps. Follow-up
executor and polish-budget checks:

- Persistent waypoint plus 40 Bellman polish gives stochastic TRL mean `0.480`
  versus Bellman-polished `0.447`; full Bellman remains `0.520`. This polish
  check predates the independent-seed fix and is diagnostic only.
- `q_transition` scored `0.24` for stochastic TRL polished on seed 0 and is
  slower than waypoint execution.
- A waypoint controller screen over action scale, gain, and smoothing did not
  produce a meaningful improvement because actions are usually saturated.
- Corrected independent-seed executor screens did not improve the headline
  PointMaze result. Persistent waypoints remained best among the tested
  executors with a three-seed, ten-episode mean `0.333`, versus mean edge
  action `0.313`, blend75 `0.327`, blend50 `0.300`, blend25 `0.233`,
  value-waypoint `0.080`, and value-waypoint-hybrid `0.053`. Files:
  `results/pointmaze_graph_waypoint_executor_independent_seed_screen.csv`,
  `results/pointmaze_graph_blend_executor_independent_seed_screen.csv`, and
  `results/pointmaze_graph_persistent_params_independent_seed_screen.csv`.
- A 15-sweep planning-budget check did not improve stochastic TRL beyond the
  11-sweep result (`0.464` mean), while Bellman matched improved only to
  `0.264`; file:
  `results/pointmaze_graph_persistent_waypoint_iters15_seeded_paired10.csv`.
  This also predates the independent-seed fix.
- A graph-resolution screen over `cell_size in {0.75, 1.0, 1.25, 1.5}` did not
  improve the result. File: `results/pointmaze_graph_resolution_screen.csv`.
  In a two-seed, five-episode screen, stochastic TRL scored `0.260`, `0.520`,
  `0.300`, and `0.180` respectively; the current `cell_size=1.0` setting
  remained best. Treat this as a tuning diagnostic because it predates the
  independent-seed fix.
- Earlier KNN, transition-value, gain, and smoothing sweeps are now treated as
  negative executor checks, not as main claims, because they predate the
  controlled teleport seeding fix.

## Paper-Facing Artifacts

Added `scripts/generate_paper_artifacts.py` to regenerate compact tables and
figures from the current CSV results.

Added a 2D stochastic grid-shortcut benchmark:

- Environment: `sto_trl.envs.stochastic_grid_shortcut`.
- Runner: `scripts/run_grid_shortcut.py`.
- Raw result: `results/grid_shortcut_2d_5seed.csv`.
- Summary: `results/grid_shortcut_2d_summary.md`.
- Paper table: `results/paper_tables/grid_shortcut_2d.csv`.
- Figure: `results/figures/grid_shortcut_success.svg`.
- Grids: `8x4`, `16x4`, `16x8`; safe path lengths `31`, `63`, `127`.
- Portal probabilities: `0.02`, `0.05`; five seeds.
- Matched sweeps: `6`, `7`, `8`.

Aggregate result: stochastic TRL reaches success `1.000`, safe-action rate
`1.000`, portal-action rate `0.000`, and regret `0.000`, matching full Bellman.
Matched Bellman and MC-positive choose the risky portal with portal-action rate
`1.000` and succeed only at the portal rate (`0.036` mean success). MC-all-goals
is partially calibrated but loses long-horizon performance at the largest grid,
with aggregate success `0.600`.

Added a reduced realized-TRL diagnostic for the 2D grid benchmark:

- Raw result: `results/grid_shortcut_realized_diagnostic.csv`.
- Paper table: `results/paper_tables/grid_realized_diagnostic.csv`.
- Grids: `8x4`, `16x4`; path lengths `31`, `63`; portal probability `0.05`;
  three seeds.
- Raw realized TRL, log realized TRL, MC-positive, and matched Bellman all
  choose the risky portal and succeed at `0.051`.
- Stochastic TRL and full Bellman choose the safe corridor and succeed at
  `1.000`.

Added a planning-budget curve on the hardest 2D grid:

- Raw result: `results/grid_budget_curve_16x8_p005_5seed.csv`.
- Paper table: `results/paper_tables/grid_budget_curve.csv`.
- Figure: `results/figures/grid_budget_curve.svg`.
- Setting: `16x8`, safe path length `127`, portal probability `0.05`, five
  seeds.
- Bellman remains at portal-rate success `0.048` through `120` sweeps and
  first reaches success `1.000` at `126` sweeps.
- Stochastic TRL remains at portal-rate success through `6` sweeps and reaches
  success `1.000` at `7` sweeps; the main matched benchmark used a
  conservative `8`-sweep budget for this grid.

Added `STOCHASTIC_TRL_PAPER_DRAFT.md`, a manuscript skeleton with:

- Working title and abstract draft.
- Core algorithm definition: stochastic Bellman calibration plus transitive
  reachability composition.
- Proposition candidates for lower-bound preservation, logarithmic propagation,
  and risky-shortcut calibration.
- Current result tables and exact source-file pointers.
- Explicit non-claims and next experiments.

Generated outputs:

- `results/paper_artifacts.md`
- `results/main_claim_verification.md`
- `results/paper_tables/main_hard_task_results.csv`
- `results/paper_tables/tabular_l128.csv`
- `results/paper_tables/tabular_safe_horizon.csv`
- `results/paper_tables/tabular_risky_aggregate.csv`
- `results/paper_tables/grid_shortcut_2d.csv`
- `results/paper_tables/grid_realized_diagnostic.csv`
- `results/paper_tables/grid_budget_curve.csv`
- `results/paper_tables/pointmaze_topology_5seed.csv`
- `results/paper_tables/pointmaze_topology_stitch_5seed.csv`
- `results/paper_tables/pointmaze_topology_stitch_support_baseline_5seed.csv`
- `results/paper_tables/pointmaze_topology_paired_stats.csv`
- `results/paper_tables/pointmaze_topology_stitch_paired_stats.csv`
- `results/paper_tables/pointmaze_graph_5seed.csv`
- `results/paper_tables/pointmaze_graph_paired_stats.csv`
- `results/paper_tables/pointmaze_graph_task_deltas.csv`
- `results/paper_tables/antmaze_bodyk16_multiseed_ep5.csv`
- `results/paper_tables/antmaze_bodyk16_ep20_seed012.csv`
- `results/paper_tables/antmaze_budget_ep3_seed0.csv`
- `results/paper_tables/antmaze_stitch_executor_ablation_ep3_seed0.csv`
- `results/figures/tabular_l128_safe_success.svg`
- `results/figures/tabular_horizon_success.svg`
- `results/figures/grid_shortcut_success.svg`
- `results/figures/grid_budget_curve.svg`
- `results/figures/pointmaze_graph_success.svg`
- `results/figures/antmaze_bodyk16_multiseed.svg`
- `results/figures/antmaze_budget.svg`

Added `scripts/verify_main_claims.py`, which recomputes the main hard-task
claims from raw CSVs and writes `results/main_claim_verification.md`. Current
status is `PASS`: all four hard-task rows use 6 matched sweeps versus 180 full
sweeps, stochastic TRL is at least 0.90 success, matched Bellman is at most
0.40 success, the improvement is at least 0.50, and stochastic TRL exactly
matches full Bellman. The verifier also checks the deterministic-support
ablation: support TRL reaches `0.449`, improves over matched Bellman but
remains at least `0.40` below stochastic TRL.

Updated `STOCHASTIC_TRL_PAPER_DRAFT.md` with a clearer relation to the
deterministic TRL paper: original TRL composes deterministic/in-trajectory
reachability products, while stochastic TRL composes calibrated stochastic
hitting probabilities after a Bellman expectation backup. The new support-TRL
ablation on PointMaze stitch is the empirical separator for this distinction:
deterministic support composition reaches `0.449`, while calibrated stochastic
TRL reaches `0.901`.

The report highlights the two strongest current claims:

- In the `L=128` safe-optimal stochastic shortcut cases, stochastic TRL reaches
  success `1.000` with zero risky-action rate, while matched Bellman and
  MC-positive baselines choose the risky shortcut.
- Across the corrected five-seed horizon sweep `L in {16, 32, 64, 128}`,
  stochastic TRL maintains success `1.000` using logarithmic matched sweeps,
  while matched Bellman and MC-positive baselines keep selecting the risky
  shortcut.
- On the 2D stochastic grid-shortcut benchmark with path lengths `31`, `63`,
  and `127`, stochastic TRL reaches success `1.000` with matched sweeps
  `6`, `7`, and `8`, while matched Bellman and MC-positive succeed only at the
  risky portal rate.
- On the reduced 2D realized-TRL diagnostic, raw/log realized TRL also choose
  the risky portal and succeed only at the portal rate, while stochastic TRL
  reaches `1.000`.
- On the hardest 2D grid, stochastic TRL reaches success `1.000` after `7`
  sweeps, whereas Bellman first succeeds at `126` sweeps.
- On the empirical PointMaze graph with controlled teleport randomness and
  independent evaluation seed ranges, stochastic TRL reaches five-seed mean
  success `0.355` with the matched 11-sweep budget, versus `0.216` for matched
  Bellman and `0.402` for the 220-sweep full Bellman reference.
- The paired seed-level PointMaze improvement over matched Bellman is `+0.139`
  with 95% CI `[0.112, 0.166]`, positive for all five seeds, and recovers
  `0.750` of the matched-to-full Bellman gap.
- On the new PointMaze dataset-inferred topology diagnostic, stochastic TRL
  reaches mean success `0.901` with the matched 6-sweep budget and exactly
  matches the 180-sweep Bellman reference; matched Bellman reaches `0.343`.
  Free cells and stochastic teleport jumps are inferred from the offline
  dataset, but the local controller is still a topology-level scaffold rather
  than a fully neural policy.
- On the matching `pointmaze-teleport-stitch-v0` dataset-inferred topology
  diagnostic with five evaluation seeds and 50 episodes per task, stochastic
  TRL again reaches `0.901` and matches full Bellman; matched Bellman reaches
  `0.343`. The paired seed-level improvement over matched Bellman is `+0.558`
  with 95% CI `[0.501, 0.614]`.
- Deterministic-support TRL ablation on PointMaze stitch reaches `0.449` with
  the same 6-sweep budget. It treats every observed stochastic teleport outcome
  as reliable and remains far below stochastic TRL at `0.901`, which supports
  the claim that stochastic calibration matters beyond transitive composition.

## AntMaze Learned-Controller Diagnostic

Added `scripts/run_antmaze_bc_topology_planner.py` to isolate value propagation
from low-level Ant execution:

- The high-level planner infers a 45-state topology and stochastic teleport
  jumps from the offline `antmaze-teleport-navigate-v0` dataset.
- All compared methods share the same learned goal-conditioned BC executor.
- KNN/snippet execution was useful for debugging but too weak for a main claim.
- `xy`-only BC subgoals were also weak. Full future-observation goals were the
  first learned-controller setup that reliably executed long safe paths.

Current best fast run:

- Summary: `results/antmaze_bc_topology_summary.md`.
- Table: `results/paper_tables/antmaze_bc_topology_20k_ep3_seed0.csv`.
- Raw CSV: `results/antmaze_bc_topology_fullgoal_20k_ep3_seed0.csv`.
- Progress log:
  `results/antmaze_bc_topology_fullgoal_20k_ep3_seed0_progress.jsonl`.
- Controller: deterministic BC MLP, hidden dims `256,256,256`, full
  future-observation goals.
- Training: 20k optimizer steps, batch size 2048, JAX on `cuda:0`.
- Evaluation: seed 0, 3 episodes per task.

| method | sweeps | mean success | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bellman matched | 6 | 0.333 | 0.000 | 0.000 | 0.333 | 0.667 | 0.667 |
| Stochastic TRL | 6 | 0.933 | 1.000 | 1.000 | 0.667 | 1.000 | 1.000 |

A separate single-episode full-Bellman check with the same 20k-step controller
gave Bellman matched `0.200`, stochastic TRL `0.800`, and Bellman full `0.800`.
This supports the interpretation that stochastic TRL is recovering the
full-Bellman high-level route under a small matched budget, while remaining
misses come from the learned executor.

Follow-up eval-only navigate check with the saved 50k-step controller and
16-candidate body-nearest waypoint goals:

- Policy checkpoint: `results/policies/antmaze_navigate_fullgoal_bc_50k.pkl`.
- Stochastic/full raw CSV:
  `results/antmaze_bc_topology_navigate_fullgoal_50k_ep10_seed0_bodyk16_sto_full.csv`.
- Matched raw CSV:
  `results/antmaze_bc_topology_navigate_fullgoal_50k_ep10_seed0_bodyk16_matched.csv`.

| method | sweeps | mean success | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bellman matched | 6 | 0.380 | 0.000 | 0.500 | 0.600 | 0.400 | 0.400 |
| Stochastic TRL | 6 | 0.940 | 1.000 | 1.000 | 0.900 | 0.900 | 0.900 |
| Bellman full | 180 | 0.940 | 1.000 | 1.000 | 0.900 | 0.900 | 0.900 |

This fixes the earlier navigate task-4 failure in a larger 10-episode screen.
The residual misses are shared by stochastic TRL and full Bellman, so they are
still best understood as executor noise rather than a high-level
value-propagation gap.

Compact multi-seed navigate check:

- Raw CSV:
  `results/antmaze_bc_topology_navigate_fullgoal_50k_ep5_seed012_bodyk16.csv`.
- Figure:
  `results/figures/antmaze_bodyk16_multiseed.svg`.
- Setting: same saved 50k-step controller, 16 body-nearest candidate waypoint
  goals, three evaluation seeds, five episodes per task.

| method | sweeps | mean success | seed std | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bellman matched | 6 | 0.347 | 0.068 | 0.000 | 0.467 | 0.467 | 0.533 | 0.267 |
| Stochastic TRL | 6 | 0.893 | 0.019 | 0.867 | 0.933 | 0.933 | 0.933 | 0.800 |
| Bellman full | 180 | 0.893 | 0.019 | 0.867 | 0.933 | 0.933 | 0.933 | 0.800 |

Important context: the official local TRL training script defaults to
`offline_steps=1000000` in `external/trl/main.py`, so 20k BC updates are still a
fast smoke budget rather than a paper-scale controller training budget.

## Teleport Stitch Screens

Added faster evaluation support to `scripts/run_antmaze_bc_topology_planner.py`:

- JIT-compiled the per-step BC actor inference.
- Added `--save-policy` and `--load-policy` for eval-only variants.
- Added `--goal-candidates-per-state` and `--goal-candidate-mode body_nearest`
  for full-observation waypoint goals in AntMaze.
- Added `--eval-action-repeat` for optional action reuse during fast screens.
- Added `--max-episode-steps` for intentionally truncated diagnostic screens.

Eval-only timing note: with a loaded AntMaze stitch 20k full-goal BC policy and
the exact strong executor setting (`--waypoint-lookahead 1`, 16 body-nearest
goal candidates), a one-episode-per-task `sto_trl_matched` smoke took about
`10.1s` under `JAX_PLATFORMS=cpu` versus `17.0s` under `JAX_PLATFORMS=cuda`,
with `1.0` success in both cases. For compact eval-only screens, CPU JAX is the
preferred default; reserve GPU for BC training or larger batched neural runs.

Downloaded and cached:

- `pointmaze-teleport-stitch-v0`
- `antmaze-teleport-stitch-v0`

PointMaze stitch screen:

- Summary: `results/teleport_stitch_screen_summary.md`.
- Raw CSV: `results/pointmaze_topology_stitch_seed0_ep10.csv`.
- Setting: dataset-inferred topology, seed 0, 10 episodes per task.

| method | sweeps | mean success |
| --- | ---: | ---: |
| Bellman matched | 6 | 0.380 |
| Stochastic TRL | 6 | 0.980 |
| Bellman full | 180 | 0.980 |

AntMaze stitch screen:

- Raw CSV: `results/antmaze_bc_topology_stitch_fullgoal_20k_seed0.csv`.
- Policy checkpoint: `results/policies/antmaze_stitch_fullgoal_bc_20k.pkl`.
- Setting: shared full-observation BC controller, 20k optimizer steps, seed 0,
  one episode per task.

| method | sweeps | mean success |
| --- | ---: | ---: |
| Bellman matched | 6 | 0.200 |
| Stochastic TRL | 6 | 1.000 |
| Bellman full | 180 | 1.000 |

Larger repeated AntMaze stitch screen:

- Stochastic/full raw CSV:
  `results/antmaze_bc_topology_stitch_fullgoal_20k_ep10_seed0_bodyk16_sto_full.csv`.
- Matched raw CSV:
  `results/antmaze_bc_topology_stitch_fullgoal_20k_ep10_seed0_bodyk16_matched.csv`.
- Setting: same saved 20k-step full-observation BC controller, 16 candidate
  dataset observations per waypoint, selected by nearest normalized non-xy body
  state, seed 0, 10 episodes per task.

| method | sweeps | mean success |
| --- | ---: | ---: |
| Bellman matched | 6 | 0.380 |
| Stochastic TRL | 6 | 0.940 |
| Bellman full | 180 | 0.960 |

Interpretation: stitch strengthens the long-horizon performance story. Under
the same small 6-sweep planning budget, stochastic TRL matches the 180-sweep
Bellman reference up to one extra executor miss on AntMaze stitch, while
matched Bellman remains too local.

Compact multi-seed stitch check:

- Raw CSV:
  `results/antmaze_bc_topology_stitch_fullgoal_20k_ep5_seed012_bodyk16.csv`.
- Setting: same saved 20k-step controller, 16 body-nearest candidate waypoint
  goals, three evaluation seeds, five episodes per task.

| method | sweeps | mean success | seed std | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bellman matched | 6 | 0.333 | 0.075 | 0.000 | 0.400 | 0.467 | 0.533 | 0.267 |
| Stochastic TRL | 6 | 0.907 | 0.038 | 0.867 | 0.933 | 1.000 | 0.800 | 0.933 |
| Bellman full | 180 | 0.907 | 0.038 | 0.867 | 0.933 | 1.000 | 0.800 | 0.933 |

20-episode AntMaze hard-task check, three evaluation seeds:

- Navigate raw CSV:
  `results/antmaze_bc_topology_navigate_fullgoal_50k_ep20_seed012_bodyk16_cpu.csv`.
- Stitch raw CSV:
  `results/antmaze_bc_topology_stitch_fullgoal_20k_ep20_seed012_bodyk16_cpu.csv`.
- Paper table:
  `results/paper_tables/antmaze_bodyk16_ep20_seed012.csv`.
- Paired stats table:
  `results/paper_tables/antmaze_bodyk16_ep20_seed012_paired_stats.csv`.
- Setting: CPU eval-only run with saved task-specific controller, 16
  body-nearest candidate waypoint goals, evaluation seeds 0, 1, and 2, 20
  episodes per task, full horizon and `eval_action_repeat=1`.

| env | Bellman matched | Stochastic TRL | Bellman full |
| --- | ---: | ---: | ---: |
| Navigate | 0.310 | 0.947 | 0.947 |
| Stitch | 0.317 | 0.960 | 0.960 |

This higher-episode multi-eval-seed check strengthens the AntMaze claim:
stochastic TRL exactly matches the 180-sweep Bellman reference on both hard
tasks while matched Bellman remains near 0.31 under the same executor. The
paired improvement over matched Bellman is +0.637 on navigate with 95% CI
[0.525, 0.749] and +0.643 on stitch with 95% CI [0.494, 0.793].

Independent controller-seed AntMaze screens:

- Navigate train-only summary:
  `results/antmaze_navigate_fullgoal_bc_50k_seed1_trainonly.json`.
- Navigate train-only progress log:
  `results/antmaze_navigate_fullgoal_bc_50k_seed1_trainonly.progress.jsonl`.
- Navigate saved policy:
  `results/policies/antmaze_navigate_fullgoal_bc_50k_seed1.pkl`.
- Navigate raw eval CSV:
  `results/antmaze_bc_topology_navigate_fullgoal_50k_bcseed1_ep20_seed012_bodyk16_cpu.csv`.
- Stitch train-only summary:
  `results/antmaze_stitch_fullgoal_bc_20k_seed1_trainonly.json`.
- Stitch train-only progress log:
  `results/antmaze_stitch_fullgoal_bc_20k_seed1_trainonly.progress.jsonl`.
- Stitch saved policy:
  `results/policies/antmaze_stitch_fullgoal_bc_20k_seed1.pkl`.
- Stitch raw eval CSV:
  `results/antmaze_bc_topology_stitch_fullgoal_20k_bcseed1_ep20_seed012_bodyk16_cpu.csv`.
- Paper table:
  `results/paper_tables/antmaze_bcseed1_ep20_seed012.csv`.
- Setting: independently trained full-observation BC controllers with
  `bc_seed=1`, 16 body-nearest candidate waypoint goals, evaluation seeds 0,
  1, and 2, 20 episodes per task. Navigate uses 50k BC optimizer steps; stitch
  uses 20k.

| env | Bellman matched | Stochastic TRL | Bellman full |
| --- | ---: | ---: | ---: |
| Navigate | 0.320 | 0.933 | 0.933 |
| Stitch | 0.323 | 0.950 | 0.950 |

These independent controller-seed robustness checks use the same 20-episode
protocol as the headline AntMaze table and show that the AntMaze result does
not depend only on the original seed-0 BC controllers.

Stitch controller-seed aggregate:

- Seed-2 train-only summary:
  `results/antmaze_stitch_fullgoal_bc_20k_seed2_trainonly.json`.
- Seed-2 train-only progress log:
  `results/antmaze_stitch_fullgoal_bc_20k_seed2_trainonly.progress.jsonl`.
- Seed-2 saved policy:
  `results/policies/antmaze_stitch_fullgoal_bc_20k_seed2.pkl`.
- Seed-2 raw eval CSV:
  `results/antmaze_bc_topology_stitch_fullgoal_20k_bcseed2_ep20_seed012_bodyk16_cpu.csv`.
- Aggregate paper table:
  `results/paper_tables/antmaze_stitch_controller_seeds_ep20_seed012.csv`.

| controller seed | Bellman matched | Stochastic TRL | Bellman full |
| ---: | ---: | ---: | ---: |
| 0 | 0.317 | 0.960 | 0.960 |
| 1 | 0.323 | 0.950 | 0.950 |
| 2 | 0.323 | 0.963 | 0.967 |

Across three stitch controller seeds, stochastic TRL remains above 0.95
success with the 6-sweep matched budget; matched Bellman remains near 0.32.
Seed 2 is a near-match to full Bellman, trailing by 0.003 absolute success.

Planning-budget AntMaze screens:

- Navigate raw CSV:
  `results/antmaze_bc_topology_navigate_budget_ep3_seed0_bodyk16.csv`.
- Stitch raw CSV:
  `results/antmaze_bc_topology_stitch_budget_ep3_seed0_bodyk16.csv`.
- Navigate summary:
  `results/antmaze_bc_topology_navigate_budget_ep3_seed0_bodyk16.json`.
- Stitch summary:
  `results/antmaze_bc_topology_stitch_budget_ep3_seed0_bodyk16.json`.
- Figure:
  `results/figures/antmaze_budget.svg`.
- Setting: saved task-specific controller, 16 body-nearest candidate waypoint
  goals, seed 0, three episodes per task.

| env | Bellman 6 | Stochastic TRL 6 | Bellman 12 | Bellman 24 | Bellman 45 | Bellman full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Navigate | 0.333 | 1.000 | 0.667 | 0.933 | 1.000 | 1.000 |
| Stitch | 0.333 | 1.000 | 0.667 | 0.933 | 1.000 | 1.000 |

This is the clearest budget-efficiency screen so far: stochastic TRL reaches
full-Bellman-level success at the 6-sweep matched budget on both AntMaze
navigate and stitch, while ordinary Bellman needs roughly 45 sweeps.

Executor ablation on AntMaze stitch:

- Nearest-xy raw CSV:
  `results/antmaze_bc_topology_stitch_fullgoal_20k_ep3_seed0_nearestxy_sto_full.csv`.
- Body-nearest raw CSV:
  `results/antmaze_bc_topology_stitch_fullgoal_20k_ep3_seed0_bodyk16_sto_full.csv`.
- Setting: saved 20k-step controller, seed 0, three episodes per task.

| executor | Stochastic TRL | Bellman full |
| --- | ---: | ---: |
| nearest-xy waypoint observation | 0.867 | 0.867 |
| 16 body-nearest waypoint candidates | 1.000 | 1.000 |

Both stochastic TRL and full Bellman move together under the executor change.
This isolates the nearest-xy failures as local-controller/goal-state mismatch
rather than high-level stochastic value-propagation errors.
