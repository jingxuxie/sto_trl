# PointMaze Empirical Graph Planner Summary

Environment: `pointmaze-teleport-navigate-v0`.

Graph construction:

- `cell_size=1.0`
- `action_bin_levels=3`
- `gamma=0.995`
- `554` graph cells
- `4197` state-action bins
- matched horizon-efficient budget: `ceil(log2(554)) + 1 = 11` sweeps
- full Bellman reference: `220` sweeps

## Topology-Level High-Success Diagnostic

Before improving the learned/empirical graph executor, I added a topology-level
diagnostic that infers coarse free cells and stochastic teleport jumps from the
offline dataset, then solves the induced stochastic cell MDP. This is not a
fully neural result, but it tests whether the stochastic transitive backup can
solve the hard teleport tasks when low-level execution is reliable.

Files:

- `results/pointmaze_topology_dataset_5seed_ep50.csv`
- `results/pointmaze_topology_dataset_5seed_ep50_summary.json`
- `results/paper_tables/pointmaze_topology_5seed.csv`
- `results/paper_tables/pointmaze_topology_paired_stats.csv`

Five-seed, 50-episode-per-task result:

| method | sweeps | mean success | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bellman matched | 6 | 0.343 | 0.000 | 0.460 | 0.440 | 0.464 | 0.352 |
| Stochastic TRL | 6 | 0.901 | 0.892 | 0.912 | 0.876 | 0.908 | 0.916 |
| Bellman full | 180 | 0.901 | 0.892 | 0.912 | 0.876 | 0.908 | 0.916 |

Paired seed-level improvement of stochastic TRL over matched Bellman:
`+0.558`, 95% CI `[0.501, 0.614]`, positive for all five seeds.

## Evaluation Fix

The teleport environment uses global NumPy randomness when selecting the
teleport exit. `scripts/run_pointmaze_graph_planner.py` now calls
`np.random.seed(episode_seed)` before every rollout, so all methods see matched
teleport randomness under the same evaluation seed.

A second evaluator fix changed the episode seed schedule from overlapping
seed ranges to `seed * 1_000_000 + task_id * 10_000 + episode_idx`. Earlier
five-seed tables used mostly overlapping rollout randomness across seed ids
and are now treated as superseded diagnostics, not paper-facing results.

This supersedes the earlier mean-action three-seed table. Under controlled
teleport randomness, mean-action execution ties all three planners exactly:

- File: `results/pointmaze_graph_mean_seeded_paired10.csv`.
- Bellman polished: `0.40, 0.40, 0.44`, mean `0.413`.
- Stochastic TRL polished: `0.40, 0.40, 0.44`, mean `0.413`.
- Bellman full: `0.40, 0.40, 0.44`, mean `0.413`.

## Main Seeded Result

Action execution mode: persistent graph waypoint controller.

Files:

- `results/pointmaze_graph_persistent_waypoint_matched_independent_seeded_paired50.csv`
- `results/paper_tables/pointmaze_graph_paired_stats.csv`
- `results/paper_tables/pointmaze_graph_task_deltas.csv`

Each seed uses 50 paired episodes per task with disjoint episode seed ranges.

| method | sweeps | seed successes | mean | std |
| --- | ---: | --- | ---: | ---: |
| Bellman matched | 11 | 0.220, 0.200, 0.200, 0.264, 0.196 | 0.216 | 0.025 |
| Stochastic TRL matched | 11 | 0.392, 0.324, 0.324, 0.388, 0.348 | 0.355 | 0.030 |
| Bellman full | 220 | 0.452, 0.360, 0.360, 0.436, 0.400 | 0.402 | 0.038 |

Task mean success across seeds:

| method | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bellman matched | 0.000 | 0.000 | 0.388 | 0.428 | 0.264 |
| Stochastic TRL matched | 0.384 | 0.228 | 0.392 | 0.428 | 0.344 |
| Bellman full | 0.436 | 0.420 | 0.392 | 0.428 | 0.332 |

Interpretation: with the same 11-sweep planning budget, the stochastic
transitive backup improves empirical success by about 64% relative to Bellman
backup (`0.355` versus `0.216`) and recovers most of the 220-sweep Bellman
reference (`0.402`). This is the cleanest current continuous-control evidence
for the horizon-efficiency claim, but it is weaker than the earlier overlapping
seed estimate and should be worded accordingly.

Paired seed-level statistics:

| comparison | mean diff | 95% CI | all seed diffs positive | full-gap recovery |
| --- | ---: | ---: | --- | ---: |
| Stochastic TRL - Bellman matched | 0.139 | [0.112, 0.166] | True | 0.750 |
| Bellman full - Stochastic TRL | 0.046 | [0.033, 0.059] | True | 0.250 |

Task deltas explain where the gain comes from: stochastic TRL improves over
matched Bellman by `0.384` on task 1 and `0.228` on task 2, where matched
Bellman has zero success. Tasks 3 and 4 are already reachable for matched
Bellman, with deltas `0.004` and `0.000`.

## Stitch Empirical Graph Boundary

I reran the same independent-seed empirical graph protocol on
`pointmaze-teleport-stitch-v0` with 50 paired episodes per task and five
evaluation seeds. This is a less privileged check than the topology scaffold:
it builds a 565-state, 4598-edge graph from discretized dataset observations
and uses the persistent graph waypoint executor.

Files:

- `results/pointmaze_stitch_graph_persistent_waypoint_matched_independent_seeded_paired50.csv`
- `results/pointmaze_stitch_graph_persistent_waypoint_matched_independent_seeded_paired50.json`

Result:

| method | sweeps | seed successes | mean | std |
| --- | ---: | --- | ---: | ---: |
| Bellman matched | 11 | 0.160, 0.140, 0.168, 0.196, 0.176 | 0.168 | 0.018 |
| Stochastic TRL matched | 11 | 0.164, 0.132, 0.116, 0.156, 0.152 | 0.144 | 0.018 |
| Bellman full | 220 | 0.076, 0.068, 0.072, 0.080, 0.092 | 0.078 | 0.008 |

Task means:

| method | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bellman matched | 0.000 | 0.000 | 0.276 | 0.328 | 0.236 |
| Stochastic TRL matched | 0.000 | 0.252 | 0.276 | 0.000 | 0.192 |
| Bellman full | 0.000 | 0.064 | 0.276 | 0.000 | 0.048 |

Interpretation: this is a negative less-privileged graph/executor boundary, not
a positive paper result. Stochastic TRL recovers task 2, but the same graph
executor collapses task 4 under both stochastic TRL and 220-sweep Bellman. Since
the long-sweep Bellman reference is also weak, the bottleneck is the empirical
graph/controller path on stitch rather than the verified stochastic-TRL topology
planner. Do not promote this screen as evidence against the high-success
topology or learned-controller stitch results.

### Stitch Task-4 Executor Sweep

To check whether the full-Bellman stitch failure was just a poor persistent
waypoint executor, I swept the available graph executors on task 4 only with
the 220-sweep Bellman policy, three evaluation seeds, and 20 episodes per seed.

Files:

- `results/pointmaze_stitch_graph_task4_executor_screen_ep20_seed012.csv`
- `results/pointmaze_stitch_graph_task4_executor_screen_ep20_seed012.json`

| executor | mean success | seed successes |
| --- | ---: | --- |
| transition_value | 0.133 | 0.200, 0.100, 0.100 |
| mean | 0.067 | 0.100, 0.000, 0.100 |
| knn_transition | 0.017 | 0.050, 0.000, 0.000 |
| center_delta / hybrid / nearest_transition / value_waypoint / persistent_waypoint / blends / q_transition variants | 0.000 | all zero |

No screened executor came close to making the 220-sweep graph policy reliable
on stitch task 4. The best mode, `transition_value`, is still too weak for a
paper claim. The next useful direction is a different representation or learned
waypoint executor for the empirical graph, not further small changes to these
hand-coded graph action decoders.

### Learned BC Graph Executor Check

I added `bc_persistent_waypoint` to
`scripts/run_pointmaze_graph_planner.py`, which sends the empirical graph's
persistent waypoint centers to the saved PointMaze full-goal BC policy instead
of using a proportional controller. On a single-seed stitch task-4 screen with
the 220-sweep Bellman graph policy and 10 episodes, this did not rescue the
failure:

File: `results/pointmaze_stitch_graph_task4_bc_executor_ep10_seed0.csv`.

| executor | task4 success |
| --- | ---: |
| `bc_persistent_waypoint` | 0.000 |
| `persistent_waypoint` | 0.000 |
| `transition_value` | 0.100 |

This is not a paper-facing estimate, but it isolates the current bottleneck:
the learned local PointMaze controller is not enough when the empirical graph
selects the wrong high-level path. The positive PointMaze stitch result should
therefore continue to be framed as learned waypoint execution on the topology
planner, while the empirical graph remains a negative boundary.

### Graph Evaluation Speed Notes

The empirical graph runner now records `eval_seconds`, average task lengths,
and optional `--profile-eval` timing fields. It also supports
`--max-episode-steps` for capped debug rollouts and caches solved graph Q
tables under `results/cache` unless `--no-cache` is passed.

A one-episode stitch task-4 profile with `transition_value` and full Bellman
showed the main fixed cost:

- First full-Bellman graph solve: `q_solve_seconds=11.09`.
- Cached full-Bellman graph solve: `q_solve_seconds=0.03`.
- Failed 1000-step rollout: `eval_seconds=0.39`.
- Per-step timing for `transition_value`: action selection `0.30 ms`, env step
  `0.08 ms`.
- Per-step timing for `bc_persistent_waypoint`: action selection `0.05 ms`, env
  step `0.09 ms`.

For fast iteration, use one seed, `--task-ids` for the hard slice, low
`--episodes`, cached Q tables, and `--profile-eval`. For final paper numbers,
keep full-horizon evaluation and scale episodes/seeds only after the single-seed
hard slice is positive.

## Superseded Seeded Result

Files:

- `results/pointmaze_graph_persistent_waypoint_matched_seeded_paired10.csv`
- `results/pointmaze_graph_persistent_waypoint_matched_seeded_paired10_seed34.csv`

These older files reported stochastic TRL mean `0.464`, matched Bellman mean
`0.224`, and full Bellman mean `0.520`, but their seed ranges overlapped
heavily across nominal seed ids. Keep them only as historical diagnostics.

## Secondary Seeded Result

File: `results/pointmaze_graph_persistent_waypoint_seeded_paired10.csv`.

With 40 Bellman polish sweeps after the matched phase. This was run before
the independent-seed fix, so treat it as a diagnostic, not a paper-facing
table:

| method | seed successes | mean | std |
| --- | --- | ---: | ---: |
| Bellman polished | 0.44, 0.42, 0.48 | 0.447 | 0.025 |
| Stochastic TRL polished | 0.46, 0.46, 0.52 | 0.480 | 0.028 |
| Bellman full | 0.50, 0.50, 0.56 | 0.520 | 0.028 |

Interpretation: Bellman polish narrows the gap, but stochastic TRL still stays
ahead under the same polished budget.

## Negative Executor Ablations

- `q_transition` scored `0.24` for stochastic TRL polished on seed 0 and is
  slower than the waypoint executor.
- `value_waypoint_hybrid` preserved a stochastic advantage on seed 0 but had
  lower absolute success than persistent waypoints.
- Under the corrected independent-seed evaluator, a three-seed, ten-episode
  stochastic-TRL executor screen confirmed that persistent waypoints remain the
  best tested executor. File:
  `results/pointmaze_graph_waypoint_executor_independent_seed_screen.csv`.
  Means: persistent waypoint `0.333`, mean edge action `0.313`,
  value-waypoint `0.080`, value-waypoint-hybrid `0.053`.
- A generic blend between persistent waypoint tracking and mean edge actions
  also did not improve the result. File:
  `results/pointmaze_graph_blend_executor_independent_seed_screen.csv`.
  Means: persistent waypoint `0.333`, blend75 `0.327`, blend50 `0.300`,
  blend25 `0.233`, mean edge action `0.313`.
- A small controller screen over `action_scale in {0.12, 0.16, 0.20, 0.28}`,
  `action_gain in {0.8, 1.0, 1.2}`, and `action_smoothing in {0.0, 0.25}`
  did not identify a real improvement; most waypoint actions saturate, so the
  settings behaved identically in the two-seed, five-episode screen.
- A corrected independent-seed persistent-waypoint parameter screen over
  `action_scale in {0.12, 0.16, 0.20, 0.28, 0.40}` and
  `action_gain in {0.8, 1.0}` reached the same best mean `0.333`; lowering
  gain to `0.8` reduced the mean to `0.307`. File:
  `results/pointmaze_graph_persistent_params_independent_seed_screen.csv`.
- Increasing the matched planning budget from 11 to 15 sweeps did not improve
  stochastic TRL: `results/pointmaze_graph_persistent_waypoint_iters15_seeded_paired10.csv`
  gave stochastic TRL mean `0.464` under the old overlapping-seed evaluator.
  This remains a diagnostic only; rerun it with independent seeds before using
  it for any paper-facing planning-budget claim.
- A graph-resolution screen over `cell_size in {0.75, 1.0, 1.25, 1.5}` with
  two seeds and five paired episodes per task did not improve over the current
  setting. File: `results/pointmaze_graph_resolution_screen.csv`.
  Mean successes were:
  - `cell_size=0.75`: Bellman matched `0.240`, stochastic TRL `0.260`, full
    Bellman `0.300`.
  - `cell_size=1.0`: Bellman matched `0.280`, stochastic TRL `0.520`, full
    Bellman `0.560`.
  - `cell_size=1.25`: Bellman matched `0.180`, stochastic TRL `0.300`, full
    Bellman `0.220`.
  - `cell_size=1.5`: Bellman matched `0.040`, stochastic TRL `0.180`, full
    Bellman `0.240`.
  The original `cell_size=1.0` configuration remains the best screened graph
  resolution. This screen also predates the independent-seed fix and should be
  treated as a tuning diagnostic.
- Earlier KNN, transition-value, gain, and smoothing sweeps remain useful as
  negative checks, but their unseeded comparisons should not be used as main
  claims.
