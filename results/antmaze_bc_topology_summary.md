# AntMaze Learned-Controller Topology Summary

Environment: `antmaze-teleport-navigate-v0`.

This diagnostic separates the high-level stochastic TRL value function from the
local Ant controller. The high-level planner infers a 45-state topology and
teleport jump model from the offline dataset. A shared goal-conditioned BC
controller executes the planner subgoals for all methods.

## Controller Variants Tested

- KNN/snippet controller: useful for debugging, but not sufficient for the main
  claim. It could solve isolated short/teleport routes but failed the long safe
  stochastic TRL routes.
- Deterministic BC with `xy` goals: trained successfully by MSE, but often
  stalled or missed long Ant routes.
- Deterministic BC with full future-observation goals: substantially better
  and closer to the official TRL goal-conditioned actor interface.
- TRL-style GCFBC flow actor: integrated and runnable. After adding
  body-compatible waypoint candidates, the latest short screen reaches 0.600
  success on AntMaze navigate task 3 after 20k updates, but it remains far
  below the current MSE BC executor and is not main evidence.

## Current Best Learned-Controller Result

Source files:

- Runner: `scripts/run_antmaze_bc_topology_planner.py`
- CSV: `results/antmaze_bc_topology_fullgoal_20k_ep3_seed0.csv`
- Summary: `results/antmaze_bc_topology_fullgoal_20k_ep3_seed0.json`
- Progress log: `results/antmaze_bc_topology_fullgoal_20k_ep3_seed0_progress.jsonl`

Setting:

- Controller: deterministic goal-conditioned BC MLP.
- Goal representation: full future observation, not just `xy`.
- Training: 20k optimizer steps, batch size 2048, hidden dims `256,256,256`.
- Hardware: JAX on `cuda:0`.
- Planner budget: matched `ceil(log2(45)) = 6` sweeps.
- Evaluation: seed 0, 3 episodes per task, 5 tasks.

| method | sweeps | mean success | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bellman matched | 6 | 0.333 | 0.000 | 0.000 | 0.333 | 0.667 | 0.667 |
| Stochastic TRL | 6 | 0.933 | 1.000 | 1.000 | 0.667 | 1.000 | 1.000 |

Interpretation: with the same learned BC controller and the same matched
planning budget, stochastic TRL is much better than matched Bellman on AntMaze
teleport. The high-level stochastic TRL paths are also the same paths selected
by full Bellman on this inferred topology, so the remaining failures are mainly
controller/execution issues rather than value propagation issues.

## Single-Episode Full-Bellman Check

Source: `results/antmaze_bc_topology_fullgoal_20k_alltasks_seed0.csv`.

With the same 20k-step controller and one episode per task:

| method | mean success |
| --- | ---: |
| Bellman matched | 0.200 |
| Stochastic TRL | 0.800 |
| Bellman full | 0.800 |

This confirms that stochastic TRL matches the high-sweep Bellman plan under
the same learned executor in this smoke setting.

## Paper-Grade Current Tables

The preliminary 3-episode screen above is superseded by the verified paper
tables:

- Navigate: `results/paper_tables/antmaze_navigate_controller_seeds_ep20_seed012.csv`
- Stitch: `results/paper_tables/antmaze_stitch_controller_seeds_ep20_seed012.csv`
- Main hard-task rows: `results/paper_tables/main_hard_task_results.csv`

Both navigate and stitch now have three independently trained BC controller
seeds, three evaluation seeds per controller, and 20 episodes per task.
Stochastic TRL stays above 0.933 success across these controller seeds while
matched Bellman stays near 0.31-0.32.

## Claim Boundary

The AntMaze evidence is now paper-grade as a learned-controller topology
diagnostic, not as an end-to-end neural TRL replacement. The next high-impact
step is to add a learned value/transition module or a stronger standard
controller only if a short screen can plausibly preserve the 0.90+ success
regime.

## Learned-Transition Screen

Summary: `results/antmaze_learned_transition_summary.md`.

The runner now supports `--transition-model learned_softmax`, which fits a
tabular-softmax transition model from sampled high-level next states before
running the value updates. In the seed-0 20-episode hard-task screen with
20 sampled outcomes per high-level row:

| env | method | mean success on tasks 4,5 |
| --- | --- | ---: |
| navigate | Bellman matched | 0.325 |
| navigate | Stochastic TRL | 0.925 |
| navigate | Bellman full | 0.925 |
| stitch | Bellman matched | 0.325 |
| stitch | Stochastic TRL | 0.950 |
| stitch | Bellman full | 0.950 |

This is screen evidence that the stochastic-TRL signal survives a learned
high-level transition module under the learned AntMaze BC executor.
In a three-transition-seed robustness check at 10 episodes per hard task,
stochastic TRL stays at 0.950 on navigate and 1.000 on stitch, always matching
full Bellman.
