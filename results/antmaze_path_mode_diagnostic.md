# AntMaze Path-Mode Diagnostic

Date: 2026-06-19.

## Purpose

Check whether the remaining AntMaze hard-task misses are caused by the
high-level stochastic value table or by waypoint path following. This diagnostic
uses the saved strong MSE BC executors and changes only the topology-agent path
mode from the default greedy replanning to persistent path tracking.

## Runs

Shared settings:

- JAX policy backend.
- Body-nearest waypoint candidates with `--goal-candidates-per-state 16`.
- `--waypoint-lookahead 1`.
- `--eval-action-repeat 1`.
- AntMaze hard tasks `4,5`.

| env | path mode | eval seeds | episodes/task | Bellman matched | Stochastic TRL | Bellman full | task4 sto | task5 sto | source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| antmaze-teleport-navigate-v0 | persistent | 0,1,2 | 10 | 0.000 | 0.917 | 0.917 | 0.967 | 0.867 | `results/antmaze_navigate_persistent_path_ep10_seed012_task45_summary.csv` |
| antmaze-teleport-stitch-v0 | persistent | 0 | 5 | 0.000 | 0.800 | 0.800 | 1.000 | 0.600 | `results/antmaze_stitch_persistent_path_ep5_seed0_task45.csv` |

## Interpretation

Persistent path tracking is a useful diagnostic but not a better default. On
AntMaze navigate hard tasks 4 and 5, seed 0 reached 1.000 stochastic-TRL
success, but the three-evaluation-seed aggregate is 0.917. Matched Bellman
falls to 0.000 under persistent pathing, so the value-propagation separation is
large, but the existing greedy 20-episode navigate hard-task aggregate remains
slightly stronger overall on tasks 4 and 5.

The same knob is not a universal improvement. On AntMaze stitch tasks 4 and 5,
persistent path tracking reduces stochastic TRL and full Bellman to 0.800 by
hurting task 5 execution. Keep greedy pathing as the default paper setting for
cross-task AntMaze tables, and keep persistent pathing as a non-promoted
executor/path-following ablation.
