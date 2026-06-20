# AntMaze Joint-Action Q-Head Hard-Task Diagnostic

Date: 2026-06-20

## Question

The PointMaze q-head diagnostic showed that a joint-action critic can carry a
generated stochastic-TRL target. This AntMaze screen tests whether the same
raw-observation state/goal q-head representation is viable with the learned BC
executor on hard tasks 4 and 5.

## Setting

- Transition model: raw-observation MLP trained on dataset jump changes.
- Value model: raw-observation joint-action q-head MLP.
- Controller: saved full-goal BC with body-nearest `k=16` waypoint goals.
- Evaluation: seed 0, tasks 4 and 5, 3 episodes per task.
- Planner budget: matched 6 sweeps; full Bellman reference 180 sweeps.

## Results

| env | value steps | matched Bellman q-head | stochastic TRL q-head | full Bellman q-head | task4 sto | task5 sto |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| antmaze-teleport-navigate-v0 | 5000 | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
| antmaze-teleport-stitch-v0 | 5000 | 0.333 | 0.667 | 0.667 | 1.000 | 0.333 |
| antmaze-teleport-stitch-v0 | 10000 | n/a | 0.667 | 0.667 | 1.000 | 0.333 |
| antmaze-teleport-stitch-v0 prev-action q-head | 5000 | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 |

## Raw Files

- `results/antmaze_navigate_qhead_hard_task45_ep3_seed0.csv`
- `results/antmaze_stitch_qhead_hard_task45_ep3_seed0.csv`
- `results/antmaze_stitch_qhead_hard_task45_ep3_seed0_10k.csv`
- `results/antmaze_stitch_prev_qhead_hard_task45_ep3_seed0.csv`
- `results/antmaze_qhead_hard_task_verification.md`

## Interpretation

This is a mixed learned-critic result. The q-head representation is positive on
AntMaze navigate hard tasks: stochastic TRL matches the learned full-Bellman
q-head at 1.000 and beats the matched Bellman q-head at 0.333. On AntMaze
stitch, stochastic TRL again matches the learned full-Bellman q-head, but both
reach only 0.667 because task 5 remains fragile. Doubling q-head fitting from
5000 to 10000 steps does not improve stitch, so the bottleneck is not simply
under-training with the current architecture.

Previous-action conditioning does not rescue the stitch q-head with the current
loss. The previous-action q-head keeps matched Bellman at 0.333 but collapses
both stochastic TRL and the full-Bellman q-head reference to 0.000 on tasks 4
and 5, despite using the same saved full-goal BC controller. This points toward
the q-head objective/representation itself rather than the low-level controller
or a missing previous-action input as the main stitch bottleneck.

For the paper, this should be supporting evidence that joint-action q-heads can
scale beyond PointMaze on at least one AntMaze variant, plus a boundary showing
that stitch needs a different q-head loss or representation before it can
replace the previous-action policy head.
