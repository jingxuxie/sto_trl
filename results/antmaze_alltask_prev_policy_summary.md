# AntMaze All-Task Previous-Action Policy Screen

Date: 2026-06-19

## Question

The verified AntMaze learned-module rows focus on hard tasks 4 and 5. This
screen checks whether the same raw-observation transition plus previous-action
high-level policy head also preserves the stochastic-TRL advantage across all
five AntMaze teleport tasks under the same saved BC controllers.

## Setting

- Transition model: raw-observation MLP trained on dataset jump changes.
- High-level head: previous-action policy MLP.
- Controller: saved full-goal BC with body-nearest `k=16` waypoint goals.
- Evaluation: seed 0, all five tasks, 3 and 5 episodes per task.
- Planner budget: matched 6 sweeps; full Bellman reference 180 sweeps.

## Results

| env | controller | episodes/task | matched Bellman | stochastic TRL | full Bellman | gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| antmaze-teleport-navigate-v0 | 50k full-goal BC | 3 | 0.200 | 0.933 | 0.933 | 0.733 |
| antmaze-teleport-stitch-v0 | 20k full-goal BC | 3 | 0.200 | 0.933 | 0.933 | 0.733 |
| antmaze-teleport-navigate-v0 | 50k full-goal BC | 5 | 0.360 | 0.920 | 0.920 | 0.560 |
| antmaze-teleport-stitch-v0 | 20k full-goal BC | 5 | 0.360 | 0.960 | 0.960 | 0.600 |

Task-level stochastic TRL success:

| env | episodes/task | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| antmaze-teleport-navigate-v0 | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.667 |
| antmaze-teleport-stitch-v0 | 3 | 1.000 | 0.667 | 1.000 | 1.000 | 1.000 |
| antmaze-teleport-navigate-v0 | 5 | 0.800 | 1.000 | 1.000 | 1.000 | 0.800 |
| antmaze-teleport-stitch-v0 | 5 | 1.000 | 0.800 | 1.000 | 1.000 | 1.000 |

## Raw Files

- `results/antmaze_navigate_prev_policy_alltasks_ep3_seed0.csv`
- `results/antmaze_stitch_prev_policy_alltasks_ep3_seed0.csv`
- `results/antmaze_navigate_prev_policy_alltasks_ep5_seed0.csv`
- `results/antmaze_stitch_prev_policy_alltasks_ep5_seed0.csv`
- `results/antmaze_alltask_prev_policy_verification.md`

## Interpretation

This is bounded supporting evidence, not a replacement for the multi-eval-seed
hard-task table. It shows that the learned high-level transition and
previous-action policy head are not only solving the selected hard tasks: with
one evaluation seed and a small episode budget, they also match full Bellman and
substantially beat matched Bellman across all five AntMaze teleport tasks.
The 5-episode rerun keeps the same qualitative result with a larger rollout
budget: stochastic TRL remains above 0.90 on both variants, matches the
full-Bellman reference, and improves over matched Bellman by at least 0.56.
