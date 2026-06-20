# AntMaze All-Task Previous-Action Policy Verification

Status: PASS
Checked environments: 4

## antmaze-teleport-navigate-v0 (3 episodes/task)

- source: `results/antmaze_navigate_prev_policy_alltasks_ep3_seed0.csv`
- matched Bellman: 0.200
- stochastic TRL: 0.933
- full Bellman: 0.933
- stochastic-minus-matched: 0.733
- value action agreement: 1.000
- transition oracle top-1: 1.000

## antmaze-teleport-stitch-v0 (3 episodes/task)

- source: `results/antmaze_stitch_prev_policy_alltasks_ep3_seed0.csv`
- matched Bellman: 0.200
- stochastic TRL: 0.933
- full Bellman: 0.933
- stochastic-minus-matched: 0.733
- value action agreement: 1.000
- transition oracle top-1: 0.994

## antmaze-teleport-navigate-v0 (5 episodes/task)

- source: `results/antmaze_navigate_prev_policy_alltasks_ep5_seed0.csv`
- matched Bellman: 0.360
- stochastic TRL: 0.920
- full Bellman: 0.920
- stochastic-minus-matched: 0.560
- value action agreement: 1.000
- transition oracle top-1: 1.000

## antmaze-teleport-stitch-v0 (5 episodes/task)

- source: `results/antmaze_stitch_prev_policy_alltasks_ep5_seed0.csv`
- matched Bellman: 0.360
- stochastic TRL: 0.960
- full Bellman: 0.960
- stochastic-minus-matched: 0.600
- value action agreement: 1.000
- transition oracle top-1: 0.994
