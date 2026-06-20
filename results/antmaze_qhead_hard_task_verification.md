# AntMaze Q-Head Hard-Task Verification

Status: PASS

## navigate q-head 5k

- source: `results/antmaze_navigate_qhead_hard_task45_ep3_seed0.csv`
- matched Bellman q-head: 0.333
- stochastic TRL q-head: 1.000
- full Bellman q-head: 1.000
- stochastic task4/task5: 1.000/1.000
- stochastic action agreement: 0.949

## stitch q-head 5k

- source: `results/antmaze_stitch_qhead_hard_task45_ep3_seed0.csv`
- matched Bellman q-head: 0.333
- stochastic TRL q-head: 0.667
- full Bellman q-head: 0.667
- stochastic task4/task5: 1.000/0.333
- stochastic action agreement: 0.948

## stitch q-head 10k

- source: `results/antmaze_stitch_qhead_hard_task45_ep3_seed0_10k.csv`
- matched Bellman q-head: n/a
- stochastic TRL q-head: 0.667
- full Bellman q-head: 0.667
- stochastic task4/task5: 1.000/0.333
- stochastic action agreement: 0.946

## stitch previous-action q-head 5k

- source: `results/antmaze_stitch_prev_qhead_hard_task45_ep3_seed0.csv`
- matched Bellman q-head: 0.333
- stochastic TRL q-head: 0.000
- full Bellman q-head: 0.000
- stochastic task4/task5: 0.000/0.000
- stochastic action agreement: 0.835

