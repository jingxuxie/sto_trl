# PointMaze Learned-Transition Diagnostic

This diagnostic fits a differentiable categorical transition model over the
inferred PointMaze cell MDP, then runs the same matched-budget Bellman and
stochastic-TRL value updates using the learned transition probabilities. It is
not an end-to-end neural TRL agent, but it is a learned transition/value-module
screen beyond directly applying dynamic programming to the fixed topology table.

Runner: `scripts/run_pointmaze_learned_transition_planner.py`.

## Collapsed Offline Cell-Change Table Softmax

Setting:

- Source MDP: dataset-inferred PointMaze teleport topology.
- Transition target: `dataset_cell_changes`, collapsed from offline observed
  cell changes.
- Transition model: tabular softmax trained by cross-entropy for 1k optimizer
  steps.
- Planner budget: matched `6` sweeps, full Bellman `180` sweeps.
- Evaluation: seed 0, 50 episodes per task, all five OGBench tasks.

| env | method | mean success | task1 | task2 | task3 | task4 | task5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| navigate | Bellman matched | 0.408 | 0.000 | 0.580 | 0.500 | 0.460 | 0.500 |
| navigate | Stochastic TRL | 0.916 | 0.900 | 0.940 | 0.880 | 0.900 | 0.960 |
| navigate | Bellman full | 0.916 | 0.900 | 0.940 | 0.880 | 0.900 | 0.960 |
| stitch | Bellman matched | 0.408 | 0.000 | 0.580 | 0.500 | 0.460 | 0.500 |
| stitch | Stochastic TRL | 0.916 | 0.900 | 0.940 | 0.880 | 0.900 | 0.960 |
| stitch | Bellman full | 0.916 | 0.900 | 0.940 | 0.880 | 0.900 | 0.960 |

Transition diagnostics:

| env | oracle L1 | oracle max abs | oracle top-1 |
| --- | ---: | ---: | ---: |
| navigate | 0.0482 | 0.9999 | 0.978 |
| stitch | 0.0483 | 0.9999 | 0.978 |

Raw files:

- `results/pointmaze_learned_transition_navigate_cellchanges_1k_ep50_seed0.csv`
- `results/pointmaze_learned_transition_stitch_cellchanges_1k_ep50_seed0.csv`

Interpretation: with a learned transition model trained from collapsed offline
cell-change targets, stochastic TRL still reaches the full-Bellman reference at
the matched logarithmic planning budget, while matched Bellman remains near
0.41 success.

## Raw-Observation MLP Transition Head

I added a less privileged transition model that shares one MLP over raw XY
observations plus a high-level action one-hot. It is trained on the same
collapsed offline cell-change targets, then evaluated as a learned transition
table for the stochastic TRL planner. This still uses the current cell
abstraction, but it is no longer a separate free parameter vector for every
high-level `(state, action)` row.

Setting:

- Source MDP: dataset-inferred PointMaze teleport topology.
- Transition target: `dataset_cell_changes`.
- Transition model: raw-observation MLP, hidden dimensions `128,128`, 2k Adam
  steps, learning rate `0.003`.
- Planner budget: matched `6` sweeps, full Bellman `180` sweeps.
- Evaluation: seed 0, transition seeds 0, 1, and 2, 50 episodes per task, all
  five OGBench tasks.

| env | method | mean success | task1 | task2 | task3 | task4 | task5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| navigate | Bellman matched | 0.512 | 0.520 | 0.580 | 0.500 | 0.460 | 0.500 |
| navigate | Stochastic TRL | 0.916 | 0.900 | 0.940 | 0.880 | 0.900 | 0.960 |
| navigate | Bellman full | 0.916 | 0.900 | 0.940 | 0.880 | 0.900 | 0.960 |
| stitch | Bellman matched | 0.483 | 0.447 | 0.580 | 0.500 | 0.460 | 0.427 |
| stitch | Stochastic TRL | 0.916 | 0.900 | 0.940 | 0.880 | 0.900 | 0.960 |
| stitch | Bellman full | 0.916 | 0.900 | 0.940 | 0.880 | 0.900 | 0.960 |

Raw files:

- `results/pointmaze_navigate_rawobs_mlp_cellchanges_ep50_seed0.csv`
- `results/pointmaze_navigate_rawobs_mlp_cellchanges_ep50_tseed1.csv`
- `results/pointmaze_navigate_rawobs_mlp_cellchanges_ep50_tseed2.csv`
- `results/pointmaze_stitch_rawobs_mlp_cellchanges_ep50_seed0.csv`
- `results/pointmaze_stitch_rawobs_mlp_cellchanges_ep50_tseed1.csv`
- `results/pointmaze_stitch_rawobs_mlp_cellchanges_ep50_tseed2.csv`

Interpretation: the raw-observation transition head preserves the main
stochastic TRL advantage and matches full Bellman on both PointMaze teleport
variants. Matched Bellman is higher than in the free table-softmax screen, but
the stochastic transitive backup still gives absolute success gains of 0.404
on navigate and 0.433 on stitch at the same 6-sweep budget, averaged across
three transition seeds.

## Direct Topology-Table Fit

As an upper-bound sanity check, fitting the softmax model directly to the
topology transition table gives near-zero transition error and the same policy
signal:

| env | Bellman matched | Stochastic TRL | Bellman full |
| --- | ---: | ---: | ---: |
| navigate | 0.444 | 0.916 | 0.916 |
| stitch | 0.444 | 0.916 | 0.916 |

Raw files:

- `results/pointmaze_learned_transition_navigate_topology_1k_ep50_seed0.csv`
- `results/pointmaze_learned_transition_stitch_topology_1k_ep50_seed0.csv`

## Negative Raw Dataset-Count Screen

The initial `dataset_counts` target infers discrete high-level actions from
raw continuous PointMaze dataset transitions. That target fit its own labels
well, but it changed the induced MDP enough that full Bellman fell to 0.530 and
stochastic TRL matched the low 0.230 matched-Bellman result on a 20-episode
stitch screen.

Raw file:

- `results/pointmaze_learned_transition_stitch_dataset_counts_1k_ep20_seed0.csv`

Interpretation: the failure is currently a transition-target construction
problem, not evidence against stochastic TRL. A stronger raw-observation
transition learner needs cleaner action semantics or an observation-level
dynamics objective.
