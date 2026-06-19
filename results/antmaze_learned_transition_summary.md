# AntMaze Learned-Transition Diagnostic

This screen combines the learned AntMaze BC executor with a learned
tabular-softmax high-level transition model. The transition model is trained
from 20 sampled high-level next states per `(state, action)` row of the
dataset-inferred AntMaze topology, then the same matched-budget Bellman and
stochastic-TRL planners are evaluated using the learned transition
probabilities.

Runner: `scripts/run_antmaze_bc_topology_planner.py` with
`--transition-model learned_softmax`.

Setting:

- Transition model: tabular softmax, 1k optimizer steps.
- Transition target: `topology_samples`, 20 sampled outcomes per high-level row.
- Controller: saved full-observation goal-conditioned BC executor.
- Waypoint goals: body-nearest `k=16`.
- Planner budget: matched `6` sweeps, full Bellman `180` sweeps.
- Evaluation: seed 0, hard tasks 4 and 5, 20 episodes per task.

| env | controller | method | mean success | task4 | task5 |
| --- | --- | --- | ---: | ---: | ---: |
| navigate | 50k BC | Bellman matched | 0.325 | 0.400 | 0.250 |
| navigate | 50k BC | Stochastic TRL | 0.925 | 1.000 | 0.850 |
| navigate | 50k BC | Bellman full | 0.925 | 1.000 | 0.850 |
| stitch | 20k BC | Bellman matched | 0.325 | 0.400 | 0.250 |
| stitch | 20k BC | Stochastic TRL | 0.950 | 0.950 | 0.950 |
| stitch | 20k BC | Bellman full | 0.950 | 0.950 | 0.950 |

Transition diagnostics:

| env | oracle L1 | oracle max abs | oracle top-1 |
| --- | ---: | ---: | ---: |
| navigate | 0.0139 | 0.2000 | 0.972 |
| stitch | 0.0135 | 0.1944 | 0.961 |

Raw files:

- `results/antmaze_navigate_learned_transition_samples20_ep20_seed0.csv`
- `results/antmaze_stitch_learned_transition_samples20_ep20_seed0.csv`

Interpretation: this is a harder learned-module screen than the PointMaze
transition diagnostic because it pairs a learned transition model with the
learned Ant controller. Stochastic TRL still reaches the full-Bellman reference
on hard AntMaze teleport tasks, while matched Bellman remains much lower. This
is not yet a verified headline row because it uses one eval seed, but the
20-episode hard-task screen is much less noisy than the earlier 5-episode
screen.

## Transition-Seed Robustness

To check that the learned-transition result is not an artifact of one sampled
transition model, I reran the hard-task screen with transition seeds 0, 1, and
2. This robustness check uses seed 0 for evaluation and 10 episodes per hard
task.

| env | transition seeds | Bellman matched | Stochastic TRL | Bellman full | oracle L1 range | oracle top-1 range |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| navigate | 0,1,2 | 0.300 | 0.950 | 0.950 | 0.0139-0.0147 | 0.961-0.978 |
| stitch | 0,1,2 | 0.367 | 1.000 | 1.000 | 0.0133-0.0152 | 0.939-0.961 |

Raw files:

- `results/antmaze_navigate_learned_transition_samples20_ep10_tseed0.csv`
- `results/antmaze_navigate_learned_transition_samples20_ep10_tseed1.csv`
- `results/antmaze_navigate_learned_transition_samples20_ep10_tseed2.csv`
- `results/antmaze_stitch_learned_transition_samples20_ep10_tseed0.csv`
- `results/antmaze_stitch_learned_transition_samples20_ep10_tseed1.csv`
- `results/antmaze_stitch_learned_transition_samples20_ep10_tseed2.csv`

Across these three independently sampled high-level transition models,
stochastic TRL always matches full Bellman and stays at or above 0.95 success,
while matched Bellman remains much lower.

## Offline Jump-Change Target Screen

I also tested a less privileged AntMaze transition target that fits only
observed long-jump/teleport outcome rows from the offline dataset and leaves
local one-cell moves at the topology scaffold. This avoids sampling from the
already-built topology transition rows for the stochastic part of the model,
while preserving the deterministic local controller scaffold.

Setting:

- Transition model: tabular softmax, 1k optimizer steps.
- Transition target: `dataset_jump_changes`.
- Controller: saved full-observation goal-conditioned BC executor.
- Waypoint goals: body-nearest `k=16`.
- Planner budget: matched `6` sweeps, full Bellman `180` sweeps.
- Evaluation: seed 0, hard tasks 4 and 5, 10 episodes per task, transition
  seeds 0, 1, and 2.

| env | controller | method | mean success | task4 | task5 |
| --- | --- | --- | ---: | ---: | ---: |
| navigate | 50k BC | Bellman matched | 0.300 | 0.400 | 0.200 |
| navigate | 50k BC | Stochastic TRL | 0.950 | 1.000 | 0.900 |
| navigate | 50k BC | Bellman full | 0.950 | 1.000 | 0.900 |
| stitch | 20k BC | Bellman matched | 0.400 | 0.400 | 0.400 |
| stitch | 20k BC | Stochastic TRL | 1.000 | 1.000 | 1.000 |
| stitch | 20k BC | Bellman full | 1.000 | 1.000 | 1.000 |

Transition diagnostics:

| env | oracle L1 | oracle max abs | oracle top-1 |
| --- | ---: | ---: | ---: |
| navigate | 0.000010 | 0.000057 | 0.972 |
| stitch | 0.000010 | 0.000058 | 1.000 |

Raw files:

- `results/antmaze_navigate_learned_transition_jumpchanges_ep10_seed0.csv`
- `results/antmaze_stitch_learned_transition_jumpchanges_ep10_seed0.csv`

Interpretation: this screen is a useful bridge between the sampled-topology
learned-transition appendix and a fully offline learned transition model. The
teleport rows are learned from observed offline jumps; the local rows still use
the topology scaffold. Under that hybrid target, stochastic TRL again reaches
the full-Bellman reference on the hardest AntMaze tasks, while matched Bellman
remains far lower.

## Raw-Observation MLP Jump-Change Target Screen

I added a shared raw-observation transition head for the AntMaze jump-change
target. The model takes each topology cell's mean 29D offline observation plus a
high-level action one-hot, trains only on observed jump/teleport rows, and
falls back to the topology scaffold for local rows. This is still a
cell-abstraction screen, but it is less privileged than a free transition-table
parameter for every high-level row.

Setting:

- Transition model: raw-observation MLP, hidden dimensions `128,128`, 1k Adam
  steps, learning rate `0.003`.
- Transition target: `dataset_jump_changes`.
- Controller: saved full-observation goal-conditioned BC executor.
- Waypoint goals: body-nearest `k=16`.
- Planner budget: matched `6` sweeps, full Bellman `180` sweeps.
- Evaluation: seed 0, hard tasks 4 and 5, 10 episodes per task.

| env | controller | method | mean success | task4 | task5 |
| --- | --- | --- | ---: | ---: | ---: |
| navigate | 50k BC | Bellman matched | 0.300 | 0.400 | 0.200 |
| navigate | 50k BC | Stochastic TRL | 0.950 | 1.000 | 0.900 |
| navigate | 50k BC | Bellman full | 0.950 | 1.000 | 0.900 |
| stitch | 20k BC | Bellman matched | 0.300 | 0.400 | 0.200 |
| stitch | 20k BC | Stochastic TRL | 1.000 | 1.000 | 1.000 |
| stitch | 20k BC | Bellman full | 1.000 | 1.000 | 1.000 |

Raw files:

- `results/antmaze_navigate_rawobs_mlp_jumpchanges_ep10_seed0.csv`
- `results/antmaze_navigate_rawobs_mlp_jumpchanges_ep10_tseed1.csv`
- `results/antmaze_navigate_rawobs_mlp_jumpchanges_ep10_tseed2.csv`
- `results/antmaze_stitch_rawobs_mlp_jumpchanges_ep10_seed0.csv`
- `results/antmaze_stitch_rawobs_mlp_jumpchanges_ep10_tseed1.csv`
- `results/antmaze_stitch_rawobs_mlp_jumpchanges_ep10_tseed2.csv`

Interpretation: the raw-observation transition head preserves the strong
AntMaze hard-task signal from the jump-change screen across three transition
seeds. Stochastic TRL reaches the same success as full Bellman with the
6-sweep budget, while matched Bellman does not propagate the long route.

## Negative Cell-Change Boundary

A direct AntMaze analogue of the PointMaze collapsed cell-change target did not
work. With `dataset_cell_changes`, the transition fit had reasonable top-1
agreement but high oracle L1 on all rows, and the learned model produced paths
that looped or stopped before task 5. In 5-episode hard-task screens,
stochastic TRL and full Bellman both reached only 0.500 success on navigate and
0.500 on stitch, while matched Bellman reached 0.600. This suggests the AntMaze
local rows are too noisy to learn by naive adjacent-cell counting; the useful
offline signal is currently the sparse jump/teleport outcome structure.

Raw files:

- `results/antmaze_navigate_learned_transition_cellchanges_ep5_seed0.csv`
- `results/antmaze_stitch_learned_transition_cellchanges_ep5_seed0.csv`
