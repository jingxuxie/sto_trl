# Teleport Stitch Screen Summary

This screen checks whether the stochastic TRL topology result transfers from
`navigate` to the harder `stitch` variants.

## Evaluation Speed Notes

AntMaze evaluation was slow for two reasons:

- Failed episodes run to the full 1000-step horizon.
- The learned BC executor was called once per environment step, and the earlier
  runner used a raw Flax apply path inside that loop.

The runner `scripts/run_antmaze_bc_topology_planner.py` now has:

- JIT-compiled per-step BC actor inference.
- `--save-policy` and `--load-policy` to avoid retraining for eval-only
  variants.
- `--goal-candidates-per-state` with `--goal-candidate-mode body_nearest` for
  full-observation waypoint goals that are compatible with the current Ant
  body state.
- `--eval-action-repeat` for optional action reuse during fast screens.
- `--max-episode-steps` for intentionally truncated diagnostic screens.

Default reported results below use `eval_action_repeat=1` and the full
environment horizon.

Fast eval-only note: for loaded AntMaze BC policies, CPU JAX was faster than
GPU JAX on a one-episode-per-task stitch smoke because the loop is dominated by
CPU environment stepping plus many small actor calls. With the exact strong
executor setting (`--waypoint-lookahead 1`, full goals, 16 body-nearest
candidates), `sto_trl_matched` took about `10.1s` on CPU versus `17.0s` on GPU
for five stitch tasks and reached `1.0` success in both runs. Use the GPU for
BC training; prefer CPU for compact eval-only screens unless a larger batch
path is added.

## PointMaze Teleport Stitch

Command:

```bash
conda run -n autoresearcher_sto_trl python scripts/run_pointmaze_topology_planner.py \
  --env-name pointmaze-teleport-stitch-v0 \
  --topology-source dataset \
  --episodes 10 \
  --seeds 0 \
  --methods bellman_matched sto_trl_matched bellman_full \
  --out results/pointmaze_topology_stitch_seed0_ep10.csv \
  --summary-out results/pointmaze_topology_stitch_seed0_ep10.json
```

| method | sweeps | mean success | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bellman matched | 6 | 0.380 | 0.000 | 0.500 | 0.600 | 0.400 | 0.400 |
| Stochastic TRL | 6 | 0.980 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 |
| Bellman full | 180 | 0.980 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 |

Interpretation: the dataset-inferred topology diagnostic transfers cleanly to
the stitch task. Stochastic TRL matches the long Bellman reference with the
matched 6-sweep budget.

Five-seed, 50-episode-per-task stitch check:

- Raw CSV: `results/pointmaze_topology_stitch_5seed_ep50.csv`
- Summary JSON: `results/pointmaze_topology_stitch_5seed_ep50.json`
- Paper table: `results/paper_tables/pointmaze_topology_stitch_5seed.csv`

| method | sweeps | mean success | seed std |
| --- | ---: | ---: | ---: |
| Bellman matched | 6 | 0.343 | 0.039 |
| Stochastic TRL | 6 | 0.901 | 0.023 |
| Bellman full | 180 | 0.901 | 0.023 |

Paired seed-level improvement over matched Bellman is `+0.558`, 95% CI
`[0.501, 0.614]`, and full Bellman minus stochastic TRL is exactly `0.000`.

Deterministic-support ablation:

- Raw CSV: `results/pointmaze_topology_stitch_support_baseline_5seed_ep50.csv`
- Summary JSON:
  `results/pointmaze_topology_stitch_support_baseline_5seed_ep50.json`
- Paper table:
  `results/paper_tables/pointmaze_topology_stitch_support_baseline_5seed.csv`

| method | sweeps | mean success | seed std |
| --- | ---: | ---: | ---: |
| Bellman matched | 6 | 0.343 | 0.039 |
| Support TRL | 6 | 0.449 | 0.049 |
| Stochastic TRL | 6 | 0.901 | 0.023 |
| Bellman full | 180 | 0.901 | 0.023 |

Support TRL treats each observed stochastic teleport outcome as reliable. It is
better than matched Bellman on task 1 but remains far below stochastic TRL,
which supports the stochastic-calibration ablation.

## AntMaze Teleport Stitch

One-episode command:

```bash
env XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  /home/eston/anaconda3/envs/autoresearcher_sto_trl/bin/python \
  scripts/run_antmaze_bc_topology_planner.py \
  --env-name antmaze-teleport-stitch-v0 \
  --bc-steps 20000 \
  --bc-log-interval 5000 \
  --bc-batch-size 2048 \
  --episodes 1 \
  --seeds 0 \
  --methods bellman_matched sto_trl_matched bellman_full \
  --waypoint-lookahead 1 \
  --path-mode greedy \
  --bc-min-future 1 \
  --bc-max-future 30 \
  --goal-representation full \
  --out results/antmaze_bc_topology_stitch_fullgoal_20k_seed0.csv \
  --summary-out results/antmaze_bc_topology_stitch_fullgoal_20k_seed0.json \
  --progress-log results/antmaze_bc_topology_stitch_fullgoal_20k_seed0_progress.jsonl \
  --save-policy results/policies/antmaze_stitch_fullgoal_bc_20k.pkl
```

| method | sweeps | mean success | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bellman matched | 6 | 0.200 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| Stochastic TRL | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Bellman full | 180 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Interpretation: with a shared 20k-step full-observation BC controller,
stochastic TRL solves all five AntMaze stitch tasks in the seed-0 one-episode
screen and matches the 180-sweep Bellman reference. Matched Bellman with the
same 6-sweep budget solves only task 3.

Larger repeated same-seed check:

- Stochastic/full source:
  `results/antmaze_bc_topology_stitch_fullgoal_20k_ep10_seed0_bodyk16_sto_full.csv`
- Matched source:
  `results/antmaze_bc_topology_stitch_fullgoal_20k_ep10_seed0_bodyk16_matched.csv`
- Compact table:
  `results/paper_tables/antmaze_stitch_bc_20k_ep10_seed0.csv`
- Executor: saved 20k-step full-observation BC controller, 16 candidate dataset
  goal observations per topology waypoint, selected by nearest normalized
  non-xy body state. Seed 0, 10 episodes per task.

| method | sweeps | mean success | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bellman matched | 6 | 0.380 | 0.000 | 0.500 | 0.600 | 0.400 | 0.400 |
| Stochastic TRL | 6 | 0.940 | 1.000 | 0.900 | 1.000 | 1.000 | 0.800 |
| Bellman full | 180 | 0.960 | 1.000 | 0.900 | 1.000 | 1.000 | 0.900 |

Interpretation: over 10 episodes per task, stochastic TRL nearly matches the
full Bellman reference and substantially beats matched Bellman under the exact
same controller and goal-selection rule. The small stochastic/full gap is one
additional task-5 executor miss for stochastic TRL.

Compact multi-seed AntMaze stitch check:

- Raw CSV:
  `results/antmaze_bc_topology_stitch_fullgoal_20k_ep5_seed012_bodyk16.csv`
- Figure:
  `results/figures/antmaze_bodyk16_multiseed.svg`
- Setting: same saved 20k-step full-observation BC controller, 16 body-nearest
  waypoint-goal candidates, three evaluation seeds, five episodes per task.

| method | sweeps | mean success | seed std | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bellman matched | 6 | 0.333 | 0.075 | 0.000 | 0.400 | 0.467 | 0.533 | 0.267 |
| Stochastic TRL | 6 | 0.907 | 0.038 | 0.867 | 0.933 | 1.000 | 0.800 | 0.933 |
| Bellman full | 180 | 0.907 | 0.038 | 0.867 | 0.933 | 1.000 | 0.800 | 0.933 |

20-episode AntMaze hard-task check, three evaluation seeds:

- Navigate raw CSV:
  `results/antmaze_bc_topology_navigate_fullgoal_50k_ep20_seed012_bodyk16_cpu.csv`
- Stitch raw CSV:
  `results/antmaze_bc_topology_stitch_fullgoal_20k_ep20_seed012_bodyk16_cpu.csv`
- Table:
  `results/paper_tables/antmaze_bodyk16_ep20_seed012.csv`
- Paired stats table:
  `results/paper_tables/antmaze_bodyk16_ep20_seed012_paired_stats.csv`
- Setting: CPU eval-only run with saved task-specific controller, 16
  body-nearest waypoint-goal candidates, evaluation seeds 0, 1, and 2, 20
  episodes per task.

| env | Bellman matched | Stochastic TRL | Bellman full |
| --- | ---: | ---: | ---: |
| Navigate | 0.310 | 0.947 | 0.947 |
| Stitch | 0.317 | 0.960 | 0.960 |

Paired improvement over matched Bellman: navigate +0.637 with 95% CI
[0.525, 0.749], stitch +0.643 with 95% CI [0.494, 0.793]. The mean gap from
stochastic TRL to Bellman full is 0.000 on both tasks.

Independent controller-seed AntMaze screens:

- Navigate saved policy:
  `results/policies/antmaze_navigate_fullgoal_bc_50k_seed1.pkl`
- Navigate raw CSV:
  `results/antmaze_bc_topology_navigate_fullgoal_50k_bcseed1_ep20_seed012_bodyk16_cpu.csv`
- Stitch saved policy:
  `results/policies/antmaze_stitch_fullgoal_bc_20k_seed1.pkl`
- Stitch raw CSV:
  `results/antmaze_bc_topology_stitch_fullgoal_20k_bcseed1_ep20_seed012_bodyk16_cpu.csv`
- Table:
  `results/paper_tables/antmaze_bcseed1_ep20_seed012.csv`
- Setting: independently trained `bc_seed=1` controllers, three evaluation
  seeds, 20 episodes per task, 16 body-nearest waypoint candidates.

| env | Bellman matched | Stochastic TRL | Bellman full |
| --- | ---: | ---: | ---: |
| Navigate | 0.320 | 0.933 | 0.933 |
| Stitch | 0.323 | 0.950 | 0.950 |

These screens support controller-seed robustness: stochastic TRL still
matches the 180-sweep Bellman reference under the matched 6-sweep budget, while
matched Bellman remains near 0.32.

Stitch controller-seed aggregate:

- Seed-2 saved policy:
  `results/policies/antmaze_stitch_fullgoal_bc_20k_seed2.pkl`
- Seed-2 raw CSV:
  `results/antmaze_bc_topology_stitch_fullgoal_20k_bcseed2_ep20_seed012_bodyk16_cpu.csv`
- Table:
  `results/paper_tables/antmaze_stitch_controller_seeds_ep20_seed012.csv`

| controller seed | Bellman matched | Stochastic TRL | Bellman full |
| ---: | ---: | ---: | ---: |
| 0 | 0.317 | 0.960 | 0.960 |
| 1 | 0.323 | 0.950 | 0.950 |
| 2 | 0.323 | 0.963 | 0.967 |

Across stitch controller seeds 0, 1, and 2, stochastic TRL stays above 0.95
success under the 6-sweep matched budget. Seed 2 trails full Bellman by one
successful episode out of 300 aggregate rollouts.

Planning-budget AntMaze screens:

- Navigate raw CSV:
  `results/antmaze_bc_topology_navigate_budget_ep3_seed0_bodyk16.csv`
- Stitch raw CSV:
  `results/antmaze_bc_topology_stitch_budget_ep3_seed0_bodyk16.csv`
- Figure:
  `results/figures/antmaze_budget.svg`
- Setting: saved task-specific controller and body-nearest waypoint executor,
  seed 0, three episodes per task.

| env | Bellman 6 | Stochastic TRL 6 | Bellman 12 | Bellman 24 | Bellman 45 | Bellman full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Navigate | 0.333 | 1.000 | 0.667 | 0.933 | 1.000 | 1.000 |
| Stitch | 0.333 | 1.000 | 0.667 | 0.933 | 1.000 | 1.000 |

Executor ablation on AntMaze stitch:

- Nearest-xy raw CSV:
  `results/antmaze_bc_topology_stitch_fullgoal_20k_ep3_seed0_nearestxy_sto_full.csv`
- Body-nearest raw CSV:
  `results/antmaze_bc_topology_stitch_fullgoal_20k_ep3_seed0_bodyk16_sto_full.csv`

| executor | Stochastic TRL | Bellman full |
| --- | ---: | ---: |
| nearest-xy waypoint observation | 0.867 | 0.867 |
| 16 body-nearest waypoint candidates | 1.000 | 1.000 |

Interpretation: both stochastic TRL and full Bellman improve together when the
executor receives body-compatible waypoint goals. This is executor sensitivity,
not a value-propagation gap.

## Main Takeaway

The harder stitch tasks strengthen the core story: stochastic transitive
composition reaches or nearly reaches the full-Bellman long-horizon plan under
a small matched planning budget, while matched Bellman is too local. The
remaining paper-grade work is to repeat the strongest AntMaze stitch/navigate
settings across seeds once the controller/evaluation loop is fast enough.

## Navigate Controller Note

A targeted eval-only 50k-step controller check on
`antmaze-teleport-navigate-v0` with the same 16-candidate body-nearest waypoint
goal rule fixed the earlier navigate task-4 failure:

- Policy: `results/policies/antmaze_navigate_fullgoal_bc_50k.pkl`
- Stochastic/full raw:
  `results/antmaze_bc_topology_navigate_fullgoal_50k_ep10_seed0_bodyk16_sto_full.csv`
- Matched raw:
  `results/antmaze_bc_topology_navigate_fullgoal_50k_ep10_seed0_bodyk16_matched.csv`
- Stochastic TRL and Bellman full both reach `0.940` over 10 episodes per task.
- Matched Bellman remains at `0.380` under the same executor.

This suggests the previous navigate task-4 failure was a full-state waypoint
selection issue, not a stochastic-TRL value-propagation failure.

Compact multi-seed navigate check:

- Raw CSV:
  `results/antmaze_bc_topology_navigate_fullgoal_50k_ep5_seed012_bodyk16.csv`
- Stochastic TRL and Bellman full both reach `0.893` over three evaluation
  seeds and five episodes per task.
- Matched Bellman reaches `0.347` under the same executor.
