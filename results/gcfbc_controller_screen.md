# AntMaze GCFBC Controller Screen

Date: 2026-06-19.

This screen tests whether a TRL-style GCFBC local controller from
`external/trl/agents/gcfbc.py` is ready to replace the simpler MSE BC executor
in the AntMaze topology-planner diagnostics.

## Runner Changes

- `scripts/run_antmaze_gcfbc_topology_planner.py` now supports the waypoint
  candidate knobs that were important for the working MSE BC runner: multiple
  goal observations per topology state, `body_nearest` candidate selection,
  `greedy` versus `persistent` path modes, action repeat, and max episode caps.
- `external/trl/agents/gcfbc.py` now applies its existing `temperature`
  argument to the initial flow noise. This makes deterministic sampling
  (`temperature=0`) and stochastic sampling (`temperature=1`) explicit.

## Results

All rows use `antmaze-teleport-navigate-v0`, seed 0, `sto_trl_matched`,
6 planner sweeps, `min_jump_count=20`, `waypoint_lookahead=1`,
`goal-candidates-per-state=16` for the body-nearest rows, and stochastic
flow sampling unless otherwise noted.

| controller screen | task ids | episodes/task | sample temp | success | final dist | source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| old GCFBC, 100 updates, nearest goal | 3 | 1 | implicit 1.0 | 0.000 | 35.26 | `results/antmaze_gcfbc_topology_task3_100_smoke.csv` |
| old GCFBC, 2k updates, nearest goal | 3 | 1 | implicit 1.0 | 0.000 | 35.71 | `results/antmaze_gcfbc_topology_task3_2k_smoke.csv` |
| GCFBC, 5k updates, body-nearest goals | 3 | 2 | 1.0 | 0.500 | 10.03 | `results/antmaze_gcfbc_topology_task3_5k_bodyk16_temp1_smoke.csv` |
| GCFBC, 5k updates, deterministic flow | 3 | 2 | 0.0 | 0.000 | 13.18 | `results/antmaze_gcfbc_topology_task3_5k_bodyk16_deterministic_smoke.csv` |
| GCFBC, 20k updates, body-nearest goals | 3 | 5 | 1.0 | 0.600 | 4.38 | `results/antmaze_gcfbc_topology_task3_20k_bodyk16_temp1_ep5.csv` |
| GCFBC, 10k updates, hard tasks body-nearest goals | 4,5 | 3 | 1.0 | 0.000 | 25.91 | `results/antmaze_gcfbc_navigate_hard_task45_10k_ep3.csv` |

The current main MSE BC executor is still much stronger in the same topology
setup. In the seed-0 20-episode AntMaze navigate table, it reaches 0.950 overall
success and 0.950 on task 3 for `sto_trl_matched`; see
`results/antmaze_bc_topology_navigate_fullgoal_50k_ep20_seed012_bodyk16_cpu.csv`.

## Interpretation

The standard GCFBC controller is no longer completely broken after adding
body-compatible waypoint candidates on the easier task-3 slice, but it is not
yet strong enough for the main paper claim. The new hard-task screen reaches
0.000 success on tasks 4 and 5 after 10k GPU updates, while the verified MSE BC
executor reaches high success on the same hard-task slice. Stochastic flow
sampling is currently better than deterministic zero-noise sampling at the
task-3 budget. For the paper, keep the main AntMaze claim on the verified MSE
BC executor and treat GCFBC as a controller ablation/negative result unless a
longer or more faithful GCFBC run reaches the 0.90+ success regime.

## Reproduction Commands

5k stochastic-sampling smoke:

```bash
env JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/eston/anaconda3/envs/autoresearcher_sto_trl/bin/python \
  scripts/run_antmaze_gcfbc_topology_planner.py \
  --env-name antmaze-teleport-navigate-v0 \
  --gcfbc-steps 5000 \
  --gcfbc-batch-size 512 \
  --gcfbc-flow-steps 3 \
  --gcfbc-sample-temperature 1.0 \
  --gcfbc-log-interval 1000 \
  --goal-candidates-per-state 16 \
  --goal-candidate-mode body_nearest \
  --waypoint-lookahead 1 \
  --path-mode greedy \
  --eval-action-repeat 1 \
  --episodes 2 \
  --seeds 0 \
  --task-ids 3 \
  --methods sto_trl_matched \
  --out results/antmaze_gcfbc_topology_task3_5k_bodyk16_temp1_smoke.csv \
  --summary-out results/antmaze_gcfbc_topology_task3_5k_bodyk16_temp1_smoke.json
```

20k stochastic-sampling smoke:

```bash
env JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/eston/anaconda3/envs/autoresearcher_sto_trl/bin/python \
  scripts/run_antmaze_gcfbc_topology_planner.py \
  --env-name antmaze-teleport-navigate-v0 \
  --gcfbc-steps 20000 \
  --gcfbc-batch-size 512 \
  --gcfbc-flow-steps 3 \
  --gcfbc-sample-temperature 1.0 \
  --gcfbc-log-interval 5000 \
  --goal-candidates-per-state 16 \
  --goal-candidate-mode body_nearest \
  --waypoint-lookahead 1 \
  --path-mode greedy \
  --eval-action-repeat 1 \
  --episodes 5 \
  --seeds 0 \
  --task-ids 3 \
  --methods sto_trl_matched \
  --out results/antmaze_gcfbc_topology_task3_20k_bodyk16_temp1_ep5.csv \
  --summary-out results/antmaze_gcfbc_topology_task3_20k_bodyk16_temp1_ep5.json
```

10k hard-task stochastic-sampling screen:

```bash
env JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/eston/anaconda3/envs/autoresearcher_sto_trl/bin/python \
  scripts/run_antmaze_gcfbc_topology_planner.py \
  --env-name antmaze-teleport-navigate-v0 \
  --gcfbc-steps 10000 \
  --gcfbc-batch-size 512 \
  --gcfbc-flow-steps 3 \
  --gcfbc-sample-temperature 1.0 \
  --gcfbc-log-interval 2500 \
  --goal-candidates-per-state 16 \
  --goal-candidate-mode body_nearest \
  --waypoint-lookahead 1 \
  --path-mode greedy \
  --eval-action-repeat 1 \
  --episodes 3 \
  --seeds 0 \
  --task-ids 4 5 \
  --methods sto_trl_matched \
  --out results/antmaze_gcfbc_navigate_hard_task45_10k_ep3.csv \
  --summary-out results/antmaze_gcfbc_navigate_hard_task45_10k_ep3.json
```
