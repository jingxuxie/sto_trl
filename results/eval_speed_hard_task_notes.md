# Evaluation Speed and Hard-Task Iteration Notes

## Why AntMaze evaluation felt slow

The slow part was not stochastic TRL value iteration. In a profiled AntMaze
stitch check, the fixed per-process setup dominated short runs:

- Dataset-topology construction scanned the full offline dataset.
- Waypoint-observation indexing scanned the dataset once per topology cell.
- Loaded BC policies were initializing a throwaway Flax parameter template
  before deserializing saved parameters.
- The actual rollout loop is still nontrivial because AntMaze episodes often
  run for 600-800 MuJoCo steps, and the BC controller is called every step.

`scripts/run_antmaze_bc_topology_planner.py` now caches the deterministic
topology and goal-observation index under `results/cache/`, restores loaded
Flax parameters directly, prints timing lines, and writes setup/eval timing
fields into CSV rows. It also caches topology cell centers in the evaluation
agent and passes the fixed task goal state through the episode loop instead of
recomputing it at every simulator step.

June 19 update: the AntMaze runner also caches learned transition fits and
learned high-level value/control heads. This specifically targets repeated
learned-module screens where the task ids, methods, or episode counts change
but the learned transition/value targets are identical. Cache keys include the
environment, topology states, dataset shape, model type, training steps,
learning rate, seed, hidden dimensions, and a hash of the table target for
value heads. CSV rows now include `transition_cache_hit` and `value_cache_hit`.

## Fast iteration settings

Use one seed, selected hard tasks, loaded policies, and only the methods needed
for the immediate question. For AntMaze, keep these controller settings unless
tuning the controller explicitly:

```bash
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/eston/anaconda3/envs/autoresearcher_sto_trl/bin/python \
scripts/run_antmaze_bc_topology_planner.py \
  --env-name antmaze-teleport-stitch-v0 \
  --load-policy results/policies/antmaze_stitch_fullgoal_bc_20k.pkl \
  --goal-representation full \
  --goal-candidates-per-state 16 \
  --goal-candidate-mode body_nearest \
  --bc-min-future 1 \
  --bc-max-future 30 \
  --bc-batch-size 2048 \
  --waypoint-lookahead 1 \
  --eval-action-repeat 1 \
  --policy-eval-backend jax \
  --methods bellman_matched sto_trl_matched \
  --episodes 20 \
  --seeds 0 \
  --task-ids 4 5
```

Hot-cache setup for this command is about 1.1 seconds on CPU. Rollouts remain
the main cost. For fast algorithm iteration, skip `bellman_full` unless the
question specifically needs the full-sweep reference; this removes one full
set of simulator rollouts. Keep `--policy-eval-backend jax` for claimed
results. The experimental `numpy` backend matched single-step BC actions to
about `1e-6`, but it changed one 600-800 step MuJoCo outcome in a hard-task
screen, so use it only for rough debugging.

For learned-transition screens, the fastest safe loop is:

1. Run one eval seed, hard tasks only, and `bellman_matched sto_trl_matched`.
2. Add `bellman_full` only after stochastic TRL is already near the desired
   success threshold.
3. Keep episodes at 5 per task for screening and scale to 10 or 20 only for
   candidates that clear the screen.

If the transition/value model is unchanged, rerun with the same model settings
to hit `results/cache/`. A raw-observation previous-action policy-head smoke
with one transition/value step went from 11.08s setup on the first run to 1.07s
setup on the cache-hit repeat. The real 2000-step learned heads should benefit
more because their training cost is much larger than this smoke.

For PointMaze topology or learned-transition screens, use the exact high-level
model proxy first:

```bash
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/eston/anaconda3/envs/autoresearcher_sto_trl/bin/python \
scripts/run_pointmaze_topology_planner.py \
  --env-name pointmaze-teleport-stitch-v0 \
  --topology-source dataset \
  --methods bellman_matched sto_trl_matched \
  --task-ids 4 5 \
  --eval-mode model
```

This rolls out the induced stochastic cell MDP exactly over the augmented
`(cell, previous_action)` state, so it avoids simulator steps and Monte Carlo
noise. It is a screening proxy only; keep real OGBench environment rollouts for
paper numbers and final controller checks.

On the June 19 AntMaze `dataset_jump_changes` checks, cached setup was about
3.7 seconds because the transition-softmax fit adds about 2.5 seconds. Value
training itself was about 0.04 seconds. Each 10-episode hard-task method then
took about 9.3-10.6 seconds of sequential rollout time. That means dropping the
full-Bellman confirmation method saves roughly one third of the wall time for
these confirmation screens.

In the latest table-value AntMaze hard-task screens, hot-cache setup was about
1.12 seconds. Rollout time remained the bottleneck: 10 episodes each on tasks 4
and 5 took about 8.8-10.2 seconds per method. Profiling split that time roughly
between JAX policy calls and MuJoCo stepping, with each around 0.33-0.35 ms per
step/call.

## Profiling check on June 19

I added `--profile-eval` to `scripts/run_antmaze_bc_topology_planner.py`.
It records reset time, policy-action time, MuJoCo step time, simulator steps,
and policy calls in each CSV row.

On `antmaze-teleport-stitch-v0` tasks 4 and 5 with 5 episodes per task,
loaded 20k-step full-observation BC, body-nearest `k=16`, JAX policy backend,
and only `sto_trl_matched`:

- Success: 1.00.
- Hot-cache setup: 1.08s.
- Evaluation: 4.74s for 6,588 simulator steps.
- Policy calls: 6,588 at 0.378 ms/call, 2.49s total.
- MuJoCo stepping: 0.337 ms/step, 2.22s total.
- Reset overhead: 0.02s.
- Raw files: `results/antmaze_stitch_hard_tasks_profile_ep5_seed0.csv`,
  `results/antmaze_stitch_hard_tasks_profile_ep5_seed0.json`.

The NumPy backend on the same screen reduced evaluation to 3.58s and policy
calls to 0.177 ms/call, but success dropped from 1.00 to 0.80. A random-batch
comparison between JAX and NumPy policy outputs showed max absolute action
difference about `2e-6`; the failure is long-horizon MuJoCo sensitivity, not a
large implementation mismatch. Keep JAX for paper numbers.

PointMaze stitch hard-task profiling after adding cached cell centers:

- Real environment, tasks 4 and 5, 5 episodes per task, matched Bellman plus
  stochastic TRL: 0.30-0.33s eval time per method, with about 0.061 ms per
  simulator step.
- Exact model proxy, same tasks and methods: 0.001-0.021s eval time per method.
- Raw files:
  `results/pointmaze_stitch_eval_profile_ep5_seed0.csv`,
  `results/pointmaze_stitch_model_eval_exact_seed0.csv`.

## Rejected speed knobs

- `--eval-action-repeat 2` made stitch hard-task success collapse to 0.50 in a
  5-episode-per-task check, so action repeat should stay at 1.
- `--path-mode persistent` is not a universal speed/performance knob. It
  reduced stitch hard-task success to 0.80 in the same small check. A
  navigate-only diagnostic reached 0.917 stochastic-TRL success across three
  evaluation seeds on AntMaze navigate tasks 4 and 5 while matched Bellman
  stayed at 0.00, but this is not better than the existing greedy hard-task
  aggregate. Keep greedy as the default and treat persistent pathing as a
  non-promoted executor ablation.
- `--goal-candidates-per-state 8` and `32` both reduced stitch hard-task
  success to 0.80 in small checks. The current `16` body-nearest setting is
  the best tested balance.
- `--policy-eval-backend numpy` reduced a six-episode hard-task stitch rollout
  from 2.87s to 2.10s, but success changed from 1.00 to 0.83 because tiny
  numerical action differences compounded over long MuJoCo rollouts.

## Latest single-seed hard-task checks

PointMaze teleport stitch, all tasks, 100 episodes per task:

- `bellman_matched`: 0.334
- `sto_trl_matched`: 0.894
- Task 5: 0.330 vs 0.900
- Raw files: `results/pointmaze_stitch_seed0_ep100_matched_vs_sto.csv`,
  `results/pointmaze_stitch_seed0_ep100_matched_vs_sto.json`

PointMaze teleport stitch, all tasks, 20 episodes per task:

- `bellman_matched`: 0.38
- `support_trl_matched`: 0.53
- `sto_trl_matched`: 0.98
- `bellman_full`: 0.98
- Raw files: `results/pointmaze_stitch_seed0_ep20_methods.csv`,
  `results/pointmaze_stitch_seed0_ep20_methods.json`

PointMaze teleport stitch hard tasks 4 and 5, 50 episodes per task, seed 0:

| method | success | task4 | task5 | eval seconds |
| --- | ---: | ---: | ---: | ---: |
| Bellman matched | 0.420 | 0.460 | 0.380 | 4.26 |
| Support TRL | 0.480 | 0.460 | 0.500 | 4.11 |
| Stochastic TRL | 0.930 | 0.900 | 0.960 | 3.22 |
| Bellman full | 0.930 | 0.900 | 0.960 | 3.21 |

Raw files:
`results/pointmaze_stitch_hard_task45_ep50_seed0_fastfocus.csv`,
`results/pointmaze_stitch_hard_task45_ep50_seed0_fastfocus.json`.

AntMaze teleport hard tasks 4 and 5, 10 episodes per task, seed 0:

| env | method | success | task4 | task5 | eval seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| navigate | Bellman matched | 0.300 | 0.400 | 0.200 | 10.19 |
| navigate | Stochastic TRL | 0.950 | 1.000 | 0.900 | 9.20 |
| navigate | Bellman full | 0.950 | 1.000 | 0.900 | 9.20 |
| stitch | Bellman matched | 0.350 | 0.300 | 0.400 | 9.65 |
| stitch | Stochastic TRL | 1.000 | 1.000 | 1.000 | 8.81 |
| stitch | Bellman full | 1.000 | 1.000 | 1.000 | 8.95 |

Raw files:
`results/antmaze_navigate_hard_task45_ep10_seed0_fastfocus.csv`,
`results/antmaze_stitch_hard_task45_ep10_seed0_fastfocus.csv`.

AntMaze teleport navigate, tasks 4 and 5, 5 episodes per task:

- `bellman_matched`: 0.40
- `sto_trl_matched`: 0.90
- `bellman_full`: 0.90
- Raw files: `results/antmaze_navigate_hard_tasks_bodyk16_seed0_methods.csv`,
  `results/antmaze_navigate_hard_tasks_bodyk16_seed0_methods.json`

AntMaze teleport stitch, tasks 4 and 5, 5 episodes per task:

- `bellman_matched`: 0.60
- `sto_trl_matched`: 1.00
- `bellman_full`: 1.00
- Raw files: `results/antmaze_stitch_hard_tasks_bodyk16_seed0_methods.csv`,
  `results/antmaze_stitch_hard_tasks_bodyk16_seed0_methods.json`

AntMaze teleport navigate, tasks 4 and 5, 20 episodes per task:

- `bellman_matched`: 0.35
- `sto_trl_matched`: 0.925
- Rollout seconds: 18.64 for matched Bellman, 19.02 for stochastic TRL.
- Raw files: `results/antmaze_navigate_hard_tasks_ep20_seed0_matched_vs_sto.csv`,
  `results/antmaze_navigate_hard_tasks_ep20_seed0_matched_vs_sto.json`

AntMaze teleport stitch, tasks 4 and 5, 20 episodes per task:

- `bellman_matched`: 0.40
- `sto_trl_matched`: 0.95
- Rollout seconds: 17.14 for matched Bellman, 17.24 for stochastic TRL.
- Raw files: `results/antmaze_stitch_hard_tasks_ep20_seed0_matched_vs_sto.csv`,
  `results/antmaze_stitch_hard_tasks_ep20_seed0_matched_vs_sto.json`

AntMaze teleport navigate with learned `dataset_jump_changes` transitions,
tasks 4 and 5, 10 episodes per task:

- `bellman_matched`: 0.30
- `sto_trl_matched`: 0.95
- `bellman_full`: 0.95
- Raw files:
  `results/antmaze_navigate_learned_transition_jumpchanges_ep10_seed0.csv`,
  `results/antmaze_navigate_learned_transition_jumpchanges_ep10_seed0.json`

AntMaze teleport stitch with learned `dataset_jump_changes` transitions, tasks
4 and 5, 10 episodes per task:

- `bellman_matched`: 0.40
- `sto_trl_matched`: 1.00
- `bellman_full`: 1.00
- Raw files:
  `results/antmaze_stitch_learned_transition_jumpchanges_ep10_seed0.csv`,
  `results/antmaze_stitch_learned_transition_jumpchanges_ep10_seed0.json`

PointMaze teleport navigate with learned raw-observation transition and learned
tie-preserving raw-observation policy head, all tasks, 20 episodes per task:

- `bellman_matched`: 0.530
- `sto_trl_matched`: 0.980
- `bellman_full`: 0.980
- Raw files:
  `results/pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv`,
  `results/pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.json`

PointMaze teleport stitch with learned raw-observation transition and learned
tie-preserving raw-observation policy head, all tasks, 20 episodes per task:

- `bellman_matched`: 0.530
- `sto_trl_matched`: 0.980
- `bellman_full`: 0.980
- Raw files:
  `results/pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv`,
  `results/pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.json`

AntMaze teleport navigate with learned BC executor and learned
tie-preserving raw-observation high-level policy head, tasks 4 and 5,
5 episodes per task:

- `bellman_matched`: 0.400
- `sto_trl_matched`: 0.900
- `bellman_full`: 0.800
- Value-head training/setup seconds: 31.93 / 32.92
- Rollout seconds: 4.73 for matched Bellman, 5.01 for stochastic TRL, 5.14
  for full Bellman
- Raw files:
  `results/antmaze_navigate_tie_policy_head_hard_tasks_ep5_seed0.csv`,
  `results/antmaze_navigate_tie_policy_head_hard_tasks_ep5_seed0.json`

AntMaze teleport stitch with learned BC executor and learned
tie-preserving raw-observation high-level policy head, tasks 4 and 5,
5 episodes per task:

- `bellman_matched`: 0.600
- `sto_trl_matched`: 1.000
- `bellman_full`: 1.000
- Value-head training/setup seconds: 30.14 / 31.12
- Rollout seconds: 3.75 for matched Bellman, 4.53 for stochastic TRL, 4.42
  for full Bellman
- Raw files:
  `results/antmaze_stitch_tie_policy_head_hard_tasks_ep5_seed0.csv`,
  `results/antmaze_stitch_tie_policy_head_hard_tasks_ep5_seed0.json`

Promoted 20-episode AntMaze teleport navigate tie-policy-head check, tasks 4
and 5:

- `bellman_matched`: 0.350
- `sto_trl_matched`: 0.925
- `bellman_full`: 0.925
- Value-head training/setup seconds: 30.69 / 31.68
- Rollout seconds: 19.74 for matched Bellman, 19.14 for stochastic TRL, 18.81
  for full Bellman
- Raw files:
  `results/antmaze_navigate_tie_policy_head_hard_tasks_ep20_seed0.csv`,
  `results/antmaze_navigate_tie_policy_head_hard_tasks_ep20_seed0.json`

Promoted 20-episode AntMaze teleport stitch tie-policy-head check, tasks 4 and
5:

- `bellman_matched`: 0.400
- `sto_trl_matched`: 0.950
- `bellman_full`: 0.950
- Value-head training/setup seconds: 31.05 / 32.06
- Rollout seconds: 18.58 for matched Bellman, 19.01 for stochastic TRL, 18.81
  for full Bellman
- Raw files:
  `results/antmaze_stitch_tie_policy_head_hard_tasks_ep20_seed0.csv`,
  `results/antmaze_stitch_tie_policy_head_hard_tasks_ep20_seed0.json`

Combined raw-observation transition plus tie-policy-head AntMaze navigate
check, tasks 4 and 5, 10 episodes per task:

- `bellman_matched`: 0.300
- `sto_trl_matched`: 0.950
- `bellman_full`: 0.950
- Transition oracle top-1 / L1: 1.000 / 0.000006
- Value-head training/setup seconds: 31.01 / 38.64
- Rollout seconds: 10.61 for matched Bellman, 9.40 for stochastic TRL, 9.58
  for full Bellman
- Raw files:
  `results/antmaze_navigate_rawobs_transition_tie_policy_head_ep10_seed0.csv`,
  `results/antmaze_navigate_rawobs_transition_tie_policy_head_ep10_seed0.json`

Combined raw-observation transition plus tie-policy-head AntMaze stitch check,
tasks 4 and 5, 10 episodes per task:

- `bellman_matched`: 0.300
- `sto_trl_matched`: 1.000
- `bellman_full`: 1.000
- Transition oracle top-1 / L1: 1.000 / 0.000065
- Value-head training/setup seconds: 30.71 / 38.14
- Rollout seconds: 10.51 for matched Bellman, 9.17 for stochastic TRL, 9.41
  for full Bellman
- Raw files:
  `results/antmaze_stitch_rawobs_transition_tie_policy_head_ep10_seed0.csv`,
  `results/antmaze_stitch_rawobs_transition_tie_policy_head_ep10_seed0.json`

Transition-seed robustness for the combined raw-observation transition plus
tie-policy-head AntMaze screen:

- Navigate transition seeds 0, 1, and 2 all reach `sto_trl_matched` 0.950 and
  `bellman_full` 0.950, while `bellman_matched` stays at 0.300.
- Stitch transition seeds 0, 1, and 2 all reach `sto_trl_matched` 1.000 and
  `bellman_full` 1.000, while `bellman_matched` stays at 0.300.
- Additional raw files:
  `results/antmaze_navigate_rawobs_transition_tie_policy_head_ep10_tseed1.csv`,
  `results/antmaze_navigate_rawobs_transition_tie_policy_head_ep10_tseed2.csv`,
  `results/antmaze_stitch_rawobs_transition_tie_policy_head_ep10_tseed1.csv`,
  `results/antmaze_stitch_rawobs_transition_tie_policy_head_ep10_tseed2.csv`

Evaluation-seed robustness for the same combined screen with transition seed 0
fixed:

- Navigate evaluation seeds 0, 1, and 2 average `bellman_matched` 0.283,
  `sto_trl_matched` 0.933, and `bellman_full` 0.933.
- Stitch evaluation seeds 0, 1, and 2 average `bellman_matched` 0.283,
  `sto_trl_matched` 0.967, and `bellman_full` 0.967.
- Additional raw files:
  `results/antmaze_navigate_rawobs_transition_tie_policy_head_ep10_tseed0_evalseed12.csv`,
  `results/antmaze_stitch_rawobs_transition_tie_policy_head_ep10_tseed0_evalseed12.csv`
