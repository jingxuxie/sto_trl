# Fast Evaluation and Hard-Task Screen

Date: 2026-06-19

## Why Evaluation Feels Slow

The slow path is real environment rollout, not the high-level value backup.
AntMaze evaluation steps MuJoCo one episode at a time for every method, task,
seed, and episode. In the profiled hard-task screens, successful stochastic TRL
episodes still take roughly 600-800 environment steps, and failed matched
Bellman episodes often run to the 1000-step timeout.

Measured AntMaze hard-slice profile:

- action / policy selection: about 0.38-0.48 ms per policy call with JAX
- environment step: about 0.34-0.39 ms per step
- cached setup and value loading: about 1.6 seconds total
- real evaluation: about 3 seconds per method for 3 episodes on each of tasks 4
  and 5

PointMaze is different. With learned high-level heads, the environment rollout
is only a few seconds; policy-head optimization dominates. For fast PointMaze
screening, use exact model evaluation.

## Recommended Fast Loop

PointMaze screens:

- use `--eval-mode model --model-rollout-mode exact`
- use `--task-ids 4 5` for hard-slice tuning
- drop `bellman_full` while tuning unless the full-Bellman comparison is needed
- keep real `--eval-mode env` for confirmation rows only

AntMaze screens:

- use saved BC policies from `results/policies/`
- keep `--task-ids 4 5` for hard long-horizon slices
- use one eval seed and 3-5 episodes per task while tuning
- keep `--profile-eval` on for runtime diagnostics
- keep `--policy-eval-backend jax` for claim rows; `numpy` can be faster but may
  cause long-horizon numerical drift
- do not use `jax_fused` or an 800-step cap for claim rows. The fused backend
  matched one-step actions within `7e-7` but dropped stitch hard-task success
  from 1.00 to 0.90 and was slower in the direct check; the 800-step cap did
  not speed up the successful stochastic row and changes the eval protocol.

Verified rejection source:
`results/eval_speed_rejection_verification.md`.

## New Bounded Screens

PointMaze teleport stitch, previous-action policy head, exact model evaluation:

Source: `results/pointmaze_stitch_prev_policy_model_exact_task45_seed0_cachecheck.csv`

| Method | Tasks | Success |
| --- | --- | ---: |
| bellman_matched | 4,5 | 0.484 |
| sto_trl_matched | 4,5 | 1.000 |
| bellman_full | 4,5 | 1.000 |

AntMaze teleport navigate, raw-observation transition + previous-action policy
head, saved 50k full-goal BC controller, seed 0, 3 episodes per task:

Source: `results/antmaze_navigate_prev_policy_fast_seed0_task45_ep3.csv`

| Method | Tasks | Success |
| --- | --- | ---: |
| bellman_matched | 4,5 | 0.333 |
| sto_trl_matched | 4,5 | 0.833 |
| bellman_full | 4,5 | 0.833 |

AntMaze teleport stitch, raw-observation transition + previous-action policy
head, saved 20k full-goal BC controller, seed 0, 3 episodes per task:

Source: `results/antmaze_stitch_prev_policy_fast_seed0_task45_ep3.csv`

| Method | Tasks | Success |
| --- | --- | ---: |
| bellman_matched | 4,5 | 0.333 |
| sto_trl_matched | 4,5 | 1.000 |
| bellman_full | 4,5 | 1.000 |

These are fast screens, not replacement paper rows. The stronger existing
multi-eval-seed AntMaze previous-action policy-head table remains:
`results/paper_tables/antmaze_rawobs_transition_prev_policy_head_ep10_evalseed012.csv`.
