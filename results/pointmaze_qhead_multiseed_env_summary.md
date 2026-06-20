# PointMaze Q-Head Multi-Seed Real-Environment Summary

Date: 2026-06-19

## Question

The generated-target joint-action q-head matched the exact high-level model and
single-seed real environment rollouts. This screen checks whether the same
learned critic remains strong across multiple PointMaze rollout seeds.

## Setting

- Environments: `pointmaze-teleport-navigate-v0`,
  `pointmaze-teleport-stitch-v0`.
- Transition model: raw-observation MLP trained on dataset cell changes.
- Critic features: raw-observation state/goal features plus state/goal one-hot
  identifiers.
- Evaluation: real PointMaze environment, all five tasks, seeds 0, 1, and 2,
  10 episodes per task.
- Controller: proportional cell controller from the topology planner.
- Planner budget: matched 6 stochastic/transitive iterations; full Bellman
  reference uses 180 sweeps.

## Results

| env | qhead Bellman TD | generated-target qhead | table matched Bellman | table stochastic TRL | table full Bellman | qhead action agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pointmaze-teleport-navigate-v0 | 0.000 | 0.933 | 0.393 | 0.933 | 0.933 | 0.966 |
| pointmaze-teleport-stitch-v0 | 0.027 | 0.933 | 0.393 | 0.933 | 0.933 | 0.959 |

Task-level generated-target q-head success:

| env | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pointmaze-teleport-navigate-v0 | 0.900 | 0.933 | 0.933 | 0.933 | 0.967 |
| pointmaze-teleport-stitch-v0 | 0.900 | 0.933 | 0.933 | 0.933 | 0.967 |

## Raw Files

- `results/pointmaze_qhead_target_fit_navigate_all_env_ep10_seed012.csv`
- `results/pointmaze_qhead_target_fit_stitch_all_env_ep10_seed012.csv`
- `results/qhead_multiseed_env_claim_verification.md`

## Interpretation

The generated-target q-head matches the stochastic-TRL and full-Bellman table
policies under real environment rollout across three evaluation seeds. This
strengthens the earlier single-seed real-env evidence while preserving the
important contrast against matched-budget q-head Bellman TD and table Bellman.
It remains a two-phase generated-target learned-critic diagnostic, not a fully
self-bootstrapped fitted-iteration result.
