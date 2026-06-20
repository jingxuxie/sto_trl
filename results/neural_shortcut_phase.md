# Neural Stochastic TRL Shortcut Phase Screen

Date: 2026-06-19

This screen is the first learned-critic evidence for the stochastic TRL idea.
It trains a small MLP goal-conditioned reachability critic on finite stochastic
shortcut MDPs, using one-hot `(state, action, goal)` inputs and fitted operator
targets.

Primary artifacts:

- `agents/sto_trl_neural_toy.py`
- `scripts/run_neural_shortcut_phase.py`
- `results/paper_tables/neural_shortcut_phase.csv`
- `results/figures/neural_shortcut_phase.pdf`
- `results/neural_shortcut_phase_summary.json`

Run configuration:

- safe lengths: 16, 32, 64
- shortcut success probabilities: 0.02, 0.05, 0.10, 0.20
- seed: 0
- offline trajectories per setting: 2000
- transition source: empirical estimate from offline trajectories
- gamma: 0.995
- matched stochastic-TRL sweeps: `ceil(log2(L)) + 1`
- MLP: hidden dims 128,128
- optimizer: Adam, lr 0.003
- warmup: 300 steps on one-step targets
- operator fit: 180 steps per operator iteration
- rank CE weight: 0.05

Aggregate result:

| Method | Exact success | Correct decision | Risky action rate |
| --- | ---: | ---: | ---: |
| neural_sto_trl | 1.000 | 1.000 | 0.000 |
| neural_bellman_td | 0.027 | 0.583 | 0.417 |
| neural_support_trl | 0.093 | 0.000 | 1.000 |
| table_sto_trl_matched | 1.000 | 1.000 | 0.000 |
| table_bellman_matched | 0.093 | 0.000 | 1.000 |
| table_support_trl | 0.093 | 0.000 | 1.000 |
| table_full_bellman | 1.000 | 1.000 | 0.000 |

Per-horizon neural stochastic TRL result:

| Safe length | Exact success | Correct decision | Risky action rate |
| ---: | ---: | ---: | ---: |
| 16 | 1.000 | 1.000 | 0.000 |
| 32 | 1.000 | 1.000 | 0.000 |
| 64 | 1.000 | 1.000 | 0.000 |

Interpretation:

The learned stochastic TRL critic recovers the table stochastic TRL decision
boundary and full-Bellman-level exact success under a logarithmic matched sweep
budget. Neural Bellman TD does not propagate the full safe route under the same
budget, and support-style transitive composition over-composes the lucky risky
shortcut outcome.

This supports the paper claim that Bellman calibration plus transitive
composition can be trained as a neural value function on controlled stochastic
long-horizon tasks. It is not yet an OGBench end-to-end actor result; the next
step is to port the same learned-critic objective to PointMaze observations.
