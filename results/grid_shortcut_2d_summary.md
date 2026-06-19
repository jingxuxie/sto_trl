# 2D Stochastic Grid Shortcut Summary

Environment family: `stochastic_grid_shortcut`.

The safe route is a deterministic snake corridor embedded in a 2D grid. The
start state also has a risky portal action that reaches the goal with
probability `p_success` and otherwise enters an absorbing trap.

Files:

- Raw rows: `results/grid_shortcut_2d_5seed.csv`
- Summary JSON: `results/grid_shortcut_2d_5seed_summary.json`
- Paper table: `results/paper_tables/grid_shortcut_2d.csv`
- Realized-TRL diagnostic table:
  `results/paper_tables/grid_realized_diagnostic.csv`
- Planning-budget curve: `results/paper_tables/grid_budget_curve.csv`
- Figure: `results/figures/grid_shortcut_success.svg`
- Budget figure: `results/figures/grid_budget_curve.svg`

Settings:

- Grids: `8x4`, `16x4`, `16x8`
- Safe path lengths: `31`, `63`, `127`
- Portal probabilities: `0.02`, `0.05`
- Seeds: `0, 1, 2, 3, 4`
- Matched stochastic TRL/Bellman sweeps: `6`, `7`, `8`
- Full Bellman reference: `4 * path_length` sweeps
- Offline data per setting: `1200` trajectories, behavior portal probability
  `0.5`, action noise `0.01`

Aggregate result:

| method family | success | safe action rate | portal action rate | regret |
| --- | ---: | ---: | ---: | ---: |
| MC positive | 0.036 | 0.000 | 1.000 | 0.263 |
| MC all goals | 0.600 | 0.933 | 0.000 | 0.044 |
| Bellman matched | 0.036 | 0.000 | 1.000 | 0.263 |
| Stochastic TRL matched | 1.000 | 1.000 | 0.000 | 0.000 |
| Bellman full | 1.000 | 1.000 | 0.000 | 0.000 |

Interpretation: on 2D stochastic long-horizon shortcut tasks, the matched
Bellman budget cannot propagate the long safe corridor back to the start, so
it chooses the short risky portal. MC-positive also overestimates successful
portal samples. Stochastic TRL composes calibrated one-step reachability and
recovers the full-Bellman safe policy using logarithmic matched sweeps.

## Realized-TRL Diagnostic

File: `results/grid_shortcut_realized_diagnostic.csv`.

This reduced diagnostic uses grids `8x4` and `16x4`, portal probability
`0.05`, three seeds, and path lengths `31` and `63`. It includes raw/log
realized TRL baselines that are too slow to include in the full grid scaling
sweep.

| method family | success | safe action rate | portal action rate | regret | portal overestimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| MC positive | 0.051 | 0.000 | 1.000 | 0.358 | 20.000 |
| Realized TRL | 0.051 | 0.000 | 1.000 | 0.358 | 20.000 |
| Log realized TRL | 0.051 | 0.000 | 1.000 | 0.358 | 20.000 |
| Bellman matched | 0.051 | 0.000 | 1.000 | 0.358 | 0.898 |
| Stochastic TRL | 1.000 | 1.000 | 0.000 | 0.000 | 0.898 |
| Bellman full | 1.000 | 1.000 | 0.000 | 0.000 | 0.898 |

Interpretation: deterministic-style realized TRL composes lucky portal
successes as if the portal were reliable. The stochastic transitive backup
keeps the portal calibrated and propagates the safe 2D corridor instead.

## Planning-Budget Curve

Files:

- Raw rows: `results/grid_budget_curve_16x8_p005_5seed.csv`
- Summary JSON: `results/grid_budget_curve_16x8_p005_5seed_summary.json`
- Paper table: `results/paper_tables/grid_budget_curve.csv`
- Figure: `results/figures/grid_budget_curve.svg`

Setting: `16x8` grid, safe path length `127`, portal probability `0.05`, five
seeds.

| method | first sweep budget with success 1.0 | success before switch |
| --- | ---: | ---: |
| Bellman | 126 | 0.048 through 120 sweeps |
| Stochastic TRL | 7 | 0.048 through 6 sweeps |

Interpretation: Bellman needs essentially the full path length to propagate
the safe corridor value back to the start. Stochastic TRL reaches the same
safe policy after `ceil(log2(127)) = 7` sweeps in the expanded budget curve;
the main matched benchmark uses a conservative 8-sweep budget for this grid.
