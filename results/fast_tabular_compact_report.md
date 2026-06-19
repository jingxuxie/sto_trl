# Fast Tabular Stochastic TRL Report

Source CSV: `results/fast_tabular_compact.csv`

## Deterministic Chain

| method | n | success | regret | long_mse | risky_over | safe_opt_risky_rate | risky_opt_risky_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bellman_matched | 3 | 0.000 | 0.5346 | 0.52483 | 0.000 |  |  |
| mc_all_goals | 3 | 1.000 | 0.0000 | 0.41300 | 1.000 |  |  |
| mc_positive | 3 | 1.000 | 0.0000 | 0.41300 | 1.000 |  |  |
| sto_trl_matched | 3 | 1.000 | 0.0000 | 0.26943 | 1.000 |  |  |
| trl_log_realized | 3 | 1.000 | 0.0000 | 0.41300 | 1.000 |  |  |
| trl_raw_realized | 3 | 1.000 | 0.0000 | 0.41300 | 1.000 |  |  |

## Risky Shortcut Aggregate

| method | n | success | regret | long_mse | risky_over | safe_opt_risky_rate | risky_opt_risky_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bellman_full | 27 | 0.955 | 0.0000 | 0.00005 | 1.012 | 0.000 | 1.000 |
| bellman_matched | 27 | 0.436 | 0.3154 | 0.45671 | 1.012 | 1.000 | 1.000 |
| mc_all_goals | 27 | 0.896 | 0.0569 | 0.00688 | 1.012 | 0.000 | 1.000 |
| mc_positive | 27 | 0.436 | 0.3154 | 0.09153 | 4.583 | 1.000 | 1.000 |
| sto_trl_matched | 27 | 0.955 | 0.0000 | 0.00005 | 1.012 | 0.000 | 1.000 |
| trl_log_realized | 27 | 0.436 | 0.3154 | 0.11951 | 4.583 | 1.000 | 1.000 |
| trl_raw_realized | 27 | 0.436 | 0.3154 | 0.09151 | 4.583 | 1.000 | 1.000 |

## Safe-Optimal Key Cases

| method | L | p | action | success | regret | pred_safe | pred_risky | true_risky |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bellman_matched | 8 | 0.1 | 1.0 | 0.098 | 0.7547 | 0.000 | 0.083 | 0.096 |
| sto_trl_matched | 8 | 0.1 | 0.0 | 1.0 | 0.0000 | 0.851 | 0.083 | 0.096 |
| trl_raw_realized | 8 | 0.1 | 1.0 | 0.098 | 0.7547 | 0.851 | 0.960 | 0.096 |
| bellman_matched | 8 | 0.4 | 1.0 | 0.432 | 0.4666 | 0.000 | 0.387 | 0.384 |
| sto_trl_matched | 8 | 0.4 | 0.0 | 1.0 | 0.0000 | 0.851 | 0.387 | 0.384 |
| trl_raw_realized | 8 | 0.4 | 1.0 | 0.432 | 0.4666 | 0.851 | 0.960 | 0.384 |
| bellman_matched | 16 | 0.1 | 1.0 | 0.096 | 0.6278 | 0.000 | 0.089 | 0.096 |
| sto_trl_matched | 16 | 0.1 | 0.0 | 1.0 | 0.0000 | 0.724 | 0.089 | 0.096 |
| trl_raw_realized | 16 | 0.1 | 1.0 | 0.096 | 0.6278 | 0.724 | 0.960 | 0.096 |
| bellman_matched | 16 | 0.4 | 1.0 | 0.356 | 0.3396 | 0.000 | 0.409 | 0.384 |
| sto_trl_matched | 16 | 0.4 | 0.0 | 1.0 | 0.0000 | 0.724 | 0.409 | 0.384 |
| trl_raw_realized | 16 | 0.4 | 1.0 | 0.356 | 0.3396 | 0.724 | 0.960 | 0.384 |
| bellman_matched | 32 | 0.1 | 1.0 | 0.11 | 0.4278 | 0.000 | 0.070 | 0.096 |
| sto_trl_matched | 32 | 0.1 | 0.0 | 1.0 | 0.0000 | 0.524 | 0.070 | 0.096 |
| trl_raw_realized | 32 | 0.1 | 1.0 | 0.11 | 0.4278 | 0.524 | 0.960 | 0.096 |
| bellman_matched | 32 | 0.4 | 1.0 | 0.37 | 0.1397 | 0.000 | 0.377 | 0.384 |
| sto_trl_matched | 32 | 0.4 | 0.0 | 1.0 | 0.0000 | 0.524 | 0.377 | 0.384 |
| trl_raw_realized | 32 | 0.4 | 1.0 | 0.37 | 0.1397 | 0.524 | 0.960 | 0.384 |
