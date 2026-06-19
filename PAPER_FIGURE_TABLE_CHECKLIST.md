# Paper Figure and Table Checklist

Use this when converting the manuscript draft into a conference paper.

## Main Tables

| paper item | source artifact | purpose |
| --- | --- | --- |
| Main hard-task table | `results/paper_tables/main_hard_task_results.csv` | PointMaze and AntMaze headline matched-budget success |
| AntMaze navigate controller seeds | `results/paper_tables/antmaze_navigate_controller_seeds_ep20_seed012.csv` | Controller-seed robustness on AntMaze navigate |
| AntMaze stitch controller seeds | `results/paper_tables/antmaze_stitch_controller_seeds_ep20_seed012.csv` | Controller-seed robustness on hardest AntMaze task |
| PointMaze support ablation | `results/paper_tables/pointmaze_topology_stitch_support_baseline_5seed.csv` | Shows stochastic calibration matters beyond support composition |
| PointMaze focused task 5 | `results/paper_tables/pointmaze_stitch_task5_ep100_seed01234.csv` | Appendix single-task high-episode check |
| PointMaze learned controller | `results/paper_tables/pointmaze_learned_controller_ep20_seed012.csv` | Shows PointMaze signal survives a learned BC executor |
| Controller execution isolation | `results/paper_tables/controller_execution_isolation.csv` | Separates weak direct final-goal actors from successful stochastic TRL waypoint execution |
| PointMaze learned high-level modules | `results/paper_tables/pointmaze_tie_policy_head_ep20_evalseed012.csv` and `results/paper_tables/pointmaze_tie_policy_head_ep20_tseed012.csv` | Shows PointMaze signal survives raw-observation transition and tie-policy heads across eval and transition seeds |
| AntMaze learned high-level modules | `results/paper_tables/antmaze_rawobs_transition_tie_policy_head_ep10_evalseed012.csv` | Shows AntMaze hard-task signal survives raw-observation transition and tie-policy heads |
| AntMaze paired stats | `results/paper_tables/antmaze_bodyk16_ep20_seed012_paired_stats.csv` | Paired eval-seed confidence intervals |
| PointMaze paired stats | `results/paper_tables/pointmaze_topology_paired_stats.csv` and `results/paper_tables/pointmaze_topology_stitch_paired_stats.csv` | Paired seed-level improvements |

## Main Figures

| paper figure | source artifact | message |
| --- | --- | --- |
| Long-horizon shortcut success | `results/figures/tabular_horizon_success.svg` | Stochastic TRL scales across safe path lengths |
| 2D grid shortcut success | `results/figures/grid_shortcut_success.svg` | Stochastic TRL avoids risky portal in 2D |
| Grid planning budget curve | `results/figures/grid_budget_curve.svg` | 7 sweeps versus 126 Bellman sweeps on hardest grid |
| PointMaze graph success | `results/figures/pointmaze_graph_success.svg` | Fully empirical graph diagnostic improves over matched Bellman |
| AntMaze body-candidate screen | `results/figures/antmaze_bodyk16_multiseed.svg` | Learned-controller AntMaze matched-budget signal |
| AntMaze planning budget | `results/figures/antmaze_budget.svg` | Matched stochastic TRL reaches full-Bellman-level success with fewer sweeps |

## Suggested Paper Layout

1. Figure 1: diagram of stochastic shortcut failure and calibrated transitive
   backup.
2. Table 1: main hard-task results.
3. Figure 2: tabular/grid long-horizon budget evidence.
4. Table 2: PointMaze support ablation.
5. Table 3 or Appendix Table: AntMaze controller-seed robustness.
6. Table 4 or Appendix Table: AntMaze learned high-level module screen.
7. Appendix: full per-task AntMaze and PointMaze tables, verifier output,
   command reproduction notes.

## Required Text Around Figures

- For topology diagnostics, say "topology-level stochastic planner with a
  learned executor" rather than "end-to-end neural stochastic TRL".
- For full Bellman, say "long-sweep reference" rather than "oracle policy".
- For controller robustness, say both AntMaze navigate and stitch have three
  independently trained controller seeds.
- For learned high-level modules, say the transition and tie-policy heads are
  learned raw-observation heads over a high-level cell abstraction, not a full
  end-to-end neural agent.

## Consistency Gates

- `scripts/verify_main_claims.py` verifies generated headline tables against
  raw result CSVs.
- `scripts/verify_latex_claims.py` verifies the LaTeX table rows against the
  generated paper-table CSVs before compiling `paper/stochastic_trl/main.tex`.
