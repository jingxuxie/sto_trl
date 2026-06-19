# Paper Artifacts

Generated from the current stochastic TRL prototype results.

## Artifact Files

- `results/main_claim_verification.md`
- `results/paper_tables/main_hard_task_results.csv`
- `results/paper_tables/hard_task_stress_seed0.csv`
- `results/paper_tables/pointmaze_stitch_task5_ep100_seed01234.csv`
- `results/paper_tables/pointmaze_learned_transition.csv`
- `results/paper_tables/pointmaze_learned_controller_ep20_seed012.csv`
- `results/paper_tables/controller_execution_isolation.csv`
- `results/paper_tables/fast_eval_profile.csv`
- `results/paper_tables/antmaze_support_ablation_ep5_seed0_task45.csv`
- `results/paper_tables/pointmaze_tie_policy_head_ep20_seed0.csv`
- `results/paper_tables/pointmaze_rawobs_transition_prev_policy_head_ep20_seed0.csv`
- `results/paper_tables/pointmaze_tie_policy_head_ep20_evalseed012.csv`
- `results/paper_tables/pointmaze_tie_policy_head_ep20_tseed012.csv`
- `results/paper_tables/antmaze_tie_policy_head_hard_tasks_ep20_seed0.csv`
- `results/paper_tables/antmaze_rawobs_transition_tie_policy_head_ep10_tseed012.csv`
- `results/paper_tables/antmaze_rawobs_transition_tie_policy_head_ep10_evalseed012.csv`
- `results/paper_tables/antmaze_rawobs_transition_prev_policy_head_ep10_evalseed012.csv`
- `results/paper_tables/antmaze_learned_transition_robustness.csv`
- `results/paper_tables/tabular_l128.csv`
- `results/paper_tables/tabular_safe_horizon.csv`
- `results/paper_tables/tabular_risky_aggregate.csv`
- `results/paper_tables/grid_shortcut_2d.csv`
- `results/paper_tables/grid_realized_diagnostic.csv`
- `results/paper_tables/grid_budget_curve.csv`
- `results/paper_tables/pointmaze_graph_5seed.csv`
- `results/paper_tables/pointmaze_graph_paired_stats.csv`
- `results/paper_tables/pointmaze_graph_task_deltas.csv`
- `results/paper_tables/pointmaze_topology_5seed.csv`
- `results/paper_tables/pointmaze_topology_stitch_5seed.csv`
- `results/paper_tables/pointmaze_topology_stitch_support_baseline_5seed.csv`
- `results/paper_tables/pointmaze_topology_paired_stats.csv`
- `results/paper_tables/pointmaze_topology_stitch_paired_stats.csv`
- `results/paper_tables/antmaze_bc_topology_20k_ep3_seed0.csv`
- `results/paper_tables/antmaze_navigate_50k_bodyk16_ep10_seed0.csv`
- `results/paper_tables/antmaze_bodyk16_multiseed_ep5.csv`
- `results/paper_tables/antmaze_bcseed1_ep20_seed012.csv`
- `results/paper_tables/antmaze_stitch_controller_seeds_ep20_seed012.csv`
- `results/paper_tables/antmaze_navigate_controller_seeds_ep20_seed012.csv`
- `results/paper_tables/antmaze_bodyk16_ep20_seed012.csv`
- `results/paper_tables/antmaze_bodyk16_ep20_seed012_paired_stats.csv`
- `results/paper_tables/antmaze_budget_ep3_seed0.csv`
- `results/paper_tables/antmaze_stitch_executor_ablation_ep3_seed0.csv`
- `results/paper_tables/teleport_stitch_screen_seed0.csv`
- `results/figures/tabular_l128_safe_success.svg`
- `results/figures/tabular_horizon_success.svg`
- `results/figures/grid_shortcut_success.svg`
- `results/figures/grid_budget_curve.svg`
- `results/figures/pointmaze_graph_success.svg`
- `results/figures/antmaze_bodyk16_multiseed.svg`
- `results/figures/antmaze_budget.svg`

## Main Hard-Task Results

| env | executor | eval_setting | controller_steps | matched_sweeps | full_sweeps | bellman_matched | stochastic_trl | bellman_full | sto_minus_matched | sto_equals_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pointmaze-teleport-navigate-v0 | dataset topology scaffold | 5 eval seeds, 50 episodes/task | 0 | 6 | 180 | 0.343 | 0.901 | 0.901 | 0.558 | True |
| pointmaze-teleport-stitch-v0 | dataset topology scaffold | 5 eval seeds, 50 episodes/task | 0 | 6 | 180 | 0.343 | 0.901 | 0.901 | 0.558 | True |
| antmaze-teleport-navigate-v0 | full-goal BC + body-nearest k16 | 3 eval seeds, 20 episodes/task | 50000 | 6 | 180 | 0.310 | 0.947 | 0.947 | 0.637 | True |
| antmaze-teleport-stitch-v0 | full-goal BC + body-nearest k16 | 3 eval seeds, 20 episodes/task | 20000 | 6 | 180 | 0.317 | 0.960 | 0.960 | 0.643 | True |

Main signal: on both PointMaze and AntMaze teleport long-horizon tasks, stochastic TRL reaches the 180-sweep Bellman reference at the matched 6-sweep budget, while matched Bellman remains substantially lower. PointMaze rows use five evaluation seeds; AntMaze rows use the learned BC executor with three evaluation seeds and 20 episodes per task.

## Hard-Task Stress Checks

| env | task_scope | eval_setting | controller_steps | bellman_matched | support_trl | stochastic_trl | bellman_full | sto_minus_matched | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pointmaze-teleport-stitch-v0 | tasks 4,5 | seed 0, 50 episodes/task | 0 | 0.420 | 0.480 | 0.930 | 0.930 | 0.510 | results/pointmaze_stitch_hard_task45_ep50_seed0_fastfocus.csv |
| antmaze-teleport-navigate-v0 | tasks 4,5 | seed 0, 10 episodes/task | 50000 | 0.300 |  | 0.950 | 0.950 | 0.650 | results/antmaze_navigate_hard_task45_ep10_seed0_fastfocus.csv |
| antmaze-teleport-stitch-v0 | tasks 4,5 | seed 0, 10 episodes/task | 20000 | 0.350 |  | 1.000 | 1.000 | 0.650 | results/antmaze_stitch_hard_task45_ep10_seed0_fastfocus.csv |

Stress-check signal: on a single eval seed focused on harder long-horizon task slices, stochastic TRL reaches 0.93 on PointMaze teleport stitch tasks 4-5, 0.95 on AntMaze navigate tasks 4-5, and 1.00 on AntMaze stitch tasks 4-5. These rows are not a replacement for the main multi-seed table; they are fast iteration evidence for the hardest task slices.

## Focused PointMaze Single-Task Check

| env | task_id | eval_setting | matched_sweeps | full_sweeps | bellman_matched | stochastic_trl | bellman_full | sto_minus_matched | sto_equals_full | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pointmaze-teleport-stitch-v0 | 5 | 5 eval seeds, 100 episodes | 6 | 180 | 0.380 | 0.908 | 0.908 | 0.528 | True | results/pointmaze_stitch_task5_ep100_seed01234.csv |

Focused signal: on PointMaze teleport stitch task 5, stochastic TRL reaches 0.908 success over five evaluation seeds and 100 episodes per seed, matching the 180-sweep Bellman reference while matched Bellman reaches 0.380. This is appendix evidence, not a replacement for the all-task headline table.

## PointMaze Learned-Transition Screen

| env | transition_model | eval_setting | transition_seeds | matched_sweeps | full_sweeps | bellman_matched | stochastic_trl | bellman_full | sto_minus_matched | oracle_l1_range | oracle_top1_range | sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pointmaze-teleport-navigate-v0 | dataset cell-change softmax | seed 0, 50 episodes/task | 0 | 6 | 180 | 0.408 | 0.916 | 0.916 | 0.508 | 0.048-0.048 | 0.978-0.978 | results/pointmaze_learned_transition_navigate_cellchanges_1k_ep50_seed0.csv |
| pointmaze-teleport-stitch-v0 | dataset cell-change softmax | seed 0, 50 episodes/task | 0 | 6 | 180 | 0.408 | 0.916 | 0.916 | 0.508 | 0.048-0.048 | 0.978-0.978 | results/pointmaze_learned_transition_stitch_cellchanges_1k_ep50_seed0.csv |
| pointmaze-teleport-navigate-v0 | raw-observation MLP cell-change | seed 0, 3 transition seeds, 50 episodes/task | 0,1,2 | 6 | 180 | 0.512 | 0.916 | 0.916 | 0.404 | 0.048-0.053 | 0.933-0.978 | results/pointmaze_navigate_rawobs_mlp_cellchanges_ep50_seed0.csv;results/pointmaze_navigate_rawobs_mlp_cellchanges_ep50_tseed1.csv;results/pointmaze_navigate_rawobs_mlp_cellchanges_ep50_tseed2.csv |
| pointmaze-teleport-stitch-v0 | raw-observation MLP cell-change | seed 0, 3 transition seeds, 50 episodes/task | 0,1,2 | 6 | 180 | 0.483 | 0.916 | 0.916 | 0.433 | 0.050-0.052 | 0.933-0.956 | results/pointmaze_stitch_rawobs_mlp_cellchanges_ep50_seed0.csv;results/pointmaze_stitch_rawobs_mlp_cellchanges_ep50_tseed1.csv;results/pointmaze_stitch_rawobs_mlp_cellchanges_ep50_tseed2.csv |

Learned-transition signal: fitting either a table-softmax transition model or a shared raw-observation MLP transition head from collapsed offline cell changes preserves the stochastic TRL gain on PointMaze teleport navigate and stitch. Stochastic TRL reaches 0.916 success with the 6-sweep matched budget in both cases; matched Bellman reaches 0.408 with the table-softmax model and 0.512/0.483 on navigate/stitch with the raw-observation MLP.

## PointMaze Learned-Controller Screen

| env | controller | eval_setting | eval_seeds | controller_steps | matched_sweeps | full_sweeps | bellman_matched | stochastic_trl | bellman_full | sto_minus_matched | sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pointmaze-teleport-navigate-v0 | 5k full-goal BC + body-nearest k16 | 3 eval seeds, 20 episodes/task | 0,1,2 | 5000 | 6 | 180 | 0.323 | 1.000 | 1.000 | 0.677 | results/pointmaze_bc_topology_navigate_fullgoal_5k_ep20_seed0.csv;results/pointmaze_bc_topology_navigate_fullgoal_5k_ep20_seed12.csv |
| pointmaze-teleport-stitch-v0 | 5k full-goal BC + body-nearest k16 | 3 eval seeds, 20 episodes/task | 0,1,2 | 5000 | 6 | 180 | 0.223 | 1.000 | 1.000 | 0.777 | results/pointmaze_bc_topology_stitch_fullgoal_5k_ep20_seed0.csv;results/pointmaze_bc_topology_stitch_fullgoal_5k_ep20_seed12.csv |

Learned-controller signal: with saved 5k-step full-goal BC executors and body-nearest waypoint goals, stochastic TRL reaches 1.000 success on both PointMaze teleport navigate and stitch over three evaluation seeds, matching full Bellman while matched Bellman remains far lower. This is appendix evidence that the PointMaze result is not tied to the simple topology executor.

### Controller Execution Isolation

| execution_path | env | controller_or_actor | eval_setting | best_success | final_success | source |
| --- | --- | --- | --- | --- | --- | --- |
| GCFBC direct final-goal actor | pointmaze-teleport-navigate-v0 | GCFBC 128,128 | seed 0, 5 episodes/task | 0.080 | 0.000 | results/ogbench_screen_exp/dummy/pointmaze_gcfbc_10k_actor_diag_cpu/sd000_20260619_104221/eval.csv |
| MSEBC direct final-goal actor | pointmaze-teleport-navigate-v0 | MSEBC LN 256,256 | seed 0, 5 episodes/task | 0.120 | 0.120 | results/ogbench_screen_exp/dummy/pointmaze_msebc_10k_ln256_actor_diag_cpu/sd000_20260619_104714/eval.csv |
| Stochastic TRL + learned waypoint BC | pointmaze-teleport-navigate-v0 | 5k full-goal BC + body-nearest k16 | 3 eval seeds, 20 episodes/task | 1.000 | 1.000 | results/pointmaze_bc_topology_navigate_fullgoal_5k_ep20_seed0.csv;results/pointmaze_bc_topology_navigate_fullgoal_5k_ep20_seed12.csv |
| Stochastic TRL + learned waypoint BC | pointmaze-teleport-stitch-v0 | 5k full-goal BC + body-nearest k16 | 3 eval seeds, 20 episodes/task | 1.000 | 1.000 | results/pointmaze_bc_topology_stitch_fullgoal_5k_ep20_seed0.csv;results/pointmaze_bc_topology_stitch_fullgoal_5k_ep20_seed12.csv |

Controller-isolation signal: short direct final-goal actor screens in the official loop remain at or below 0.120 success on PointMaze teleport navigate, while the learned waypoint BC executor reaches 1.000 when driven by stochastic TRL waypoints on both navigate and stitch. This isolates the current direct-actor bottleneck from the high-level stochastic value-propagation result.

### Fast Evaluation Profile

| screen | role | env | task_ids | seed | episodes_per_task | eval_action_repeat | matched_success | stochastic_trl | sto_minus_matched | setup_seconds | matched_eval_seconds | sto_eval_seconds | sto_env_step_ms | sto_action_ms_per_call | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze navigate hard slice | recommended fast screen | antmaze-teleport-navigate-v0 | 4,5 | 0 | 5 | 1 | 0.400 | 0.900 | 0.500 | 1.09 | 4.56 | 4.85 | 0.334 | 0.342 | results/antmaze_navigate_fast_profile_repeat1_ep5_seed0_task45.csv |
| antmaze stitch hard slice | recommended fast screen | antmaze-teleport-stitch-v0 | 4,5 | 0 | 5 | 1 | 0.600 | 1.000 | 0.400 | 1.15 | 3.65 | 4.66 | 0.344 | 0.359 | results/antmaze_stitch_fast_profile_repeat1_ep5_seed0_task45.csv |
| antmaze stitch hard slice | two-episode baseline | antmaze-teleport-stitch-v0 | 4,5 | 0 | 2 | 1 | 0.500 | 1.000 | 0.500 | 1.09 | 1.78 | 2.03 | 0.362 | 0.397 | results/antmaze_stitch_fast_profile_repeat1_ep2_seed0_task45.csv |
| antmaze stitch hard slice | action-repeat ablation | antmaze-teleport-stitch-v0 | 4,5 | 0 | 2 | 2 | 0.500 | 0.750 | 0.250 | 1.09 | 1.50 | 1.96 | 0.314 | 0.449 | results/antmaze_stitch_fast_profile_repeat2_ep2_seed0_task45.csv |

Fast-eval signal: cached topology and saved BC policies make single-seed hard-slice AntMaze screening cheap. With `--task-ids 4 5`, `--episodes 5`, and `--methods bellman_matched sto_trl_matched`, navigate reaches 0.900 stochastic TRL success and stitch reaches 1.000 in roughly five seconds per stochastic evaluation row. Increasing `--eval-action-repeat` to 2 is not claim-safe in the current stitch screen: it slightly reduces policy calls but drops stochastic TRL success from 1.000 to 0.750.

Empirical graph speed note: `scripts/run_pointmaze_graph_planner.py` caches solved graph Q tables and can profile rollouts with `--profile-eval`. On `pointmaze-teleport-stitch-v0` task 4, the first 220-sweep Bellman graph solve took `11.09s`, while a cache hit took `0.03s`; a failed 1000-step rollout took `0.39s`. The learned BC graph executor was faster per step than `transition_value`, but still failed stitch task 4 in the 10-episode seed-0 screen; see `results/pointmaze_graph_summary.md`.

### AntMaze Deterministic-Support Ablation

| env | controller | eval_setting | matched_sweeps | full_sweeps | bellman_matched | support_trl | stochastic_trl | bellman_full | support_minus_matched | sto_minus_support | sto_eval_seconds | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-teleport-navigate-v0 | 50k full-goal BC + body-nearest k16 | seed 0, tasks 4,5, 5 episodes/task | 6 | 180 | 0.400 | 0.400 | 0.900 | 0.900 | 0.000 | 0.500 | 4.80 | results/antmaze_navigate_support_ablation_ep5_seed0_task45.csv |
| antmaze-teleport-stitch-v0 | 20k full-goal BC + body-nearest k16 | seed 0, tasks 4,5, 5 episodes/task | 6 | 180 | 0.600 | 0.600 | 0.900 | 1.000 | 0.000 | 0.300 | 4.99 | results/antmaze_stitch_support_ablation_ep5_seed0_task45.csv |

Ablation signal: on AntMaze hard tasks, optimistic support-only TRL matches the low matched-Bellman success in these fast slices, while stochastic TRL reaches 0.900 on both navigate and stitch. This isolates the AntMaze gain from mere transitive reachability over observed teleport support.

## PointMaze Tie-Preserving Policy-Head Screen

| env | transition_model | control_head | eval_setting | eval_seed | matched_sweeps | full_sweeps | bellman_matched | stochastic_trl | bellman_full | sto_minus_matched | transition_top1 | value_action_agreement | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pointmaze-teleport-navigate-v0 | raw-observation MLP cell-change | raw-observation tie-policy MLP | seed 0, all tasks, 20 episodes/task | 0 | 6 | 180 | 0.530 | 0.980 | 0.980 | 0.450 | 1.000 | 0.982 | results/pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv |
| pointmaze-teleport-stitch-v0 | raw-observation MLP cell-change | raw-observation tie-policy MLP | seed 0, all tasks, 20 episodes/task | 0 | 6 | 180 | 0.530 | 0.980 | 0.980 | 0.450 | 0.959 | 0.978 | results/pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv |

Neural-head signal: on PointMaze teleport navigate and stitch, a raw-observation MLP transition model plus a tie-preserving raw-observation policy head recovers 0.980 stochastic TRL success and matches the full-Bellman reference, while matched Bellman reaches 0.530. This is a positive value/control-head diagnostic over the cell abstraction, not a complete end-to-end neural TRL result.

### PointMaze Previous-Action Policy-Head Screen

| env | transition_model | control_head | eval_setting | eval_seed | value_steps | matched_sweeps | full_sweeps | bellman_matched | stochastic_trl | bellman_full | sto_minus_matched | transition_top1 | value_action_agreement | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pointmaze-teleport-navigate-v0 | raw-observation MLP cell-change | previous-action policy MLP | seed 0, all tasks, 20 episodes/task | 0 | 1000 | 6 | 180 | 0.530 | 0.980 | 0.980 | 0.450 | 1.000 | 1.000 | results/pointmaze_navigate_rawobs_mlp_transition_prev_policy_ep20_seed0.csv |
| pointmaze-teleport-stitch-v0 | raw-observation MLP cell-change | previous-action policy MLP | seed 0, all tasks, 20 episodes/task | 0 | 1000 | 6 | 180 | 0.530 | 0.980 | 0.980 | 0.450 | 0.959 | 1.000 | results/pointmaze_stitch_rawobs_mlp_transition_prev_policy_ep20_seed0.csv |

Previous-action signal: making the previous high-level action explicit lets a conventional single-label raw-observation policy head reproduce the successful PointMaze table policy on both teleport variants. With the raw-observation MLP transition head and 20 episodes per task, stochastic TRL reaches 0.980 on navigate and stitch, matching full Bellman while matched Bellman reaches 0.530. The shared evaluator now applies the same sticky previous-action tie-break to 4D previous-action policy scores; without that tie-break, the learned head could drift out of the goal cell after arrival.

### PointMaze Tie-Preserving Policy-Head Eval-Seed Screen

| env | transition_model | control_head | eval_setting | eval_seeds | matched_sweeps | full_sweeps | bellman_matched | stochastic_trl | bellman_full | sto_minus_matched | transition_top1 | value_action_agreement_min | sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pointmaze-teleport-navigate-v0 | raw-observation MLP cell-change | raw-observation tie-policy MLP | 3 eval seeds, all tasks, 20 episodes/task | 0,1,2 | 6 | 180 | 0.417 | 0.927 | 0.927 | 0.510 | 1.000 | 0.982 | results/pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv;results/pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_evalseed12.csv |
| pointmaze-teleport-stitch-v0 | raw-observation MLP cell-change | raw-observation tie-policy MLP | 3 eval seeds, all tasks, 20 episodes/task | 0,1,2 | 6 | 180 | 0.417 | 0.927 | 0.927 | 0.510 | 0.959 | 0.978 | results/pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv;results/pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_evalseed12.csv |

Eval-seed signal: with transition seed 0 fixed, the same PointMaze raw-observation transition plus tie-policy head reaches 0.927 stochastic TRL success on both navigate and stitch across three evaluation seeds, matching full Bellman while matched Bellman reaches 0.417.

### PointMaze Tie-Preserving Policy-Head Transition-Seed Screen

| env | transition_model | control_head | eval_setting | eval_seed | transition_seeds | matched_sweeps | full_sweeps | bellman_matched | stochastic_trl | bellman_full | sto_minus_matched | transition_top1_min | value_action_agreement_min | sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pointmaze-teleport-navigate-v0 | raw-observation MLP cell-change | raw-observation tie-policy MLP | seed 0, 3 transition seeds, all tasks, 20 episodes/task | 0 | 0,1,2 | 6 | 180 | 0.530 | 0.980 | 0.980 | 0.450 | 0.918 | 0.978 | results/pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv;results/pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_tseed1.csv;results/pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_tseed2.csv |
| pointmaze-teleport-stitch-v0 | raw-observation MLP cell-change | raw-observation tie-policy MLP | seed 0, 3 transition seeds, all tasks, 20 episodes/task | 0 | 0,1,2 | 6 | 180 | 0.530 | 0.980 | 0.980 | 0.450 | 0.918 | 0.978 | results/pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv;results/pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_tseed1.csv;results/pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_tseed2.csv |

Transition-seed signal: with evaluation seed 0 fixed, the PointMaze raw-observation transition plus tie-policy head reaches 0.980 stochastic TRL success on both navigate and stitch across three transition seeds, matching full Bellman while matched Bellman reaches 0.530.

## AntMaze Tie-Preserving Policy-Head Screen

| env | controller | control_head | eval_setting | eval_seed | transition_seeds | task_ids | controller_steps | matched_sweeps | full_sweeps | bellman_matched | stochastic_trl | bellman_full | sto_minus_matched | value_action_agreement | value_steps | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-teleport-navigate-v0 | 50k full-goal BC + body-nearest k16 | raw-observation tie-policy MLP | seed 0, tasks 4-5, 20 episodes/task | 0 |  | 4,5 | 50000 | 6 | 180 | 0.350 | 0.925 | 0.925 | 0.575 | 1.000 | 2000 | results/antmaze_navigate_tie_policy_head_hard_tasks_ep20_seed0.csv |
| antmaze-teleport-stitch-v0 | 20k full-goal BC + body-nearest k16 | raw-observation tie-policy MLP | seed 0, tasks 4-5, 20 episodes/task | 0 |  | 4,5 | 20000 | 6 | 180 | 0.400 | 0.950 | 0.950 | 0.550 | 1.000 | 2000 | results/antmaze_stitch_tie_policy_head_hard_tasks_ep20_seed0.csv |

Neural-head signal: on AntMaze hard tasks with the learned BC executor and topology transition model, a raw-observation tie-preserving high-level policy head keeps stochastic TRL high at 0.925 on navigate and 0.950 on stitch, while matched Bellman reaches 0.350 and 0.400. This is a single-seed hard-task diagnostic, not a replacement for the multi-seed AntMaze topology table.

## AntMaze Raw-Observation Transition And Policy-Head Screen

| env | controller | transition_model | control_head | eval_setting | eval_seed | transition_seeds | task_ids | controller_steps | matched_sweeps | full_sweeps | bellman_matched | stochastic_trl | bellman_full | sto_minus_matched | transition_oracle_l1_max | transition_oracle_top1_min | value_action_agreement_min | sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-teleport-navigate-v0 | 50k full-goal BC + body-nearest k16 | raw-observation MLP jump-change | raw-observation tie-policy MLP | seed 0, tasks 4-5, 10 episodes/task | 0 | 0,1,2 | 4,5 | 50000 | 6 | 180 | 0.300 | 0.950 | 0.950 | 0.650 | 0.000 | 0.956 | 0.999 | results/antmaze_navigate_rawobs_transition_tie_policy_head_ep10_seed0.csv;results/antmaze_navigate_rawobs_transition_tie_policy_head_ep10_tseed1.csv;results/antmaze_navigate_rawobs_transition_tie_policy_head_ep10_tseed2.csv |
| antmaze-teleport-stitch-v0 | 20k full-goal BC + body-nearest k16 | raw-observation MLP jump-change | raw-observation tie-policy MLP | seed 0, tasks 4-5, 10 episodes/task | 0 | 0,1,2 | 4,5 | 20000 | 6 | 180 | 0.300 | 1.000 | 1.000 | 0.700 | 0.000 | 1.000 | 0.999 | results/antmaze_stitch_rawobs_transition_tie_policy_head_ep10_seed0.csv;results/antmaze_stitch_rawobs_transition_tie_policy_head_ep10_tseed1.csv;results/antmaze_stitch_rawobs_transition_tie_policy_head_ep10_tseed2.csv |

Combined learned-module signal: on AntMaze hard tasks with a learned BC executor, a raw-observation MLP jump-change transition head plus a raw-observation tie-policy head reaches 0.950 stochastic TRL success on navigate and 1.000 on stitch across three transition seeds, matching the full-Bellman reference while matched Bellman reaches 0.300 on both tasks. This is still a high-level cell-abstraction diagnostic, but it removes the table transition and table policy from the AntMaze screen.

### AntMaze Combined Learned-Module Eval-Seed Screen

| env | controller | transition_model | control_head | eval_setting | eval_seeds | transition_seed | task_ids | controller_steps | matched_sweeps | full_sweeps | bellman_matched | stochastic_trl | bellman_full | sto_minus_matched | transition_oracle_top1 | value_action_agreement_min | sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-teleport-navigate-v0 | 50k full-goal BC + body-nearest k16 | raw-observation MLP jump-change | raw-observation tie-policy MLP | 3 eval seeds, tasks 4-5, 10 episodes/task | 0,1,2 | 0 | 4,5 | 50000 | 6 | 180 | 0.283 | 0.933 | 0.933 | 0.650 | 1.000 | 0.999 | results/antmaze_navigate_rawobs_transition_tie_policy_head_ep10_seed0.csv;results/antmaze_navigate_rawobs_transition_tie_policy_head_ep10_tseed0_evalseed12.csv |
| antmaze-teleport-stitch-v0 | 20k full-goal BC + body-nearest k16 | raw-observation MLP jump-change | raw-observation tie-policy MLP | 3 eval seeds, tasks 4-5, 10 episodes/task | 0,1,2 | 0 | 4,5 | 20000 | 6 | 180 | 0.283 | 0.967 | 0.967 | 0.683 | 1.000 | 0.999 | results/antmaze_stitch_rawobs_transition_tie_policy_head_ep10_seed0.csv;results/antmaze_stitch_rawobs_transition_tie_policy_head_ep10_tseed0_evalseed12.csv |

Eval-seed signal: with transition seed 0 fixed, the same raw-observation transition plus tie-policy head reaches 0.933 stochastic TRL success on navigate and 0.967 on stitch across three evaluation seeds, matching full Bellman while matched Bellman remains below 0.300.

### AntMaze Previous-Action Policy-Head Eval-Seed Screen

| env | controller | transition_model | control_head | eval_setting | eval_seeds | transition_seed | task_ids | controller_steps | matched_sweeps | full_sweeps | bellman_matched | stochastic_trl | bellman_full | sto_minus_matched | transition_oracle_top1 | value_action_agreement_min | sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-teleport-navigate-v0 | 50k full-goal BC + body-nearest k16 | raw-observation MLP jump-change | previous-action policy MLP | 3 eval seeds, tasks 4-5, 10 episodes/task | 0,1,2 | 0 | 4,5 | 50000 | 6 | 180 | 0.350 | 0.933 | 0.933 | 0.583 | 1.000 | 1.000 | results/antmaze_navigate_rawobs_transition_prev_policy_head_ep10_seed012.csv |
| antmaze-teleport-stitch-v0 | 20k full-goal BC + body-nearest k16 | raw-observation MLP jump-change | previous-action policy MLP | 3 eval seeds, tasks 4-5, 10 episodes/task | 0,1,2 | 0 | 4,5 | 20000 | 6 | 180 | 0.283 | 0.967 | 0.950 | 0.683 | 0.994 | 1.000 | results/antmaze_stitch_rawobs_transition_prev_policy_head_ep10_seed012.csv |

Previous-action signal: making the previous high-level action explicit lets a single-label raw-observation policy head preserve the stochastic TRL advantage across three evaluation seeds on AntMaze hard tasks. With a raw-observation jump-change transition head and 10 episodes per hard task, stochastic TRL reaches 0.933 on navigate and 0.967 on stitch, within 0.02 of the full-Bellman reference while matched Bellman remains at or below 0.350.

## AntMaze Learned-Transition Screens

| env | controller | transition_model | eval_setting | transition_seeds | matched_sweeps | full_sweeps | bellman_matched | stochastic_trl | bellman_full | sto_minus_matched | oracle_l1_range | oracle_top1_range | sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-teleport-navigate-v0 | 50k BC | table softmax, 20 samples/row | seed 0, tasks 4-5, 10 episodes/task | 0,1,2 | 6 | 180 | 0.300 | 0.950 | 0.950 | 0.650 | 0.014-0.015 | 0.961-0.978 | results/antmaze_navigate_learned_transition_samples20_ep10_tseed0.csv;results/antmaze_navigate_learned_transition_samples20_ep10_tseed1.csv;results/antmaze_navigate_learned_transition_samples20_ep10_tseed2.csv |
| antmaze-teleport-stitch-v0 | 20k BC | table softmax, 20 samples/row | seed 0, tasks 4-5, 10 episodes/task | 0,1,2 | 6 | 180 | 0.367 | 1.000 | 1.000 | 0.633 | 0.013-0.015 | 0.939-0.961 | results/antmaze_stitch_learned_transition_samples20_ep10_tseed0.csv;results/antmaze_stitch_learned_transition_samples20_ep10_tseed1.csv;results/antmaze_stitch_learned_transition_samples20_ep10_tseed2.csv |
| antmaze-teleport-navigate-v0 | 50k BC | raw-observation MLP jump-change | seed 0, tasks 4-5, 10 episodes/task | 0,1,2 | 6 | 180 | 0.300 | 0.950 | 0.950 | 0.650 | 0.000-0.000 | 0.956-1.000 | results/antmaze_navigate_rawobs_mlp_jumpchanges_ep10_seed0.csv;results/antmaze_navigate_rawobs_mlp_jumpchanges_ep10_tseed1.csv;results/antmaze_navigate_rawobs_mlp_jumpchanges_ep10_tseed2.csv |
| antmaze-teleport-stitch-v0 | 20k BC | raw-observation MLP jump-change | seed 0, tasks 4-5, 10 episodes/task | 0,1,2 | 6 | 180 | 0.300 | 1.000 | 1.000 | 0.700 | 0.000-0.000 | 1.000-1.000 | results/antmaze_stitch_rawobs_mlp_jumpchanges_ep10_seed0.csv;results/antmaze_stitch_rawobs_mlp_jumpchanges_ep10_tseed1.csv;results/antmaze_stitch_rawobs_mlp_jumpchanges_ep10_tseed2.csv |

Learned-transition signal: on AntMaze hard tasks with a learned BC executor, three independent table-softmax transition-model seeds keep stochastic TRL at 0.950 success on navigate and 1.000 on stitch. Three raw-observation MLP jump-change transition seeds reach the same 0.950 and 1.000 success. These are learned high-level transition screens, not a replacement for the main multi-evaluation-seed topology table.

## Long-Horizon Tabular Safe-Optimal Cases

| p_success | optimal_action | method | n | success_rate | risky_action_rate | regret | pred_safe | pred_risky | long_horizon_mse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.02 | safe | MC positive | 3 | 0.017 | 1.000 | 0.056 | 0.073 | 0.960 | 0.02155 |
| 0.02 | safe | Bellman matched | 3 | 0.017 | 1.000 | 0.056 | 0.000 | 0.018 | 0.21682 |
| 0.02 | safe | Stochastic TRL | 3 | 1.000 | 0.000 | 0.000 | 0.075 | 0.018 | 0.00000 |
| 0.05 | safe | MC positive | 3 | 0.038 | 1.000 | 0.027 | 0.073 | 0.960 | 0.02155 |
| 0.05 | safe | Bellman matched | 3 | 0.038 | 1.000 | 0.027 | 0.000 | 0.033 | 0.21682 |
| 0.05 | safe | Stochastic TRL | 3 | 1.000 | 0.000 | 0.000 | 0.075 | 0.033 | 0.00000 |

Main signal: when the risky shortcut is suboptimal, stochastic TRL reaches success 1.0 with the matched transitive sweep budget, while matched Bellman and realized MC-positive baselines choose the risky shortcut.

## Horizon Scaling

| safe_length | p_success | method | n | matched_sweeps | success_rate | risky_action_rate | regret | long_horizon_mse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 0.02 | MC positive | 5 |  | 0.018 | 1.000 | 0.705 | 0.02257 |
| 16 | 0.02 | MC all goals | 5 |  | 1.000 | 0.000 | 0.000 | 0.00313 |
| 16 | 0.02 | Bellman matched | 5 | 5 | 0.018 | 1.000 | 0.705 | 0.63138 |
| 16 | 0.02 | Stochastic TRL | 5 | 5 | 1.000 | 0.000 | 0.000 | 0.00000 |
| 16 | 0.02 | Bellman full | 5 |  | 1.000 | 0.000 | 0.000 | 0.00000 |
| 16 | 0.05 | MC positive | 5 |  | 0.045 | 1.000 | 0.676 | 0.02142 |
| 16 | 0.05 | MC all goals | 5 |  | 1.000 | 0.000 | 0.000 | 0.00315 |
| 16 | 0.05 | Bellman matched | 5 | 5 | 0.045 | 1.000 | 0.676 | 0.63137 |
| 16 | 0.05 | Stochastic TRL | 5 | 5 | 1.000 | 0.000 | 0.000 | 0.00000 |
| 16 | 0.05 | Bellman full | 5 |  | 1.000 | 0.000 | 0.000 | 0.00000 |
| 32 | 0.02 | MC positive | 5 |  | 0.021 | 1.000 | 0.505 | 0.01725 |
| 32 | 0.02 | MC all goals | 5 |  | 1.000 | 0.000 | 0.000 | 0.01448 |
| 32 | 0.02 | Bellman matched | 5 | 6 | 0.021 | 1.000 | 0.505 | 0.53627 |
| 32 | 0.02 | Stochastic TRL | 5 | 6 | 1.000 | 0.000 | 0.000 | 0.00000 |
| 32 | 0.02 | Bellman full | 5 |  | 1.000 | 0.000 | 0.000 | 0.00000 |
| 32 | 0.05 | MC positive | 5 |  | 0.047 | 1.000 | 0.476 | 0.01709 |
| 32 | 0.05 | MC all goals | 5 |  | 1.000 | 0.000 | 0.000 | 0.01448 |
| 32 | 0.05 | Bellman matched | 5 | 6 | 0.047 | 1.000 | 0.476 | 0.53627 |
| 32 | 0.05 | Stochastic TRL | 5 | 6 | 1.000 | 0.000 | 0.000 | 0.00000 |
| 32 | 0.05 | Bellman full | 5 |  | 1.000 | 0.000 | 0.000 | 0.00000 |
| 64 | 0.02 | MC positive | 5 |  | 0.023 | 1.000 | 0.255 | 0.01783 |
| 64 | 0.02 | MC all goals | 5 |  | 1.000 | 0.000 | 0.000 | 0.01728 |
| 64 | 0.02 | Bellman matched | 5 | 7 | 0.023 | 1.000 | 0.255 | 0.36361 |
| 64 | 0.02 | Stochastic TRL | 5 | 7 | 1.000 | 0.000 | 0.000 | 0.00000 |
| 64 | 0.02 | Bellman full | 5 |  | 1.000 | 0.000 | 0.000 | 0.00000 |
| 64 | 0.05 | MC positive | 5 |  | 0.054 | 1.000 | 0.226 | 0.01784 |
| 64 | 0.05 | MC all goals | 5 |  | 1.000 | 0.000 | 0.000 | 0.01729 |
| 64 | 0.05 | Bellman matched | 5 | 7 | 0.054 | 1.000 | 0.226 | 0.36361 |
| 64 | 0.05 | Stochastic TRL | 5 | 7 | 1.000 | 0.000 | 0.000 | 0.00000 |
| 64 | 0.05 | Bellman full | 5 |  | 1.000 | 0.000 | 0.000 | 0.00000 |
| 128 | 0.02 | MC positive | 5 |  | 0.020 | 1.000 | 0.056 | 0.00957 |
| 128 | 0.02 | MC all goals | 5 |  | 0.200 | 0.000 | 0.060 | 0.00945 |
| 128 | 0.02 | Bellman matched | 5 | 8 | 0.020 | 1.000 | 0.056 | 0.21682 |
| 128 | 0.02 | Stochastic TRL | 5 | 8 | 1.000 | 0.000 | 0.000 | 0.00000 |
| 128 | 0.02 | Bellman full | 5 |  | 1.000 | 0.000 | 0.000 | 0.00000 |
| 128 | 0.05 | MC positive | 5 |  | 0.043 | 1.000 | 0.027 | 0.00957 |
| 128 | 0.05 | MC all goals | 5 |  | 0.200 | 0.000 | 0.060 | 0.00945 |
| 128 | 0.05 | Bellman matched | 5 | 8 | 0.043 | 1.000 | 0.027 | 0.21682 |
| 128 | 0.05 | Stochastic TRL | 5 | 8 | 1.000 | 0.000 | 0.000 | 0.00000 |
| 128 | 0.05 | Bellman full | 5 |  | 1.000 | 0.000 | 0.000 | 0.00000 |

Main signal: across safe path lengths 16, 32, 64, and 128, stochastic TRL keeps success 1.0 with only logarithmic matched sweeps, while matched Bellman and MC-positive baselines keep choosing the risky shortcut.

## Risky-Shortcut Aggregate

| method | n | success_rate | regret | long_horizon_mse | safe_opt_risky_rate | risky_opt_risky_rate | risky_overestimate_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MC positive | 27 | 0.436 | 0.315 | 0.09153 | 1.000 | 1.000 | 4.583 |
| MC all goals | 27 | 0.896 | 0.057 | 0.00688 | 0.000 | 1.000 | 1.012 |
| Realized TRL | 27 | 0.436 | 0.315 | 0.09151 | 1.000 | 1.000 | 4.583 |
| Log realized TRL | 27 | 0.436 | 0.315 | 0.11951 | 1.000 | 1.000 | 4.583 |
| Bellman matched | 27 | 0.436 | 0.315 | 0.45671 | 1.000 | 1.000 | 1.012 |
| Stochastic TRL | 27 | 0.955 | 0.000 | 0.00005 | 0.000 | 1.000 | 1.012 |
| Bellman full | 27 | 0.955 | 0.000 | 0.00005 | 0.000 | 1.000 | 1.012 |

Main signal: stochastic TRL matches the full Bellman solution on the stochastic shortcut family while realized-trajectory variants overestimate lucky risky outcomes.

## 2D Stochastic Grid Shortcut

| grid | safe_length | p_success | method | n | matched_sweeps | success_rate | safe_action_rate | portal_action_rate | regret | portal_overestimate_ratio | long_horizon_mse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8x4 | 31 | 0.02 | MC positive | 5 | 6 | 0.022 | 0.000 | 1.000 | 0.515 | 50.000 | 0.33661 |
| 8x4 | 31 | 0.02 | MC all goals | 5 | 6 | 1.000 | 1.000 | 0.000 | 0.000 | 0.928 | 0.33789 |
| 8x4 | 31 | 0.02 | Bellman matched | 5 | 6 | 0.022 | 0.000 | 1.000 | 0.515 | 0.928 | 0.53740 |
| 8x4 | 31 | 0.02 | Stochastic TRL | 5 | 6 | 1.000 | 1.000 | 0.000 | 0.000 | 0.928 | 0.26769 |
| 8x4 | 31 | 0.02 | Bellman full | 5 | 6 | 1.000 | 1.000 | 0.000 | 0.000 | 0.928 | 0.26769 |
| 8x4 | 31 | 0.05 | MC positive | 5 | 6 | 0.052 | 0.000 | 1.000 | 0.486 | 20.000 | 0.33606 |
| 8x4 | 31 | 0.05 | MC all goals | 5 | 6 | 1.000 | 1.000 | 0.000 | 0.000 | 0.920 | 0.33736 |
| 8x4 | 31 | 0.05 | Bellman matched | 5 | 6 | 0.052 | 0.000 | 1.000 | 0.486 | 0.920 | 0.53650 |
| 8x4 | 31 | 0.05 | Stochastic TRL | 5 | 6 | 1.000 | 1.000 | 0.000 | 0.000 | 0.920 | 0.26726 |
| 8x4 | 31 | 0.05 | Bellman full | 5 | 6 | 1.000 | 1.000 | 0.000 | 0.000 | 0.920 | 0.26726 |
| 16x4 | 63 | 0.02 | MC positive | 5 | 7 | 0.022 | 0.000 | 1.000 | 0.260 | 50.000 | 0.24407 |
| 16x4 | 63 | 0.02 | MC all goals | 5 | 7 | 0.800 | 0.800 | 0.000 | 0.056 | 0.915 | 0.24448 |
| 16x4 | 63 | 0.02 | Bellman matched | 5 | 7 | 0.022 | 0.000 | 1.000 | 0.260 | 0.915 | 0.37484 |
| 16x4 | 63 | 0.02 | Stochastic TRL | 5 | 7 | 1.000 | 1.000 | 0.000 | 0.000 | 0.915 | 0.19053 |
| 16x4 | 63 | 0.02 | Bellman full | 5 | 7 | 1.000 | 1.000 | 0.000 | 0.000 | 0.915 | 0.19053 |
| 16x4 | 63 | 0.05 | MC positive | 5 | 7 | 0.048 | 0.000 | 1.000 | 0.231 | 20.000 | 0.24390 |
| 16x4 | 63 | 0.05 | MC all goals | 5 | 7 | 0.800 | 0.800 | 0.000 | 0.056 | 0.937 | 0.24432 |
| 16x4 | 63 | 0.05 | Bellman matched | 5 | 7 | 0.048 | 0.000 | 1.000 | 0.231 | 0.937 | 0.37456 |
| 16x4 | 63 | 0.05 | Stochastic TRL | 5 | 7 | 1.000 | 1.000 | 0.000 | 0.000 | 0.937 | 0.19032 |
| 16x4 | 63 | 0.05 | Bellman full | 5 | 7 | 1.000 | 1.000 | 0.000 | 0.000 | 0.937 | 0.19032 |
| 16x8 | 127 | 0.02 | MC positive | 5 | 8 | 0.022 | 0.000 | 1.000 | 0.057 | 50.000 | 0.14916 |
| 16x8 | 127 | 0.02 | MC all goals | 5 | 8 | 0.000 | 1.000 | 0.000 | 0.077 | 0.996 | 0.14924 |
| 16x8 | 127 | 0.02 | Bellman matched | 5 | 8 | 0.022 | 0.000 | 1.000 | 0.057 | 0.996 | 0.22762 |
| 16x8 | 127 | 0.02 | Stochastic TRL | 5 | 8 | 1.000 | 1.000 | 0.000 | 0.000 | 0.996 | 0.11844 |
| 16x8 | 127 | 0.02 | Bellman full | 5 | 8 | 1.000 | 1.000 | 0.000 | 0.000 | 0.996 | 0.11844 |
| 16x8 | 127 | 0.05 | MC positive | 5 | 8 | 0.050 | 0.000 | 1.000 | 0.028 | 20.000 | 0.14913 |
| 16x8 | 127 | 0.05 | MC all goals | 5 | 8 | 0.000 | 1.000 | 0.000 | 0.077 | 0.994 | 0.14920 |
| 16x8 | 127 | 0.05 | Bellman matched | 5 | 8 | 0.050 | 0.000 | 1.000 | 0.028 | 0.994 | 0.22757 |
| 16x8 | 127 | 0.05 | Stochastic TRL | 5 | 8 | 1.000 | 1.000 | 0.000 | 0.000 | 0.994 | 0.11840 |
| 16x8 | 127 | 0.05 | Bellman full | 5 | 8 | 1.000 | 1.000 | 0.000 | 0.000 | 0.994 | 0.11840 |

Main signal: on 2D snake-corridor grids with safe path lengths 31, 63, and 127, stochastic TRL reaches success 1.0 with 6, 7, and 8 matched sweeps. Matched Bellman and MC-positive choose the risky portal and succeed only at the portal rate, while full Bellman also reaches success 1.0.

### Realized-TRL Diagnostic

| method | n | success_rate | safe_action_rate | portal_action_rate | regret | portal_overestimate_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| MC positive | 6 | 0.051 | 0.000 | 1.000 | 0.358 | 20.000 |
| Realized TRL | 6 | 0.051 | 0.000 | 1.000 | 0.358 | 20.000 |
| Log realized TRL | 6 | 0.051 | 0.000 | 1.000 | 0.358 | 20.000 |
| Bellman matched | 6 | 0.051 | 0.000 | 1.000 | 0.358 | 0.898 |
| Stochastic TRL | 6 | 1.000 | 1.000 | 0.000 | 0.000 | 0.898 |
| Bellman full | 6 | 1.000 | 1.000 | 0.000 | 0.000 | 0.898 |

Diagnostic signal: on the reduced 2D grid diagnostic with path lengths 31 and 63, raw/log realized TRL choose the risky portal and succeed at the portal rate, while stochastic TRL again matches full Bellman.

### Planning-Budget Curve

| method | sweeps | n | success_rate | safe_action_rate | portal_action_rate | regret | pred_safe | pred_portal | long_horizon_mse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bellman | 1 | 5 | 0.048 | 0.000 | 1.000 | 0.028 | 0.000 | 0.049 | 0.24026 |
| Bellman | 2 | 5 | 0.048 | 0.000 | 1.000 | 0.028 | 0.028 | 0.049 | 0.24026 |
| Bellman | 4 | 5 | 0.048 | 0.000 | 1.000 | 0.028 | 0.028 | 0.049 | 0.24026 |
| Bellman | 8 | 5 | 0.048 | 0.000 | 1.000 | 0.028 | 0.028 | 0.049 | 0.22757 |
| Bellman | 16 | 5 | 0.048 | 0.000 | 1.000 | 0.028 | 0.028 | 0.049 | 0.19030 |
| Bellman | 32 | 5 | 0.048 | 0.000 | 1.000 | 0.028 | 0.028 | 0.049 | 0.14918 |
| Bellman | 64 | 5 | 0.048 | 0.000 | 1.000 | 0.028 | 0.028 | 0.049 | 0.12321 |
| Bellman | 96 | 5 | 0.048 | 0.000 | 1.000 | 0.028 | 0.028 | 0.049 | 0.11884 |
| Bellman | 120 | 5 | 0.048 | 0.000 | 1.000 | 0.028 | 0.028 | 0.049 | 0.11841 |
| Bellman | 126 | 5 | 1.000 | 1.000 | 0.000 | 0.000 | 0.077 | 0.049 | 0.11840 |
| Bellman | 127 | 5 | 1.000 | 1.000 | 0.000 | 0.000 | 0.077 | 0.049 | 0.11840 |
| Bellman | 128 | 5 | 1.000 | 1.000 | 0.000 | 0.000 | 0.077 | 0.049 | 0.11840 |
| Stochastic TRL | 1 | 5 | 0.048 | 0.000 | 1.000 | 0.028 | 0.000 | 0.049 | 0.24026 |
| Stochastic TRL | 2 | 5 | 0.048 | 0.000 | 1.000 | 0.028 | 0.028 | 0.049 | 0.24026 |
| Stochastic TRL | 4 | 5 | 0.048 | 0.000 | 1.000 | 0.028 | 0.028 | 0.049 | 0.19474 |
| Stochastic TRL | 6 | 5 | 0.048 | 0.000 | 1.000 | 0.028 | 0.028 | 0.049 | 0.12357 |
| Stochastic TRL | 7 | 5 | 1.000 | 1.000 | 0.000 | 0.000 | 0.077 | 0.049 | 0.11840 |
| Stochastic TRL | 8 | 5 | 1.000 | 1.000 | 0.000 | 0.000 | 0.077 | 0.049 | 0.11840 |

Budget signal: on the hardest 16x8 grid with safe path length 127, stochastic TRL reaches success 1.0 after 7 sweeps. Bellman remains at the portal success rate through 120 sweeps and first reaches success 1.0 at 126 sweeps.

## PointMaze Empirical Graph Planner

### Topology-Level Planner

| method | sweeps | n | mean_success | std_success | task1_mean | task2_mean | task3_mean | task4_mean | task5_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bellman matched | 6 | 5 | 0.343 | 0.039 | 0.000 | 0.460 | 0.440 | 0.464 | 0.352 |
| Stochastic TRL | 6 | 5 | 0.901 | 0.023 | 0.892 | 0.912 | 0.876 | 0.908 | 0.916 |
| Bellman full | 180 | 5 | 0.901 | 0.023 | 0.892 | 0.912 | 0.876 | 0.908 | 0.916 |

| comparison | n_seeds | mean_diff | sample_sd_diff | sem_diff | ci95_low | ci95_high | min_seed_diff | max_seed_diff | all_seed_diffs_nonnegative | relative_gain_vs_right |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stochastic TRL - Bellman matched | 5 | 0.558 | 0.045 | 0.020 | 0.501 | 0.614 | 0.492 | 0.604 | True | 1.625 |
| Bellman full - Stochastic TRL | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | 0.000 |

High-success signal: with a coarse topology inferred from the offline dataset as the low-level routing scaffold, stochastic TRL reaches mean success 0.901 with the 6-sweep matched budget and matches the 180-sweep Bellman reference. Matched Bellman with the same 6 sweeps reaches 0.343.

### Topology-Level Teleport Stitch Planner

| method | sweeps | n | mean_success | std_success | task1_mean | task2_mean | task3_mean | task4_mean | task5_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bellman matched | 6 | 5 | 0.343 | 0.039 | 0.000 | 0.460 | 0.440 | 0.464 | 0.352 |
| Stochastic TRL | 6 | 5 | 0.901 | 0.023 | 0.892 | 0.912 | 0.876 | 0.908 | 0.916 |
| Bellman full | 180 | 5 | 0.901 | 0.023 | 0.892 | 0.912 | 0.876 | 0.908 | 0.916 |

| comparison | n_seeds | mean_diff | sample_sd_diff | sem_diff | ci95_low | ci95_high | min_seed_diff | max_seed_diff | all_seed_diffs_nonnegative | relative_gain_vs_right |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stochastic TRL - Bellman matched | 5 | 0.558 | 0.045 | 0.020 | 0.501 | 0.614 | 0.492 | 0.604 | True | 1.625 |
| Bellman full - Stochastic TRL | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | 0.000 |

Stitch signal: on `pointmaze-teleport-stitch-v0`, stochastic TRL again reaches mean success 0.901 with the 6-sweep matched budget and matches the 180-sweep Bellman reference, while matched Bellman reaches 0.343.

### Topology-Level Stitch Deterministic-Support Ablation

| method | sweeps | n | mean_success | std_success | task1_mean | task2_mean | task3_mean | task4_mean | task5_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bellman matched | 6 | 5 | 0.343 | 0.039 | 0.000 | 0.460 | 0.440 | 0.464 | 0.352 |
| Support TRL | 6 | 5 | 0.449 | 0.049 | 0.484 | 0.460 | 0.440 | 0.464 | 0.396 |
| Stochastic TRL | 6 | 5 | 0.901 | 0.023 | 0.892 | 0.912 | 0.876 | 0.908 | 0.916 |
| Bellman full | 180 | 5 | 0.901 | 0.023 | 0.892 | 0.912 | 0.876 | 0.908 | 0.916 |

Ablation signal: optimistic support TRL treats every observed stochastic teleport outcome as a reliable transition. It improves over matched Bellman but reaches only 0.449 mean success, far below calibrated stochastic TRL at 0.901. This supports the claim that stochastic Bellman calibration, not transitive composition alone, is responsible for the high-success stitch result.

### Empirical Graph Planner

| method | sweeps | executor | seed0_success | seed1_success | seed2_success | seed3_success | seed4_success | mean_success | std_success | task1_mean | task2_mean | task3_mean | task4_mean | task5_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bellman matched | 11 | persistent_waypoint | 0.220 | 0.200 | 0.200 | 0.264 | 0.196 | 0.216 | 0.025 | 0.000 | 0.000 | 0.388 | 0.428 | 0.264 |
| Stochastic TRL | 11 | persistent_waypoint | 0.392 | 0.324 | 0.324 | 0.388 | 0.348 | 0.355 | 0.030 | 0.384 | 0.228 | 0.392 | 0.428 | 0.344 |
| Bellman full | 220 | persistent_waypoint | 0.452 | 0.360 | 0.360 | 0.436 | 0.400 | 0.402 | 0.038 | 0.436 | 0.420 | 0.392 | 0.428 | 0.332 |

Main signal: after fixing the evaluation seed schedule so the five seeds use disjoint rollout randomness, stochastic TRL reaches mean success 0.355 versus 0.216 for matched Bellman with the same 11-sweep planning budget. It approaches the 220-sweep full Bellman reference at 0.402.

### PointMaze Paired Statistics

| comparison | n_seeds | mean_diff | sample_sd_diff | sem_diff | ci95_low | ci95_high | min_seed_diff | max_seed_diff | all_seed_diffs_positive | relative_gain_vs_right | full_gap_recovery |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stochastic TRL - Bellman matched | 5 | 0.139 | 0.022 | 0.010 | 0.112 | 0.166 | 0.124 | 0.172 | True | 0.644 | 0.750 |
| Bellman full - Stochastic TRL | 5 | 0.046 | 0.010 | 0.005 | 0.033 | 0.059 | 0.036 | 0.060 | True | 0.131 | 0.250 |

The paired seed-level improvement over matched Bellman is positive for all five seeds. Stochastic TRL recovers about 75% of the gap between matched Bellman and the 220-sweep Bellman reference.

### PointMaze Task Deltas

| task | bellman_matched | sto_trl | bellman_full | sto_minus_matched | full_minus_sto |
| --- | --- | --- | --- | --- | --- |
| task1 | 0.000 | 0.384 | 0.436 | 0.384 | 0.052 |
| task2 | 0.000 | 0.228 | 0.420 | 0.228 | 0.192 |
| task3 | 0.388 | 0.392 | 0.392 | 0.004 | 0.000 |
| task4 | 0.428 | 0.428 | 0.428 | 0.000 | 0.000 |
| task5 | 0.264 | 0.344 | 0.332 | 0.080 | -0.012 |

Task-level means show that the matched-budget improvement comes mainly from tasks 1 and 2, where matched Bellman gets zero success, while tasks 3 and 4 are already reachable for matched Bellman.

## AntMaze Learned-Controller Diagnostic

| method | sweeps | n | mean_success | task1_mean | task2_mean | task3_mean | task4_mean | task5_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bellman matched | 6 | 1 | 0.333 | 0.000 | 0.000 | 0.333 | 0.667 | 0.667 |
| Stochastic TRL | 6 | 1 | 0.933 | 1.000 | 1.000 | 0.667 | 1.000 | 1.000 |

Preliminary signal: on `antmaze-teleport-navigate-v0`, stochastic TRL reaches mean success 0.933 with the matched 6-sweep inferred-topology budget and a shared 20k-step full-observation BC controller, while matched Bellman reaches 0.333. This is a seed-0, 3-episode-per-task smoke, not yet the final multi-seed AntMaze table.

### Navigate Body-Candidate Check

| method | sweeps | n | mean_success | task1_mean | task2_mean | task3_mean | task4_mean | task5_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bellman matched | 6 | 1 | 0.380 | 0.000 | 0.500 | 0.600 | 0.400 | 0.400 |
| Stochastic TRL | 6 | 1 | 0.940 | 1.000 | 1.000 | 0.900 | 0.900 | 0.900 |
| Bellman full | 180 | 1 | 0.940 | 1.000 | 1.000 | 0.900 | 0.900 | 0.900 |

Executor signal: with a saved 50k-step full-observation BC controller and 16 body-nearest candidate waypoint goals, stochastic TRL matches the 180-sweep Bellman reference at 0.940 success over 10 episodes per task while matched Bellman remains at 0.380.

### AntMaze Body-Candidate Multi-Seed Screen

| env | episodes_per_task | eval_seeds | controller_steps | method | sweeps | mean_success | success_std | task1_mean | task2_mean | task3_mean | task4_mean | task5_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-teleport-navigate-v0 | 5 | 3 | 50000 | Bellman matched | 6 | 0.347 | 0.068 | 0.000 | 0.467 | 0.467 | 0.533 | 0.267 |
| antmaze-teleport-navigate-v0 | 5 | 3 | 50000 | Stochastic TRL | 6 | 0.893 | 0.019 | 0.867 | 0.933 | 0.933 | 0.933 | 0.800 |
| antmaze-teleport-navigate-v0 | 5 | 3 | 50000 | Bellman full | 180 | 0.893 | 0.019 | 0.867 | 0.933 | 0.933 | 0.933 | 0.800 |
| antmaze-teleport-stitch-v0 | 5 | 3 | 20000 | Bellman matched | 6 | 0.333 | 0.075 | 0.000 | 0.400 | 0.467 | 0.533 | 0.267 |
| antmaze-teleport-stitch-v0 | 5 | 3 | 20000 | Stochastic TRL | 6 | 0.907 | 0.038 | 0.867 | 0.933 | 1.000 | 0.800 | 0.933 |
| antmaze-teleport-stitch-v0 | 5 | 3 | 20000 | Bellman full | 180 | 0.907 | 0.038 | 0.867 | 0.933 | 1.000 | 0.800 | 0.933 |

Multi-seed signal: across three evaluation seeds and five episodes per task, stochastic TRL matches the 180-sweep Bellman reference on both AntMaze navigate and stitch, while matched Bellman remains near 0.34 success.

### AntMaze Independent Controller-Seed Screen

| env | controller_seed | episodes_per_task | eval_seeds | controller_steps | method | sweeps | mean_success | success_std | task1_mean | task2_mean | task3_mean | task4_mean | task5_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-teleport-navigate-v0 | 1 | 20 | 3 | 50000 | Bellman matched | 6 | 0.320 | 0.043 | 0.000 | 0.483 | 0.383 | 0.417 | 0.317 |
| antmaze-teleport-navigate-v0 | 1 | 20 | 3 | 50000 | Stochastic TRL | 6 | 0.933 | 0.024 | 0.933 | 0.883 | 0.983 | 0.917 | 0.950 |
| antmaze-teleport-navigate-v0 | 1 | 20 | 3 | 50000 | Bellman full | 180 | 0.933 | 0.024 | 0.933 | 0.883 | 0.983 | 0.917 | 0.950 |
| antmaze-teleport-stitch-v0 | 1 | 20 | 3 | 20000 | Bellman matched | 6 | 0.323 | 0.042 | 0.000 | 0.500 | 0.383 | 0.417 | 0.317 |
| antmaze-teleport-stitch-v0 | 1 | 20 | 3 | 20000 | Stochastic TRL | 6 | 0.950 | 0.022 | 0.983 | 0.950 | 0.917 | 0.933 | 0.967 |
| antmaze-teleport-stitch-v0 | 1 | 20 | 3 | 20000 | Bellman full | 180 | 0.950 | 0.022 | 0.983 | 0.950 | 0.917 | 0.933 | 0.967 |

Controller-seed signal: with independently trained `bc_seed=1` AntMaze controllers, stochastic TRL again matches the 180-sweep Bellman reference over three evaluation seeds and 20 episodes per task. It reaches 0.933 on navigate and 0.950 on stitch, while matched Bellman remains near 0.32 on both tasks. This strengthens the AntMaze claim beyond evaluation-seed robustness for a single saved controller.

### AntMaze Stitch Controller-Seed Aggregate

| env | controller_seed | episodes_per_task | eval_seeds | controller_steps | method | sweeps | mean_success | success_std | task1_mean | task2_mean | task3_mean | task4_mean | task5_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-teleport-stitch-v0 | 0 | 20 | 3 | 20000 | Bellman matched | 6 | 0.317 | 0.049 | 0.000 | 0.467 | 0.383 | 0.417 | 0.317 |
| antmaze-teleport-stitch-v0 | 0 | 20 | 3 | 20000 | Stochastic TRL | 6 | 0.960 | 0.000 | 0.933 | 0.950 | 1.000 | 0.967 | 0.950 |
| antmaze-teleport-stitch-v0 | 0 | 20 | 3 | 20000 | Bellman full | 180 | 0.960 | 0.000 | 0.933 | 0.950 | 1.000 | 0.967 | 0.950 |
| antmaze-teleport-stitch-v0 | 1 | 20 | 3 | 20000 | Bellman matched | 6 | 0.323 | 0.042 | 0.000 | 0.500 | 0.383 | 0.417 | 0.317 |
| antmaze-teleport-stitch-v0 | 1 | 20 | 3 | 20000 | Stochastic TRL | 6 | 0.950 | 0.022 | 0.983 | 0.950 | 0.917 | 0.933 | 0.967 |
| antmaze-teleport-stitch-v0 | 1 | 20 | 3 | 20000 | Bellman full | 180 | 0.950 | 0.022 | 0.983 | 0.950 | 0.917 | 0.933 | 0.967 |
| antmaze-teleport-stitch-v0 | 2 | 20 | 3 | 20000 | Bellman matched | 6 | 0.323 | 0.042 | 0.000 | 0.500 | 0.383 | 0.417 | 0.317 |
| antmaze-teleport-stitch-v0 | 2 | 20 | 3 | 20000 | Stochastic TRL | 6 | 0.963 | 0.017 | 0.983 | 0.983 | 0.967 | 0.933 | 0.950 |
| antmaze-teleport-stitch-v0 | 2 | 20 | 3 | 20000 | Bellman full | 180 | 0.967 | 0.021 | 0.983 | 0.983 | 0.967 | 0.950 | 0.950 |

Stitch robustness signal: across controller seeds 0, 1, and 2, stochastic TRL stays in the 0.950-0.963 success range under the 6-sweep matched budget, while matched Bellman remains near 0.32. The seed-2 run is a near-match to the 180-sweep Bellman reference, trailing it by one successful episode out of 300 aggregate rollouts.

### AntMaze Navigate Controller-Seed Aggregate

| env | controller_seed | episodes_per_task | eval_seeds | controller_steps | method | sweeps | mean_success | success_std | task1_mean | task2_mean | task3_mean | task4_mean | task5_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-teleport-navigate-v0 | 0 | 20 | 3 | 50000 | Bellman matched | 6 | 0.310 | 0.037 | 0.000 | 0.500 | 0.383 | 0.417 | 0.250 |
| antmaze-teleport-navigate-v0 | 0 | 20 | 3 | 50000 | Stochastic TRL | 6 | 0.947 | 0.005 | 0.900 | 1.000 | 0.967 | 0.983 | 0.883 |
| antmaze-teleport-navigate-v0 | 0 | 20 | 3 | 50000 | Bellman full | 180 | 0.947 | 0.005 | 0.900 | 1.000 | 0.967 | 0.983 | 0.883 |
| antmaze-teleport-navigate-v0 | 1 | 20 | 3 | 50000 | Bellman matched | 6 | 0.320 | 0.043 | 0.000 | 0.483 | 0.383 | 0.417 | 0.317 |
| antmaze-teleport-navigate-v0 | 1 | 20 | 3 | 50000 | Stochastic TRL | 6 | 0.933 | 0.024 | 0.933 | 0.883 | 0.983 | 0.917 | 0.950 |
| antmaze-teleport-navigate-v0 | 1 | 20 | 3 | 50000 | Bellman full | 180 | 0.933 | 0.024 | 0.933 | 0.883 | 0.983 | 0.917 | 0.950 |
| antmaze-teleport-navigate-v0 | 2 | 20 | 3 | 50000 | Bellman matched | 6 | 0.310 | 0.050 | 0.000 | 0.483 | 0.367 | 0.383 | 0.317 |
| antmaze-teleport-navigate-v0 | 2 | 20 | 3 | 50000 | Stochastic TRL | 6 | 0.940 | 0.016 | 0.983 | 0.917 | 0.933 | 0.983 | 0.883 |
| antmaze-teleport-navigate-v0 | 2 | 20 | 3 | 50000 | Bellman full | 180 | 0.940 | 0.008 | 0.967 | 0.917 | 0.950 | 0.983 | 0.883 |

Navigate robustness signal: across controller seeds 0, 1, and 2, stochastic TRL stays in the 0.933-0.947 success range under the 6-sweep matched budget and matches the 180-sweep Bellman reference, while matched Bellman remains near 0.31.

### AntMaze 20-Episode Three-Seed Hard-Task Check

| env | episodes_per_task | eval_seed | controller_steps | method | sweeps | mean_success | task1_mean | task2_mean | task3_mean | task4_mean | task5_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-teleport-navigate-v0 | 20 | 0 | 50000 | Bellman matched | 6 | 0.360 | 0.000 | 0.600 | 0.500 | 0.400 | 0.300 |
| antmaze-teleport-navigate-v0 | 20 | 1 | 50000 | Bellman matched | 6 | 0.300 | 0.000 | 0.550 | 0.400 | 0.350 | 0.200 |
| antmaze-teleport-navigate-v0 | 20 | 2 | 50000 | Bellman matched | 6 | 0.270 | 0.000 | 0.350 | 0.250 | 0.500 | 0.250 |
| antmaze-teleport-navigate-v0 | 20 | 0 | 50000 | Stochastic TRL | 6 | 0.950 | 0.950 | 1.000 | 0.950 | 1.000 | 0.850 |
| antmaze-teleport-navigate-v0 | 20 | 1 | 50000 | Stochastic TRL | 6 | 0.940 | 0.850 | 1.000 | 0.950 | 1.000 | 0.900 |
| antmaze-teleport-navigate-v0 | 20 | 2 | 50000 | Stochastic TRL | 6 | 0.950 | 0.900 | 1.000 | 1.000 | 0.950 | 0.900 |
| antmaze-teleport-navigate-v0 | 20 | 0 | 50000 | Bellman full | 180 | 0.950 | 0.950 | 1.000 | 0.950 | 1.000 | 0.850 |
| antmaze-teleport-navigate-v0 | 20 | 1 | 50000 | Bellman full | 180 | 0.940 | 0.850 | 1.000 | 0.950 | 1.000 | 0.900 |
| antmaze-teleport-navigate-v0 | 20 | 2 | 50000 | Bellman full | 180 | 0.950 | 0.900 | 1.000 | 1.000 | 0.950 | 0.900 |
| antmaze-teleport-stitch-v0 | 20 | 0 | 20000 | Bellman matched | 6 | 0.380 | 0.000 | 0.600 | 0.500 | 0.400 | 0.400 |
| antmaze-teleport-stitch-v0 | 20 | 1 | 20000 | Bellman matched | 6 | 0.310 | 0.000 | 0.550 | 0.400 | 0.350 | 0.250 |
| antmaze-teleport-stitch-v0 | 20 | 2 | 20000 | Bellman matched | 6 | 0.260 | 0.000 | 0.250 | 0.250 | 0.500 | 0.300 |
| antmaze-teleport-stitch-v0 | 20 | 0 | 20000 | Stochastic TRL | 6 | 0.960 | 0.950 | 0.950 | 1.000 | 0.950 | 0.950 |
| antmaze-teleport-stitch-v0 | 20 | 1 | 20000 | Stochastic TRL | 6 | 0.960 | 0.900 | 0.950 | 1.000 | 1.000 | 0.950 |
| antmaze-teleport-stitch-v0 | 20 | 2 | 20000 | Stochastic TRL | 6 | 0.960 | 0.950 | 0.950 | 1.000 | 0.950 | 0.950 |
| antmaze-teleport-stitch-v0 | 20 | 0 | 20000 | Bellman full | 180 | 0.960 | 0.950 | 0.950 | 1.000 | 0.950 | 0.950 |
| antmaze-teleport-stitch-v0 | 20 | 1 | 20000 | Bellman full | 180 | 0.960 | 0.900 | 0.950 | 1.000 | 1.000 | 0.950 |
| antmaze-teleport-stitch-v0 | 20 | 2 | 20000 | Bellman full | 180 | 0.960 | 0.950 | 0.950 | 1.000 | 0.950 | 0.950 |

| env | n_eval_seeds | sto_minus_matched_mean | ci95_low | ci95_high | min_seed_diff | max_seed_diff | full_minus_sto_mean | all_seed_diffs_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-teleport-navigate-v0 | 3 | 0.637 | 0.525 | 0.749 | 0.590 | 0.680 | 0.000 | True |
| antmaze-teleport-stitch-v0 | 3 | 0.643 | 0.494 | 0.793 | 0.580 | 0.700 | 0.000 | True |

High-episode signal: with the same body-nearest executor and full-horizon evaluation, stochastic TRL matches the 180-sweep Bellman reference on both AntMaze navigate and stitch over three evaluation seeds and 20 episodes per task, while matched Bellman remains near 0.31 success.

### AntMaze Planning-Budget Screens

| env | planner | sweeps | mean_success | task1_mean | task2_mean | task3_mean | task4_mean | task5_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-teleport-navigate-v0 | Bellman | 6 | 0.333 | 0.000 | 0.000 | 0.333 | 0.667 | 0.667 |
| antmaze-teleport-navigate-v0 | Stochastic TRL | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| antmaze-teleport-navigate-v0 | Bellman | 12 | 0.667 | 0.667 | 1.000 | 0.333 | 0.667 | 0.667 |
| antmaze-teleport-navigate-v0 | Bellman | 24 | 0.933 | 1.000 | 1.000 | 1.000 | 1.000 | 0.667 |
| antmaze-teleport-navigate-v0 | Bellman | 45 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| antmaze-teleport-navigate-v0 | Bellman | 90 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| antmaze-teleport-navigate-v0 | Bellman full | 180 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| antmaze-teleport-stitch-v0 | Bellman | 6 | 0.333 | 0.000 | 0.000 | 0.333 | 0.667 | 0.667 |
| antmaze-teleport-stitch-v0 | Stochastic TRL | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| antmaze-teleport-stitch-v0 | Bellman | 12 | 0.667 | 0.667 | 1.000 | 0.333 | 0.667 | 0.667 |
| antmaze-teleport-stitch-v0 | Bellman | 24 | 0.933 | 1.000 | 1.000 | 1.000 | 1.000 | 0.667 |
| antmaze-teleport-stitch-v0 | Bellman | 45 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| antmaze-teleport-stitch-v0 | Bellman | 90 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| antmaze-teleport-stitch-v0 | Bellman full | 180 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Budget signal: on both AntMaze navigate and stitch seed-0 screens, stochastic TRL reaches 1.000 success with 6 sweeps. Ordinary Bellman is 0.333 at 6 sweeps and first reaches 1.000 at 45 sweeps in these screens.

### AntMaze Stitch Executor Ablation

| executor | method | sweeps | mean_success | task1_mean | task2_mean | task3_mean | task4_mean | task5_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_xy | Stochastic TRL | 6 | 0.867 | 0.667 | 1.000 | 1.000 | 1.000 | 0.667 |
| nearest_xy | Bellman full | 180 | 0.867 | 0.667 | 1.000 | 1.000 | 1.000 | 0.667 |
| body_nearest_k16 | Stochastic TRL | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| body_nearest_k16 | Bellman full | 180 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Executor signal: on the AntMaze stitch seed-0 screen, replacing a single nearest-xy waypoint observation with 16 body-nearest waypoint candidates improves both stochastic TRL and full Bellman from 0.867 to 1.000. This isolates the remaining failures as low-level executor sensitivity, not value-propagation errors.

## Teleport Stitch Screens

| env | executor | episodes_per_task | controller_steps | method | sweeps | mean_success | task1_mean | task2_mean | task3_mean | task4_mean | task5_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pointmaze-teleport-stitch-v0 | topology_proportional | 10 | 0 | Bellman matched | 6 | 0.38 | 0.0 | 0.5 | 0.6 | 0.4 | 0.4 |
| pointmaze-teleport-stitch-v0 | topology_proportional | 10 | 0 | Stochastic TRL | 6 | 0.98 | 0.9 | 1.0 | 1.0 | 1.0 | 1.0 |
| pointmaze-teleport-stitch-v0 | topology_proportional | 10 | 0 | Bellman full | 180 | 0.98 | 0.9 | 1.0 | 1.0 | 1.0 | 1.0 |
| antmaze-teleport-stitch-v0 | full_obs_bc_bodyk16 | 10 | 20000 | Bellman matched | 6 | 0.38 | 0.0 | 0.5 | 0.6 | 0.4 | 0.4 |
| antmaze-teleport-stitch-v0 | full_obs_bc_bodyk16 | 10 | 20000 | Stochastic TRL | 6 | 0.9400000000000001 | 1.0 | 0.9 | 1.0 | 1.0 | 0.8 |
| antmaze-teleport-stitch-v0 | full_obs_bc_bodyk16 | 10 | 20000 | Bellman full | 180 | 0.96 | 1.0 | 0.9 | 1.0 | 1.0 | 0.9 |

Stitch signal: stochastic TRL matches the long-sweep Bellman reference on PointMaze and nearly matches it on AntMaze under the matched 6-sweep budget. These are seed-0 fast screens, but they are harder long-horizon variants than the navigate-only task.
