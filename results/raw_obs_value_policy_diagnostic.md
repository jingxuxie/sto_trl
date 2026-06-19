# Raw-Observation Value/Policy Head Diagnostic

Date: 2026-06-19.

## Purpose

Test whether the learned-transition/topology screens can move beyond table
values by replacing the planner Q-table with a raw-observation neural head over
cell-representative observations, high-level actions, and raw goal features.

This is still a cell-level diagnostic, but it is a stricter learned-module
screen than using a table of action values at execution time.

## Screen

Primary negative-screen environment: `pointmaze-teleport-stitch-v0`.
Positive tie-policy confirmation environments:
`pointmaze-teleport-navigate-v0`, `pointmaze-teleport-stitch-v0`,
`antmaze-teleport-navigate-v0`, and `antmaze-teleport-stitch-v0`.

Shared transition model:

- `--transition-model raw_obs_mlp`
- `--transition-target-source dataset_cell_changes`
- `--transition-steps 2000`
- `--transition-seed 0`

Evaluation:

- seed 0
- PointMaze: 20 episodes per task, all five tasks
- AntMaze: 20 episodes per task, tasks 4 and 5, learned BC executor with
  body-nearest k16 waypoint goals

## Results

| value/control head | method | success | fit metric | action metric | raw file |
| --- | --- | ---: | --- | --- | --- |
| table Q reference | stochastic TRL | 0.980 | n/a | n/a | `results/pointmaze_stitch_rawobs_mlp_transition_table_ep20_seed0_check.csv` |
| raw XY action-value MLP | Bellman matched | 0.000 | MSE 0.001327 | action agreement 0.514 | `results/pointmaze_stitch_rawobs_mlp_transition_value_ep20_seed0.csv` |
| raw XY action-value MLP | stochastic TRL | 0.000 | MSE 0.000081 | action agreement 0.560 | `results/pointmaze_stitch_rawobs_mlp_transition_value_ep20_seed0.csv` |
| raw XY action-value MLP | Bellman full | 0.000 | MSE 0.000114 | action agreement 0.615 | `results/pointmaze_stitch_rawobs_mlp_transition_value_ep20_seed0.csv` |
| raw XY action-value MLP + ranking CE | stochastic TRL | 0.000 | MSE 0.041514 | action agreement 0.332 | `results/pointmaze_stitch_rawobs_mlp_transition_value_rank_ep20_seed0.csv` |
| raw XY policy MLP distillation | stochastic TRL | 0.000 | MSE 0.626885 | static action agreement 1.000 | `results/pointmaze_stitch_rawobs_mlp_transition_policy_ep20_seed0.csv` |
| raw XY tie-policy MLP, navigate | Bellman matched | 0.530 | MSE 0.348771 | action agreement 0.968 | `results/pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv` |
| raw XY tie-policy MLP, navigate | stochastic TRL | 0.980 | MSE 0.610539 | action agreement 0.982 | `results/pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv` |
| raw XY tie-policy MLP, navigate | Bellman full | 0.980 | MSE 0.618319 | action agreement 0.977 | `results/pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv` |
| raw XY tie-policy MLP | Bellman matched | 0.530 | MSE 0.348617 | action agreement 0.968 | `results/pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv` |
| raw XY tie-policy MLP | stochastic TRL | 0.980 | MSE 0.610715 | action agreement 0.978 | `results/pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv` |
| raw XY tie-policy MLP | Bellman full | 0.980 | MSE 0.618418 | action agreement 0.978 | `results/pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv` |
| PointMaze raw transition + previous-action policy MLP, navigate | Bellman matched | 0.530 | transition top-1 1.000 | action agreement 0.990 | `results/pointmaze_navigate_rawobs_mlp_transition_prev_policy_ep20_seed0.csv` |
| PointMaze raw transition + previous-action policy MLP, navigate | stochastic TRL | 0.980 | transition top-1 1.000 | action agreement 1.000 | `results/pointmaze_navigate_rawobs_mlp_transition_prev_policy_ep20_seed0.csv` |
| PointMaze raw transition + previous-action policy MLP, navigate | Bellman full | 0.980 | transition top-1 1.000 | action agreement 1.000 | `results/pointmaze_navigate_rawobs_mlp_transition_prev_policy_ep20_seed0.csv` |
| PointMaze raw transition + previous-action policy MLP, stitch | Bellman matched | 0.530 | transition top-1 0.959 | action agreement 0.997 | `results/pointmaze_stitch_rawobs_mlp_transition_prev_policy_ep20_seed0.csv` |
| PointMaze raw transition + previous-action policy MLP, stitch | stochastic TRL | 0.980 | transition top-1 0.959 | action agreement 1.000 | `results/pointmaze_stitch_rawobs_mlp_transition_prev_policy_ep20_seed0.csv` |
| PointMaze raw transition + previous-action policy MLP, stitch | Bellman full | 0.980 | transition top-1 0.959 | action agreement 1.000 | `results/pointmaze_stitch_rawobs_mlp_transition_prev_policy_ep20_seed0.csv` |
| AntMaze raw obs tie-policy MLP, navigate | Bellman matched | 0.350 | MSE 0.693998 | action agreement 1.000 | `results/antmaze_navigate_tie_policy_head_hard_tasks_ep20_seed0.csv` |
| AntMaze raw obs tie-policy MLP, navigate | stochastic TRL | 0.925 | MSE 0.630426 | action agreement 1.000 | `results/antmaze_navigate_tie_policy_head_hard_tasks_ep20_seed0.csv` |
| AntMaze raw obs tie-policy MLP, navigate | Bellman full | 0.925 | MSE 0.635721 | action agreement 1.000 | `results/antmaze_navigate_tie_policy_head_hard_tasks_ep20_seed0.csv` |
| AntMaze raw obs tie-policy MLP, stitch | Bellman matched | 0.400 | MSE 0.694200 | action agreement 1.000 | `results/antmaze_stitch_tie_policy_head_hard_tasks_ep20_seed0.csv` |
| AntMaze raw obs tie-policy MLP, stitch | stochastic TRL | 0.950 | MSE 0.630718 | action agreement 1.000 | `results/antmaze_stitch_tie_policy_head_hard_tasks_ep20_seed0.csv` |
| AntMaze raw obs tie-policy MLP, stitch | Bellman full | 0.950 | MSE 0.635905 | action agreement 1.000 | `results/antmaze_stitch_tie_policy_head_hard_tasks_ep20_seed0.csv` |
| AntMaze previous-action policy MLP, navigate | Bellman matched | 0.400 | MSE 0.004167 | action agreement 1.000 | `results/antmaze_navigate_prev_policy_head_ep5_seed0_task45.csv` |
| AntMaze previous-action policy MLP, navigate | stochastic TRL | 0.900 | MSE 0.004167 | action agreement 1.000 | `results/antmaze_navigate_prev_policy_head_ep5_seed0_task45.csv` |
| AntMaze previous-action policy MLP, navigate | Bellman full | 0.900 | MSE 0.004167 | action agreement 1.000 | `results/antmaze_navigate_prev_policy_head_ep5_seed0_task45.csv` |
| AntMaze previous-action policy MLP, stitch | Bellman matched | 0.600 | MSE 0.004167 | action agreement 1.000 | `results/antmaze_stitch_prev_policy_head_ep5_seed0_task45.csv` |
| AntMaze previous-action policy MLP, stitch | stochastic TRL | 1.000 | MSE 0.004167 | action agreement 1.000 | `results/antmaze_stitch_prev_policy_head_ep5_seed0_task45.csv` |
| AntMaze previous-action policy MLP, stitch | Bellman full | 1.000 | MSE 0.004167 | action agreement 1.000 | `results/antmaze_stitch_prev_policy_head_ep5_seed0_task45.csv` |
| AntMaze raw transition + tie-policy MLP, navigate | Bellman matched | 0.300 | transition top-1 1.000 | action agreement 0.994 | `results/antmaze_navigate_rawobs_transition_tie_policy_head_ep10_seed0.csv` |
| AntMaze raw transition + tie-policy MLP, navigate | stochastic TRL | 0.950 | transition top-1 1.000 | action agreement 0.999 | `results/antmaze_navigate_rawobs_transition_tie_policy_head_ep10_seed0.csv` |
| AntMaze raw transition + tie-policy MLP, navigate | Bellman full | 0.950 | transition top-1 1.000 | action agreement 0.999 | `results/antmaze_navigate_rawobs_transition_tie_policy_head_ep10_seed0.csv` |
| AntMaze raw transition + tie-policy MLP, stitch | Bellman matched | 0.300 | transition top-1 1.000 | action agreement 0.988 | `results/antmaze_stitch_rawobs_transition_tie_policy_head_ep10_seed0.csv` |
| AntMaze raw transition + tie-policy MLP, stitch | stochastic TRL | 1.000 | transition top-1 1.000 | action agreement 0.999 | `results/antmaze_stitch_rawobs_transition_tie_policy_head_ep10_seed0.csv` |
| AntMaze raw transition + tie-policy MLP, stitch | Bellman full | 1.000 | transition top-1 1.000 | action agreement 0.999 | `results/antmaze_stitch_rawobs_transition_tie_policy_head_ep10_seed0.csv` |
| AntMaze raw transition + previous-action policy MLP, navigate | Bellman matched | 0.400 | transition top-1 1.000 | action agreement 1.000 | `results/antmaze_navigate_rawobs_transition_prev_policy_head_ep5_seed0.csv` |
| AntMaze raw transition + previous-action policy MLP, navigate | stochastic TRL | 0.900 | transition top-1 1.000 | action agreement 1.000 | `results/antmaze_navigate_rawobs_transition_prev_policy_head_ep5_seed0.csv` |
| AntMaze raw transition + previous-action policy MLP, navigate | Bellman full | 0.900 | transition top-1 1.000 | action agreement 1.000 | `results/antmaze_navigate_rawobs_transition_prev_policy_head_ep5_seed0.csv` |
| AntMaze raw transition + previous-action policy MLP, stitch | Bellman matched | 0.400 | transition top-1 0.875 | action agreement 1.000 | `results/antmaze_stitch_rawobs_transition_prev_policy_head_ep5_seed0.csv` |
| AntMaze raw transition + previous-action policy MLP, stitch | stochastic TRL | 1.000 | transition top-1 0.875 | action agreement 1.000 | `results/antmaze_stitch_rawobs_transition_prev_policy_head_ep5_seed0.csv` |
| AntMaze raw transition + previous-action policy MLP, stitch | Bellman full | 1.000 | transition top-1 0.875 | action agreement 1.000 | `results/antmaze_stitch_rawobs_transition_prev_policy_head_ep5_seed0.csv` |
| AntMaze raw transition + previous-action policy MLP, navigate, 3 eval seeds | Bellman matched | 0.350 | transition oracle top-1 1.000 | action agreement 1.000 | `results/paper_tables/antmaze_rawobs_transition_prev_policy_head_ep10_evalseed012.csv` |
| AntMaze raw transition + previous-action policy MLP, navigate, 3 eval seeds | stochastic TRL | 0.933 | transition oracle top-1 1.000 | action agreement 1.000 | `results/paper_tables/antmaze_rawobs_transition_prev_policy_head_ep10_evalseed012.csv` |
| AntMaze raw transition + previous-action policy MLP, navigate, 3 eval seeds | Bellman full | 0.933 | transition oracle top-1 1.000 | action agreement 1.000 | `results/paper_tables/antmaze_rawobs_transition_prev_policy_head_ep10_evalseed012.csv` |
| AntMaze raw transition + previous-action policy MLP, stitch, 3 eval seeds | Bellman matched | 0.283 | transition oracle top-1 0.994 | action agreement 1.000 | `results/paper_tables/antmaze_rawobs_transition_prev_policy_head_ep10_evalseed012.csv` |
| AntMaze raw transition + previous-action policy MLP, stitch, 3 eval seeds | stochastic TRL | 0.967 | transition oracle top-1 0.994 | action agreement 1.000 | `results/paper_tables/antmaze_rawobs_transition_prev_policy_head_ep10_evalseed012.csv` |
| AntMaze raw transition + previous-action policy MLP, stitch, 3 eval seeds | Bellman full | 0.950 | transition oracle top-1 0.994 | action agreement 1.000 | `results/paper_tables/antmaze_rawobs_transition_prev_policy_head_ep10_evalseed012.csv` |

## Interpretation

The learned transition model is not the bottleneck here: the table-Q reference
with the same raw-observation transition MLP reaches 0.980 success. The failure
appears when replacing the Q-table with a low-dimensional raw XY neural head
that cannot preserve the executor's tie semantics.

Low global value MSE is insufficient for long-horizon control: small ranking
errors around bottlenecks or teleporter-adjacent states can destroy rollout
success. The policy-distillation result also shows that static action-label
agreement is not enough; the PointMaze executor uses sticky tie-breaking through
the previous high-level action, and arbitrary tie labels can produce a policy
that agrees with a static argmax diagnostic but fails in rollout.

The tie-policy MLP is the first positive neural value/control-head result in
this screen. It preserves all near-optimal actions as a binary action mask
instead of forcing a single arbitrary argmax label. With that representation,
stochastic TRL matches the full Bellman reference at 0.980 success on both
PointMaze teleport navigate and stitch, while the matched-budget Bellman
planner remains much lower at 0.530.

The AntMaze tie-policy screen keeps the topology transition model and learned
BC executor fixed, so it isolates the high-level value/control representation.
On tasks 4 and 5, the tie-policy head reaches 0.925 stochastic-TRL success on
navigate and 0.950 on stitch with 20 episodes per task, matching full Bellman,
while matched Bellman reaches 0.350 and 0.400. This is a compact single-seed
hard-task confirmation, not a full neural transition/value result.

A previous-action-conditioned single-label policy MLP fixes the specific
sticky-tie failure mode without using a tie mask. On PointMaze, with the
raw-observation transition head, it reaches 0.980 stochastic-TRL success on
both navigate and stitch over all tasks, matching full Bellman while matched
Bellman reaches 0.530. The key implementation detail is that 4D
previous-action policy scores must still receive the same tiny sticky
previous-action tie-break used by table Q-values; without it, the policy can
drift out of the goal cell after arrival. With topology transitions on AntMaze,
the same head reaches 0.900 stochastic-TRL success on navigate and 1.000 on
stitch over the hard-task slice, matching full Bellman while preserving 1.000
action agreement to the previous-action-conditioned table policy.

The combined AntMaze screen replaces the topology transition model with a
raw-observation MLP jump-change transition head and keeps the tie-policy head.
Across three transition seeds with 10 episodes per hard task, stochastic TRL
reaches 0.950 on navigate and 1.000 on stitch, again matching full Bellman
while matched Bellman reaches 0.300 on both. This is the strongest current
learned-module AntMaze diagnostic, though it still operates over the high-level
cell abstraction and learned BC executor.

With transition seed 0 fixed and evaluation seeds 0, 1, and 2, the same
combined screen reaches 0.933 on navigate and 0.967 on stitch, matching full
Bellman while matched Bellman reaches 0.283. This confirms the result is not
only a single rollout-seed artifact on the AntMaze hard-task slice.

Replacing the tie-policy head in the combined AntMaze screen with the
previous-action single-label head remains positive across evaluation seeds 0,
1, and 2 with transition seed 0 fixed and 10 episodes per hard task:
stochastic TRL reaches 0.933 on navigate and 0.967 on stitch, staying within
0.02 of full Bellman, while matched Bellman reaches 0.350 and 0.283. This is a
more direct learned policy-head story than the tie-mask diagnostic, though it
still operates over the high-level cell abstraction.

## Next Neural Step

Promote the tie-preserving policy head and the previous-action-conditioned
single-label policy head as positive neural-control diagnostics over the
high-level abstraction. Do not promote the scalar value MLP or previous-action
free single-label policy MLP failures as solved. A stronger neural screen should
add one of:

- topology-aware or wall-aware features while keeping raw observations as input;
- an action-ranking loss that preserves near-ties rather than forcing arbitrary
  argmax labels;
- direct rollout/path imitation from the successful stochastic TRL table policy.
