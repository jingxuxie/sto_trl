# Reproduce Main Stochastic TRL Results

Run commands from the repository root. These commands use the conda environment
`autoresearcher_sto_trl`.

```bash
PY=/home/eston/anaconda3/envs/autoresearcher_sto_trl/bin/python
```

Some OGBench screen scripts expect the reference TRL checkout under
`external/trl`. Recreate it in a fresh clone with:

```bash
git clone https://github.com/aoberai/trl.git external/trl
git -C external/trl apply ../trl_local_changes.patch
```

## Fast Verification Path

Regenerate derived paper artifacts and verify headline claims from the current
raw CSVs:

```bash
$PY scripts/generate_paper_artifacts.py
$PY scripts/verify_main_claims.py
$PY scripts/verify_latex_claims.py
```

Expected verifier status: `PASS`.

Primary outputs:

- `results/paper_tables/main_hard_task_results.csv`
- `results/paper_tables/hard_task_stress_seed0.csv`
- `results/paper_tables/pointmaze_stitch_task5_ep100_seed01234.csv`
- `results/paper_tables/pointmaze_learned_transition.csv`
- `results/paper_tables/antmaze_learned_transition_robustness.csv`
- `results/paper_tables/pointmaze_tie_policy_head_ep20_seed0.csv`
- `results/paper_tables/antmaze_tie_policy_head_hard_tasks_ep20_seed0.csv`
- `results/paper_tables/antmaze_rawobs_transition_tie_policy_head_ep10_tseed012.csv`
- `results/paper_tables/antmaze_rawobs_transition_tie_policy_head_ep10_evalseed012.csv`
- `results/paper_tables/antmaze_navigate_controller_seeds_ep20_seed012.csv`
- `results/paper_tables/antmaze_stitch_controller_seeds_ep20_seed012.csv`
- `results/paper_artifacts.md`
- `results/main_claim_verification.md`
- `paper/stochastic_trl/latex_claim_verification.md`
- `PAPER_CLAIM_PACKAGE.md`
- `STOCHASTIC_TRL_MANUSCRIPT.md`
- `PAPER_FIGURE_TABLE_CHECKLIST.md`

## Full Main-Table Regeneration

PointMaze teleport navigate:

```bash
$PY scripts/run_pointmaze_topology_planner.py \
  --env-name pointmaze-teleport-navigate-v0 \
  --topology-source dataset \
  --gamma 0.995 \
  --full-iters 180 \
  --episodes 50 \
  --seeds 0 1 2 3 4 \
  --methods bellman_matched sto_trl_matched bellman_full \
  --out results/pointmaze_topology_dataset_5seed_ep50.csv \
  --summary-out results/pointmaze_topology_dataset_5seed_ep50.json
```

PointMaze teleport stitch:

```bash
$PY scripts/run_pointmaze_topology_planner.py \
  --env-name pointmaze-teleport-stitch-v0 \
  --topology-source dataset \
  --gamma 0.995 \
  --full-iters 180 \
  --episodes 50 \
  --seeds 0 1 2 3 4 \
  --methods bellman_matched sto_trl_matched bellman_full \
  --out results/pointmaze_topology_stitch_5seed_ep50.csv \
  --summary-out results/pointmaze_topology_stitch_5seed_ep50.json
```

AntMaze teleport navigate, eval-only from saved 50k-step BC controller:

```bash
env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
  scripts/run_antmaze_bc_topology_planner.py \
  --env-name antmaze-teleport-navigate-v0 \
  --gamma 0.995 \
  --full-iters 180 \
  --episodes 20 \
  --seeds 0 1 2 \
  --min-jump-count 20 \
  --load-policy results/policies/antmaze_navigate_fullgoal_bc_50k.pkl \
  --bc-batch-size 2048 \
  --bc-min-future 1 \
  --bc-max-future 30 \
  --goal-representation full \
  --goal-candidates-per-state 16 \
  --goal-candidate-mode body_nearest \
  --waypoint-lookahead 1 \
  --eval-action-repeat 1 \
  --methods bellman_matched sto_trl_matched bellman_full \
  --out results/antmaze_bc_topology_navigate_fullgoal_50k_ep20_seed012_bodyk16_cpu.csv \
  --summary-out results/antmaze_bc_topology_navigate_fullgoal_50k_ep20_seed012_bodyk16_cpu.json \
  --progress-log results/antmaze_bc_topology_navigate_fullgoal_50k_ep20_seed012_bodyk16_cpu.progress.jsonl
```

AntMaze teleport stitch, eval-only from saved 20k-step BC controller:

```bash
env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
  scripts/run_antmaze_bc_topology_planner.py \
  --env-name antmaze-teleport-stitch-v0 \
  --gamma 0.995 \
  --full-iters 180 \
  --episodes 20 \
  --seeds 0 1 2 \
  --min-jump-count 20 \
  --load-policy results/policies/antmaze_stitch_fullgoal_bc_20k.pkl \
  --bc-batch-size 2048 \
  --bc-min-future 1 \
  --bc-max-future 30 \
  --goal-representation full \
  --goal-candidates-per-state 16 \
  --goal-candidate-mode body_nearest \
  --waypoint-lookahead 1 \
  --eval-action-repeat 1 \
  --methods bellman_matched sto_trl_matched bellman_full \
  --out results/antmaze_bc_topology_stitch_fullgoal_20k_ep20_seed012_bodyk16_cpu.csv \
  --summary-out results/antmaze_bc_topology_stitch_fullgoal_20k_ep20_seed012_bodyk16_cpu.json \
  --progress-log results/antmaze_bc_topology_stitch_fullgoal_20k_ep20_seed012_bodyk16_cpu.progress.jsonl
```

Then regenerate and verify:

```bash
$PY scripts/generate_paper_artifacts.py
$PY scripts/verify_main_claims.py
$PY scripts/verify_latex_claims.py
```

Compile the LaTeX draft:

```bash
cd paper/stochastic_trl
latexmk -pdf -interaction=nonstopmode main.tex
```

## Fast Hard-Task Stress Checks

These are short single-evaluation-seed checks for fast iteration on harder task
slices. They are generated into `results/paper_tables/hard_task_stress_seed0.csv`
and verified against raw CSVs by `scripts/verify_main_claims.py`.

PointMaze teleport stitch, all tasks:

```bash
$PY scripts/run_pointmaze_topology_planner.py \
  --env-name pointmaze-teleport-stitch-v0 \
  --topology-source env \
  --methods bellman_matched sto_trl_matched bellman_full support_trl_matched \
  --episodes 20 \
  --seeds 0 \
  --out results/pointmaze_stitch_seed0_ep20_methods.csv \
  --summary-out results/pointmaze_stitch_seed0_ep20_methods.json
```

Focused PointMaze teleport stitch task 5, 100 episodes per seed:

```bash
$PY scripts/run_pointmaze_topology_planner.py \
  --env-name pointmaze-teleport-stitch-v0 \
  --topology-source dataset \
  --gamma 0.995 \
  --full-iters 180 \
  --episodes 100 \
  --seeds 0 1 2 3 4 \
  --task-ids 5 \
  --methods bellman_matched sto_trl_matched bellman_full \
  --out results/pointmaze_stitch_task5_ep100_seed01234.csv \
  --summary-out results/pointmaze_stitch_task5_ep100_seed01234.json
```

AntMaze teleport navigate, tasks 4 and 5:

```bash
env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
  scripts/run_antmaze_bc_topology_planner.py \
  --env-name antmaze-teleport-navigate-v0 \
  --load-policy results/policies/antmaze_navigate_fullgoal_bc_50k.pkl \
  --goal-representation full \
  --goal-candidates-per-state 16 \
  --goal-candidate-mode body_nearest \
  --bc-min-future 1 \
  --bc-max-future 30 \
  --bc-batch-size 2048 \
  --waypoint-lookahead 1 \
  --eval-action-repeat 1 \
  --methods bellman_matched sto_trl_matched bellman_full \
  --episodes 5 \
  --seeds 0 \
  --task-ids 4 5 \
  --out results/antmaze_navigate_hard_tasks_bodyk16_seed0_methods.csv \
  --summary-out results/antmaze_navigate_hard_tasks_bodyk16_seed0_methods.json
```

AntMaze teleport stitch, tasks 4 and 5:

```bash
env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
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
  --methods bellman_matched sto_trl_matched bellman_full \
  --episodes 5 \
  --seeds 0 \
  --task-ids 4 5 \
  --out results/antmaze_stitch_hard_tasks_bodyk16_seed0_methods.csv \
  --summary-out results/antmaze_stitch_hard_tasks_bodyk16_seed0_methods.json
```

### Fast Profiled AntMaze Screens

For iteration, use the saved-policy, cached-topology, two-method profile runs
below before spending time on full Bellman, more episodes, or more seeds. The
generated summary is `results/paper_tables/fast_eval_profile.csv`.

AntMaze navigate hard slice:

```bash
env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
  scripts/run_antmaze_bc_topology_planner.py \
  --env-name antmaze-teleport-navigate-v0 \
  --load-policy results/policies/antmaze_navigate_fullgoal_bc_50k.pkl \
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
  --episodes 5 \
  --seeds 0 \
  --task-ids 4 5 \
  --profile-eval \
  --out results/antmaze_navigate_fast_profile_repeat1_ep5_seed0_task45.csv \
  --summary-out results/antmaze_navigate_fast_profile_repeat1_ep5_seed0_task45.json \
  --progress-log results/antmaze_navigate_fast_profile_repeat1_ep5_seed0_task45.progress.jsonl
```

AntMaze stitch hard slice:

```bash
env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
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
  --episodes 5 \
  --seeds 0 \
  --task-ids 4 5 \
  --profile-eval \
  --out results/antmaze_stitch_fast_profile_repeat1_ep5_seed0_task45.csv \
  --summary-out results/antmaze_stitch_fast_profile_repeat1_ep5_seed0_task45.json \
  --progress-log results/antmaze_stitch_fast_profile_repeat1_ep5_seed0_task45.progress.jsonl
```

Do not use `--eval-action-repeat 2` for claimed AntMaze screens yet. The
current stitch profile shows it is not claim-safe: stochastic TRL drops from
1.000 to 0.750 on the two-episode hard-slice comparison.

## Learned-Transition Appendix Screens

These screens fit high-level transition models from collapsed offline
cell-change targets on PointMaze and finite sampled outcomes on AntMaze. The
PointMaze appendix now includes both a table-softmax model and a shared
raw-observation MLP transition head. They are generated into
`results/paper_tables/pointmaze_learned_transition.csv` and
`results/paper_tables/antmaze_learned_transition_robustness.csv`.

PointMaze teleport navigate/stitch, collapsed offline cell-change targets:

```bash
$PY scripts/run_pointmaze_learned_transition_planner.py \
  --env-name pointmaze-teleport-navigate-v0 \
  --topology-source dataset \
  --transition-target-source dataset_cell_changes \
  --transition-steps 1000 \
  --episodes 50 \
  --seeds 0 \
  --methods bellman_matched sto_trl_matched bellman_full \
  --out results/pointmaze_learned_transition_navigate_cellchanges_1k_ep50_seed0.csv \
  --summary-out results/pointmaze_learned_transition_navigate_cellchanges_1k_ep50_seed0.json

$PY scripts/run_pointmaze_learned_transition_planner.py \
  --env-name pointmaze-teleport-stitch-v0 \
  --topology-source dataset \
  --transition-target-source dataset_cell_changes \
  --transition-steps 1000 \
  --episodes 50 \
  --seeds 0 \
  --methods bellman_matched sto_trl_matched bellman_full \
  --out results/pointmaze_learned_transition_stitch_cellchanges_1k_ep50_seed0.csv \
  --summary-out results/pointmaze_learned_transition_stitch_cellchanges_1k_ep50_seed0.json
```

PointMaze raw-observation MLP transition heads, same task/evaluation protocol.
The generated table aggregates transition seeds 0, 1, and 2. The commands
below show transition seed 0; repeat with `--transition-seed 1` and
`--transition-seed 2`, changing the output stems to `_tseed1` and `_tseed2`.

```bash
env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
  scripts/run_pointmaze_learned_transition_planner.py \
  --env-name pointmaze-teleport-navigate-v0 \
  --topology-source dataset \
  --transition-model raw_obs_mlp \
  --transition-target-source dataset_cell_changes \
  --transition-steps 2000 \
  --transition-seed 0 \
  --transition-log-interval 1000 \
  --transition-mlp-lr 0.003 \
  --episodes 50 \
  --seeds 0 \
  --methods bellman_matched sto_trl_matched bellman_full \
  --out results/pointmaze_navigate_rawobs_mlp_cellchanges_ep50_seed0.csv \
  --summary-out results/pointmaze_navigate_rawobs_mlp_cellchanges_ep50_seed0.json

env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
  scripts/run_pointmaze_learned_transition_planner.py \
  --env-name pointmaze-teleport-stitch-v0 \
  --topology-source dataset \
  --transition-model raw_obs_mlp \
  --transition-target-source dataset_cell_changes \
  --transition-steps 2000 \
  --transition-seed 0 \
  --transition-log-interval 1000 \
  --transition-mlp-lr 0.003 \
  --episodes 50 \
  --seeds 0 \
  --methods bellman_matched sto_trl_matched bellman_full \
  --out results/pointmaze_stitch_rawobs_mlp_cellchanges_ep50_seed0.csv \
  --summary-out results/pointmaze_stitch_rawobs_mlp_cellchanges_ep50_seed0.json
```

AntMaze learned-transition robustness uses the saved BC policies and transition
seeds 0, 1, and 2. Replace `--transition-seed` and output filenames for each
seed.

```bash
env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
  scripts/run_antmaze_bc_topology_planner.py \
  --env-name antmaze-teleport-navigate-v0 \
  --load-policy results/policies/antmaze_navigate_fullgoal_bc_50k.pkl \
  --transition-model learned_softmax \
  --transition-target-source topology_samples \
  --samples-per-row 20 \
  --transition-steps 1000 \
  --transition-seed 0 \
  --goal-representation full \
  --goal-candidates-per-state 16 \
  --goal-candidate-mode body_nearest \
  --bc-min-future 1 \
  --bc-max-future 30 \
  --bc-batch-size 2048 \
  --waypoint-lookahead 1 \
  --eval-action-repeat 1 \
  --policy-eval-backend jax \
  --methods bellman_matched sto_trl_matched bellman_full \
  --episodes 10 \
  --seeds 0 \
  --task-ids 4 5 \
  --out results/antmaze_navigate_learned_transition_samples20_ep10_tseed0.csv \
  --summary-out results/antmaze_navigate_learned_transition_samples20_ep10_tseed0.json \
  --progress-log results/antmaze_navigate_learned_transition_samples20_ep10_tseed0.progress.jsonl

env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
  scripts/run_antmaze_bc_topology_planner.py \
  --env-name antmaze-teleport-stitch-v0 \
  --load-policy results/policies/antmaze_stitch_fullgoal_bc_20k.pkl \
  --transition-model learned_softmax \
  --transition-target-source topology_samples \
  --samples-per-row 20 \
  --transition-steps 1000 \
  --transition-seed 0 \
  --goal-representation full \
  --goal-candidates-per-state 16 \
  --goal-candidate-mode body_nearest \
  --bc-min-future 1 \
  --bc-max-future 30 \
  --bc-batch-size 2048 \
  --waypoint-lookahead 1 \
  --eval-action-repeat 1 \
  --policy-eval-backend jax \
  --methods bellman_matched sto_trl_matched bellman_full \
  --episodes 10 \
  --seeds 0 \
  --task-ids 4 5 \
  --out results/antmaze_stitch_learned_transition_samples20_ep10_tseed0.csv \
  --summary-out results/antmaze_stitch_learned_transition_samples20_ep10_tseed0.json \
  --progress-log results/antmaze_stitch_learned_transition_samples20_ep10_tseed0.progress.jsonl
```

AntMaze raw-observation MLP jump-change transition heads, same hard-task
evaluation protocol. Repeat each command with `--transition-seed 1` and
`--transition-seed 2`, changing the output suffix to `_tseed1` and `_tseed2`;
the seed-0 filenames below keep the original `_seed0` suffix.

```bash
env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
  scripts/run_antmaze_bc_topology_planner.py \
  --env-name antmaze-teleport-navigate-v0 \
  --load-policy results/policies/antmaze_navigate_fullgoal_bc_50k.pkl \
  --transition-model raw_obs_mlp \
  --transition-target-source dataset_jump_changes \
  --transition-steps 1000 \
  --transition-log-interval 500 \
  --transition-mlp-lr 0.003 \
  --goal-representation full \
  --goal-candidates-per-state 16 \
  --goal-candidate-mode body_nearest \
  --bc-min-future 1 \
  --bc-max-future 30 \
  --bc-batch-size 2048 \
  --waypoint-lookahead 1 \
  --eval-action-repeat 1 \
  --policy-eval-backend jax \
  --methods bellman_matched sto_trl_matched bellman_full \
  --episodes 10 \
  --seeds 0 \
  --task-ids 4 5 \
  --out results/antmaze_navigate_rawobs_mlp_jumpchanges_ep10_seed0.csv \
  --summary-out results/antmaze_navigate_rawobs_mlp_jumpchanges_ep10_seed0.json \
  --progress-log results/antmaze_navigate_rawobs_mlp_jumpchanges_ep10_seed0.progress.jsonl

env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
  scripts/run_antmaze_bc_topology_planner.py \
  --env-name antmaze-teleport-stitch-v0 \
  --load-policy results/policies/antmaze_stitch_fullgoal_bc_20k.pkl \
  --transition-model raw_obs_mlp \
  --transition-target-source dataset_jump_changes \
  --transition-steps 1000 \
  --transition-log-interval 500 \
  --transition-mlp-lr 0.003 \
  --goal-representation full \
  --goal-candidates-per-state 16 \
  --goal-candidate-mode body_nearest \
  --bc-min-future 1 \
  --bc-max-future 30 \
  --bc-batch-size 2048 \
  --waypoint-lookahead 1 \
  --eval-action-repeat 1 \
  --policy-eval-backend jax \
  --methods bellman_matched sto_trl_matched bellman_full \
  --episodes 10 \
  --seeds 0 \
  --task-ids 4 5 \
  --out results/antmaze_stitch_rawobs_mlp_jumpchanges_ep10_seed0.csv \
  --summary-out results/antmaze_stitch_rawobs_mlp_jumpchanges_ep10_seed0.json \
  --progress-log results/antmaze_stitch_rawobs_mlp_jumpchanges_ep10_seed0.progress.jsonl
```

## Combined Raw-Observation Transition And Tie-Policy Screens

These are the learned high-level module rows promoted into the main paper
narrative. They use the raw-observation MLP jump-change transition head and the
raw-observation tie-preserving policy head over the high-level cell
abstraction. The generated tables are:

- `results/paper_tables/pointmaze_tie_policy_head_ep20_evalseed012.csv`
- `results/paper_tables/pointmaze_tie_policy_head_ep20_tseed012.csv`
- `results/paper_tables/antmaze_rawobs_transition_tie_policy_head_ep10_tseed012.csv`
- `results/paper_tables/antmaze_rawobs_transition_tie_policy_head_ep10_evalseed012.csv`

PointMaze uses raw-observation MLP cell-change transitions plus a
raw-observation tie-policy head. The eval-seed table fixes transition seed 0
and aggregates eval seeds 0, 1, and 2. The commands below reproduce the
eval-seed 1/2 sources; seed 0 is
`results/pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv`
and
`results/pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv`.
The transition-seed table fixes eval seed 0 and aggregates transition seeds 0,
1, and 2. To reproduce transition seeds 1 and 2, run the same PointMaze
commands with `--seeds 0`, change `--transition-seed` to `1` or `2`, and use
output stems ending in `_tseed1` or `_tseed2`.

PointMaze navigate, transition seed 0, evaluation seeds 1 and 2:

```bash
env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
  scripts/run_pointmaze_learned_transition_planner.py \
  --env-name pointmaze-teleport-navigate-v0 \
  --topology-source dataset \
  --transition-model raw_obs_mlp \
  --transition-target-source dataset_cell_changes \
  --transition-steps 2000 \
  --transition-seed 0 \
  --transition-log-interval 1000 \
  --transition-mlp-lr 0.003 \
  --value-model raw_obs_tie_policy_mlp \
  --value-steps 3000 \
  --value-seed 0 \
  --value-log-interval 1000 \
  --value-mlp-lr 0.003 \
  --episodes 20 \
  --seeds 1 2 \
  --methods bellman_matched sto_trl_matched bellman_full \
  --profile-eval \
  --out results/pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_evalseed12.csv \
  --summary-out results/pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_evalseed12.json
```

PointMaze stitch, transition seed 0, evaluation seeds 1 and 2:

```bash
env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
  scripts/run_pointmaze_learned_transition_planner.py \
  --env-name pointmaze-teleport-stitch-v0 \
  --topology-source dataset \
  --transition-model raw_obs_mlp \
  --transition-target-source dataset_cell_changes \
  --transition-steps 2000 \
  --transition-seed 0 \
  --transition-log-interval 1000 \
  --transition-mlp-lr 0.003 \
  --value-model raw_obs_tie_policy_mlp \
  --value-steps 3000 \
  --value-seed 0 \
  --value-log-interval 1000 \
  --value-mlp-lr 0.003 \
  --episodes 20 \
  --seeds 1 2 \
  --methods bellman_matched sto_trl_matched bellman_full \
  --profile-eval \
  --out results/pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_evalseed12.csv \
  --summary-out results/pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_evalseed12.json
```

For AntMaze, the transition-seed table uses transition seeds 0, 1, and 2 with evaluation
seed 0. The eval-seed table fixes transition seed 0 and aggregates evaluation
seeds 0, 1, and 2. The commands below reproduce the eval-seed 1/2 files; run
the same commands with `--seeds 0` and output suffix `_seed0` for the seed-0
source, or use the already saved seed-0 sources
`results/antmaze_navigate_rawobs_transition_tie_policy_head_ep10_seed0.csv`
and
`results/antmaze_stitch_rawobs_transition_tie_policy_head_ep10_seed0.csv`.

AntMaze navigate, transition seed 0, evaluation seeds 1 and 2:

```bash
env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
  scripts/run_antmaze_bc_topology_planner.py \
  --env-name antmaze-teleport-navigate-v0 \
  --load-policy results/policies/antmaze_navigate_fullgoal_bc_50k.pkl \
  --transition-model raw_obs_mlp \
  --transition-target-source dataset_jump_changes \
  --transition-steps 1000 \
  --transition-log-interval 500 \
  --transition-mlp-lr 0.003 \
  --transition-seed 0 \
  --value-model raw_obs_tie_policy_mlp \
  --value-steps 2000 \
  --value-log-interval 1000 \
  --value-mlp-lr 0.003 \
  --value-seed 0 \
  --goal-representation full \
  --goal-candidates-per-state 16 \
  --goal-candidate-mode body_nearest \
  --bc-min-future 1 \
  --bc-max-future 30 \
  --bc-batch-size 2048 \
  --waypoint-lookahead 1 \
  --eval-action-repeat 1 \
  --policy-eval-backend jax \
  --methods bellman_matched sto_trl_matched bellman_full \
  --episodes 10 \
  --seeds 1 2 \
  --task-ids 4 5 \
  --out results/antmaze_navigate_rawobs_transition_tie_policy_head_ep10_tseed0_evalseed12.csv \
  --summary-out results/antmaze_navigate_rawobs_transition_tie_policy_head_ep10_tseed0_evalseed12.json \
  --progress-log results/antmaze_navigate_rawobs_transition_tie_policy_head_ep10_tseed0_evalseed12.progress.jsonl
```

AntMaze stitch, transition seed 0, evaluation seeds 1 and 2:

```bash
env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
  scripts/run_antmaze_bc_topology_planner.py \
  --env-name antmaze-teleport-stitch-v0 \
  --load-policy results/policies/antmaze_stitch_fullgoal_bc_20k.pkl \
  --transition-model raw_obs_mlp \
  --transition-target-source dataset_jump_changes \
  --transition-steps 1000 \
  --transition-log-interval 500 \
  --transition-mlp-lr 0.003 \
  --transition-seed 0 \
  --value-model raw_obs_tie_policy_mlp \
  --value-steps 2000 \
  --value-log-interval 1000 \
  --value-mlp-lr 0.003 \
  --value-seed 0 \
  --goal-representation full \
  --goal-candidates-per-state 16 \
  --goal-candidate-mode body_nearest \
  --bc-min-future 1 \
  --bc-max-future 30 \
  --bc-batch-size 2048 \
  --waypoint-lookahead 1 \
  --eval-action-repeat 1 \
  --policy-eval-backend jax \
  --methods bellman_matched sto_trl_matched bellman_full \
  --episodes 10 \
  --seeds 1 2 \
  --task-ids 4 5 \
  --out results/antmaze_stitch_rawobs_transition_tie_policy_head_ep10_tseed0_evalseed12.csv \
  --summary-out results/antmaze_stitch_rawobs_transition_tie_policy_head_ep10_tseed0_evalseed12.json \
  --progress-log results/antmaze_stitch_rawobs_transition_tie_policy_head_ep10_tseed0_evalseed12.progress.jsonl
```

## Current Main Claims

`scripts/verify_main_claims.py` checks the raw CSVs against these requirements:

- matched Bellman and stochastic TRL use the same 6-sweep planning budget
- full Bellman uses 180 sweeps
- stochastic TRL success is at least 0.90 on each main hard-task row
- matched Bellman success is at most 0.40
- stochastic TRL improves by at least 0.50 absolute success
- stochastic TRL exactly matches the 180-sweep Bellman reference
- AntMaze raw `bc_steps` metadata matches the headline controller budget
- hard-task stress rows match raw CSVs, stochastic TRL is at least 0.90, and
  stochastic TRL matches full Bellman
- the focused PointMaze teleport stitch task-5 table uses five seeds, 100
  episodes per seed, and preserves the matched-budget stochastic TRL gap
- learned-transition appendix screens match raw CSVs, use collapsed offline
  cell-change targets on PointMaze and 20 sampled outcomes per state-action row
  on AntMaze, and keep stochastic TRL at least 0.90 on PointMaze and 0.95 on
  AntMaze hard-task screens
- tie-preserving policy-head screens match raw CSVs, use the JAX policy
  backend, preserve action agreement thresholds, and keep stochastic TRL at or
  near full-Bellman success on PointMaze and AntMaze
- the combined AntMaze raw-observation transition plus tie-policy eval-seed
  table matches raw CSVs across evaluation seeds 0, 1, and 2 with transition
  seed 0 fixed

`scripts/verify_latex_claims.py` checks that the LaTeX table rows in
`paper/stochastic_trl/main.tex` match the generated paper-table CSVs exactly.

Current verified headline table:

| env | Bellman matched | Stochastic TRL | Bellman full | improvement |
| --- | ---: | ---: | ---: | ---: |
| PointMaze teleport navigate | 0.343 | 0.901 | 0.901 | +0.558 |
| PointMaze teleport stitch | 0.343 | 0.901 | 0.901 | +0.558 |
| AntMaze teleport navigate | 0.310 | 0.947 | 0.947 | +0.637 |
| AntMaze teleport stitch | 0.317 | 0.960 | 0.960 | +0.643 |

## PointMaze Learned-Controller Appendix Screen

The learned-controller appendix table is
`results/paper_tables/pointmaze_learned_controller_ep20_seed012.csv`. It uses
saved 5k-step full-goal BC policies and aggregates eval seeds 0, 1, and 2.
The seed-0 files are already saved separately; the commands below reproduce the
seed-1/2 files.

The controller-isolation appendix table is
`results/paper_tables/controller_execution_isolation.csv`. It combines the
learned-controller table below with the saved short direct-actor diagnostic
eval CSVs under `results/ogbench_screen_exp/dummy/`, and is regenerated by
`scripts/generate_paper_artifacts.py`.

```bash
env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
  scripts/run_antmaze_bc_topology_planner.py \
  --env-name pointmaze-teleport-navigate-v0 \
  --load-policy results/policies/pointmaze_navigate_fullgoal_bc_5k.pkl \
  --goal-representation full \
  --goal-candidates-per-state 16 \
  --goal-candidate-mode body_nearest \
  --bc-min-future 1 \
  --bc-max-future 30 \
  --bc-batch-size 2048 \
  --waypoint-lookahead 1 \
  --eval-action-repeat 1 \
  --policy-eval-backend jax \
  --methods bellman_matched sto_trl_matched bellman_full \
  --episodes 20 \
  --seeds 1 2 \
  --out results/pointmaze_bc_topology_navigate_fullgoal_5k_ep20_seed12.csv \
  --summary-out results/pointmaze_bc_topology_navigate_fullgoal_5k_ep20_seed12.json \
  --progress-log results/pointmaze_bc_topology_navigate_fullgoal_5k_ep20_seed12.progress.jsonl

env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY \
  scripts/run_antmaze_bc_topology_planner.py \
  --env-name pointmaze-teleport-stitch-v0 \
  --load-policy results/policies/pointmaze_stitch_fullgoal_bc_5k.pkl \
  --goal-representation full \
  --goal-candidates-per-state 16 \
  --goal-candidate-mode body_nearest \
  --bc-min-future 1 \
  --bc-max-future 30 \
  --bc-batch-size 2048 \
  --waypoint-lookahead 1 \
  --eval-action-repeat 1 \
  --policy-eval-backend jax \
  --methods bellman_matched sto_trl_matched bellman_full \
  --episodes 20 \
  --seeds 1 2 \
  --out results/pointmaze_bc_topology_stitch_fullgoal_5k_ep20_seed12.csv \
  --summary-out results/pointmaze_bc_topology_stitch_fullgoal_5k_ep20_seed12.json \
  --progress-log results/pointmaze_bc_topology_stitch_fullgoal_5k_ep20_seed12.progress.jsonl
```

## Caveats

- The main AntMaze table uses one saved task-specific BC controller per task
  with multiple evaluation seeds.
- Independent `bc_seed=1` AntMaze controller-seed screens are available at
  `results/paper_tables/antmaze_bcseed1_ep20_seed012.csv`; they use the same
  20-episode protocol as the headline AntMaze table.
- Three-controller navigate and stitch aggregates are available at
  `results/paper_tables/antmaze_navigate_controller_seeds_ep20_seed012.csv`
  and
  `results/paper_tables/antmaze_stitch_controller_seeds_ep20_seed012.csv`.
- The AntMaze result is a learned-controller topology diagnostic, not yet a
  complete end-to-end neural TRL replacement.
- CPU JAX is preferred for loaded AntMaze eval-only reproduction because the
  loop is dominated by environment stepping and many small actor calls.
