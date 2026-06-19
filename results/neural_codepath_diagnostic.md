# Official TRL Neural-Codepath Diagnostic

Date: 2026-06-19.

## Question

Can the current official TRL training/evaluation loop already provide a
positive neural end-to-end result on OGBench PointMaze teleport, or is the
remaining gap mostly an actor/controller bottleneck?

## New Diagnostic

I added clean `gcfbc` support to `scripts/run_pointmaze_screen.py` so the
screen runner can launch the official `external/trl/agents/gcfbc.py` baseline
without injecting TRL-specific critic or RPG config overrides. I then added a
minimal deterministic MSE BC agent at `external/trl/agents/msebc.py` and
registered it in the official agent registry to test whether the flow actor was
the main issue.

Command:

```bash
env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/eston/anaconda3/envs/autoresearcher_sto_trl/bin/python \
  scripts/run_pointmaze_screen.py \
  --env-name pointmaze-teleport-navigate-v0 \
  --agents gcfbc \
  --steps 10000 \
  --log-interval 2000 \
  --eval-interval 5000 \
  --eval-episodes 5 \
  --batch-size 256 \
  --hidden-dims '(128,128)' \
  --run-group pointmaze_gcfbc_10k_actor_diag_cpu \
  --cpu
```

Output directory:

`results/ogbench_screen_exp/dummy/pointmaze_gcfbc_10k_actor_diag_cpu/sd000_20260619_104221`

GCFBC result:

| agent | step | task1 | task2 | task3 | task4 | task5 | overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GCFBC | 5000 | 0.000 | 0.000 | 0.000 | 0.400 | 0.000 | 0.080 |
| GCFBC | 10000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

MSE BC commands:

```bash
env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/eston/anaconda3/envs/autoresearcher_sto_trl/bin/python \
  scripts/run_pointmaze_screen.py \
  --env-name pointmaze-teleport-navigate-v0 \
  --agents msebc \
  --steps 10000 \
  --log-interval 2000 \
  --eval-interval 5000 \
  --eval-episodes 5 \
  --batch-size 256 \
  --hidden-dims '(128,128)' \
  --run-group pointmaze_msebc_10k_actor_diag_cpu \
  --cpu

env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/eston/anaconda3/envs/autoresearcher_sto_trl/bin/python \
  scripts/run_pointmaze_screen.py \
  --env-name pointmaze-teleport-navigate-v0 \
  --agents msebc \
  --steps 10000 \
  --log-interval 2000 \
  --eval-interval 5000 \
  --eval-episodes 5 \
  --batch-size 512 \
  --hidden-dims '(256,256)' \
  --layer-norm \
  --run-group pointmaze_msebc_10k_ln256_actor_diag_cpu \
  --cpu
```

MSE BC results:

| agent | setting | step | task1 | task2 | task3 | task4 | task5 | overall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MSEBC | 128,128 | 5000 | 0.200 | 0.000 | 0.000 | 0.400 | 0.000 | 0.120 |
| MSEBC | 128,128 | 10000 | 0.000 | 0.000 | 0.000 | 0.400 | 0.000 | 0.080 |
| MSEBC | LN 256,256 | 5000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| MSEBC | LN 256,256 | 10000 | 0.000 | 0.000 | 0.000 | 0.600 | 0.000 | 0.120 |

These screens are intentionally short. They are not final BC benchmarks, but
they show that under the same official evaluation surface, direct pure actor
baselines do not quickly solve PointMaze teleport. The deterministic MSE actor
improves slightly over GCFBC but still peaks at only `0.120` overall success,
concentrated on task 4. This makes the existing weak neural TRL screens less
informative as algorithm evidence: poor success can arise from the official
actor/controller path before the stochastic critic question is isolated.

## Existing Neural TRL Screens

`results/ogbench_screen_report.md` now reports both final and best evaluation
success. The existing official-codepath screens are all low on PointMaze
teleport:

- Original TRL and log/MC/relax variants at 3k to 30k updates: best success
  is generally `0.000` to `0.100`.
- `trl_log_relax_mc` at the longer 100k-update screen reports `0.100` at the
  available 50k evaluation.
- The new GCFBC actor diagnostic peaks at `0.080` at 5k and drops to `0.000`
  at 10k.
- The new deterministic MSE BC diagnostics peak at `0.120`.

## Interpretation

This is a negative boundary for the current paper, not a failure of the
stochastic TRL value-propagation claim. The high-level stochastic planner
already reaches high success with topology/BC executors and learned
raw-observation high-level transition plus tie-policy heads. The official
neural codepath still mixes several unresolved issues:

- actor training quality under the official PointMaze teleport evaluation;
- long-horizon goal reaching without explicit high-level waypoint execution;
- stochastic critic calibration and transitive propagation.

For a high-impact paper, the current strongest evidence should remain the
verified high-level stochastic TRL results. A full neural TRL claim needs a
separate short screen that first gets the actor/controller path above a
reasonable success threshold.

## Direct Actor Versus Waypoint Execution

The sharpest current controller diagnostic is the contrast between direct
final-goal actors in the official loop and an explicit waypoint-conditioned
learned executor. The waypoint executor is still learned from data: it is a
5k-step full-observation MSE BC controller, but it is driven by the
stochastic-TRL high-level waypoint policy with body-nearest `k=16` waypoint
observations.

Sources:

- Generated comparison:
  `results/paper_tables/controller_execution_isolation.csv`.
- Direct actors: `results/ogbench_screen_report.md`.
- Waypoint executor:
  `results/paper_tables/pointmaze_learned_controller_ep20_seed012.csv`.

| execution path | env scope | eval setting | best/final success |
| --- | --- | --- | ---: |
| official GCFBC direct final-goal actor | PointMaze teleport navigate | seed 0, 5 eps/task | 0.080 best, 0.000 final |
| official MSEBC direct final-goal actor | PointMaze teleport navigate | seed 0, 5 eps/task | 0.120 best, 0.120 final |
| stochastic TRL + learned waypoint BC | PointMaze teleport navigate | 3 eval seeds, 20 eps/task | 1.000 |
| stochastic TRL + learned waypoint BC | PointMaze teleport stitch | 3 eval seeds, 20 eps/task | 1.000 |

This supports a clear experimental boundary: direct goal-conditioned neural
execution is the current bottleneck, while stochastic TRL supplies the
long-horizon waypoint structure needed for the learned executor to succeed.
For the paper, the learned-controller result should be framed as a positive
planner-plus-learned-executor diagnostic, and the official direct-actor screens
should stay as negative boundary evidence.

## Next Action

Do not spend long runs on direct official neural TRL variants until a short
actor or waypoint-conditioned neural screen reaches at least `0.50` to `0.70`
success on a PointMaze teleport task. The more promising next route is explicit
waypoint-conditioned execution or a differentiable high-level auxiliary loss
that preserves the currently verified stochastic planner behavior.
