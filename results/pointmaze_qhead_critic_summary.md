# PointMaze Joint-Action Critic Diagnostic

Date: 2026-06-19

## Question

The earlier scalar PointMaze neural critic failed even with low supervised value
loss. This diagnostic tests whether the failure is caused by the scalar
per-action critic interface scrambling action rankings. The replacement critic
uses a joint-action head:

```text
Q_theta(s, g) -> R^4
```

It is trained on raw-observation state/goal features plus state/goal one-hot
features, with a learned raw-observation MLP transition model fixed underneath.
Evaluation uses both exact high-level model rollout over the induced stochastic
PointMaze cell MDP and real PointMaze environment rollout with the existing
proportional cell controller.

## All-Task Exact Model Results

| env | method | success | action agreement to full |
| --- | --- | ---: | ---: |
| pointmaze-teleport-navigate-v0 | qhead Bellman TD | 0.066 | 0.551 |
| pointmaze-teleport-navigate-v0 | qhead generated stochastic target | 1.000 | 0.966 |
| pointmaze-teleport-navigate-v0 | qhead full-Bellman target critic | 1.000 | 0.978 |
| pointmaze-teleport-navigate-v0 | table matched Bellman | 0.476 | 0.739 |
| pointmaze-teleport-navigate-v0 | table stochastic TRL | 1.000 | 0.988 |
| pointmaze-teleport-navigate-v0 | table full Bellman | 1.000 | 1.000 |
| pointmaze-teleport-stitch-v0 | qhead Bellman TD | 0.072 | 0.555 |
| pointmaze-teleport-stitch-v0 | qhead generated stochastic target | 1.000 | 0.959 |
| pointmaze-teleport-stitch-v0 | qhead full-Bellman target critic | 1.000 | 0.977 |
| pointmaze-teleport-stitch-v0 | table matched Bellman | 0.481 | 0.716 |
| pointmaze-teleport-stitch-v0 | table stochastic TRL | 1.000 | 0.981 |
| pointmaze-teleport-stitch-v0 | table full Bellman | 1.000 | 1.000 |

## Real Environment Rollout Results

Real environment rollouts with the same proportional PointMaze controller used
by the topology planner:

| env | task scope | eval seeds | episodes/task | qhead Bellman TD | qhead generated stochastic target | table matched Bellman | table stochastic TRL | table full Bellman |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pointmaze-teleport-navigate-v0 | all tasks | 0,1,2 | 10 | 0.000 | 0.933 | 0.393 | 0.933 | 0.933 |
| pointmaze-teleport-stitch-v0 | all tasks | 0,1,2 | 10 | 0.027 | 0.933 | 0.393 | 0.933 | 0.933 |
| pointmaze-teleport-navigate-v0 | all tasks | 0 | 10 | 0.000 | 0.980 | 0.520 | 0.980 | 0.980 |
| pointmaze-teleport-stitch-v0 | all tasks | 0 | 10 | 0.040 | 0.980 | 0.520 | 0.980 | 0.980 |
| pointmaze-teleport-navigate-v0 | tasks 4,5 | 0 | 20 | 0.000 | 1.000 | 0.500 | 1.000 | 1.000 |
| pointmaze-teleport-stitch-v0 | tasks 4,5 | 0 | 20 | 0.125 | 1.000 | 0.500 | 1.000 | 1.000 |

Raw files:

- `results/pointmaze_qhead_critic_navigate_all_exact.csv`
- `results/pointmaze_qhead_critic_stitch_all_exact.csv`
- `results/pointmaze_qhead_target_fit_navigate_all_exact.csv`
- `results/pointmaze_qhead_target_fit_stitch_all_exact.csv`
- `results/pointmaze_qhead_target_fit_navigate_all_env_ep10_seed012.csv`
- `results/pointmaze_qhead_target_fit_stitch_all_env_ep10_seed012.csv`
- `results/pointmaze_qhead_target_fit_navigate_all_env_ep10_seed0.csv`
- `results/pointmaze_qhead_target_fit_stitch_all_env_ep10_seed0.csv`
- `results/pointmaze_qhead_target_fit_navigate_task45_env_ep20_seed0.csv`
- `results/pointmaze_qhead_target_fit_stitch_task45_env_ep20_seed0.csv`
- `results/pointmaze_qhead_critic_stitch_task45_screen.csv`
- `results/pointmaze_qhead_critic_stitch_task45_onehot_heavy.csv`
- `results/pointmaze_qhead_buffered_stitch_task45_consolidated.csv`
- `results/pointmaze_qhead_buffered_navigate_all_exact.csv`
- `results/pointmaze_qhead_buffered_stitch_all_exact.csv`
- `results/pointmaze_qhead_target_iter_navigate_all_exact.csv`
- `results/pointmaze_qhead_fresh_iter_navigate_all_exact.csv`
- `results/pointmaze_qhead_guided_rank_navigate_all_exact.csv`
- `results/pointmaze_qhead_guided_rank10_navigate_all_exact.csv`
- `results/pointmaze_qhead_buffered_reset_final_navigate_all_exact.csv`
- `results/pointmaze_qhead_buffered_reset_final_stitch_all_exact.csv`
- `results/pointmaze_qhead_buffered_reset_final_navigate_all_env_ep10_seed0.csv`
- `results/pointmaze_qhead_buffered_reset_final_stitch_all_env_ep10_seed0.csv`
- `results/pointmaze_qhead_buffered_reset_final_navigate_all_env_ep10_seed012.csv`
- `results/pointmaze_qhead_buffered_reset_final_stitch_all_env_ep10_seed012.csv`
- `results/pointmaze_qhead_buffered_topk4_reset_final_navigate_all_exact.csv`
- `results/pointmaze_qhead_buffered_topk4_reset_final_stitch_all_exact.csv`
- `results/pointmaze_qhead_buffered_topk4_reset_final_navigate_all_env_ep10_seed012.csv`
- `results/pointmaze_qhead_buffered_topk4_reset_final_stitch_all_env_ep10_seed012.csv`
- `results/pointmaze_qhead_sampled_buffered_nav_exact_s64_c32_k4.csv`
- `results/pointmaze_qhead_sampled_buffered_stitch_exact_s64_c32_k4.csv`
- `results/pointmaze_qhead_sampled_buffered_nav_env_ep10_seed0_s64_c32_k4.csv`
- `results/pointmaze_qhead_sampled_buffered_stitch_env_ep10_seed0_s64_c32_k4.csv`
- `results/pointmaze_qhead_sampled_buffered_nav_exact_s64_c32_k4_no_final.csv`
- `results/pointmaze_qhead_sampled_target_net_nav_exact_s64_c32_k4.csv`
- `results/pointmaze_qhead_sampled_target_net_nav_exact_s64_c32_k4_heavy.csv`
- `results/pointmaze_qhead_sampled_target_replay_nav_exact_s64_c32_k4.csv`
- `results/pointmaze_qhead_multiseed_env_summary.md`
- `results/qhead_stabilization_boundary_verification.md`
- `results/qhead_buffered_reset_final_verification.md`
- `results/qhead_sampled_buffered_verification.md`
- `results/qhead_multiseed_env_claim_verification.md`

Verification:

- `scripts/verify_qhead_critic_claim.py`
- `results/qhead_critic_claim_verification.md`
- `scripts/verify_qhead_multiseed_env_claim.py`
- `results/qhead_multiseed_env_claim_verification.md`
- `scripts/verify_qhead_stabilization_boundaries.py`
- `results/qhead_stabilization_boundary_verification.md`

## Interpretation

This is a positive learned-critic bridge beyond the previous scalar value-head
failure. A joint-action critic can represent and execute a stochastic-TRL target
that is generated internally from the learned transition model by matched-budget
stochastic transitive iteration. It matches the table stochastic and
full-Bellman references on both PointMaze teleport variants in exact high-level
model evaluation. The same q-head also matches the table stochastic and
full-Bellman policies in real environment rollouts: 0.933 success over all
five tasks across evaluation seeds 0, 1, and 2, and 1.000 success on hard tasks
4 and 5 in the seed-0 screen for both navigate and stitch. The Bellman TD
q-head remains low under the matched budget, so the result still isolates the
long-horizon stochastic target.

The raw self-bootstrapped fitted-iteration q-head is not solved yet. On the
stitch hard slice, `qhead_sto_trl` still collapses to near-zero exact success
even with heavier fitting and one-hot features. A target-buffered variant with
final consolidation is partially positive: it reaches 1.000 exact success on
PointMaze stitch across all tasks and on stitch hard tasks 4 and 5, while the
same protocol solves only task 5 on PointMaze navigate. A diagnostic confirmed
that the internally generated target-iteration buffer exactly matches the table
stochastic-TRL target on navigate; the failure is therefore an optimization-path
issue from warm-starting through intermediate targets, not an error in the
stochastic transitive target. A reset-final version of the same target-buffered
protocol fixes that optimization-path issue: after generating the buffer, it
resets the q-head and optimizer before final consolidation, reaching 1.000
exact success on both PointMaze navigate and stitch all-task screens. A
three-evaluation-seed real-environment support check with 10 episodes per task
also reaches 0.933 on both variants, matching generated-target q-head, table
stochastic TRL, and full Bellman.
The same result survives a top-4 bridge waypoint restriction: the target-buffer
update retrieves only four candidate waypoints per state-goal pair, yet still
reaches 1.000 exact success and 0.933 three-evaluation-seed real-environment
success on both PointMaze teleport variants.

June 19 stabilization check: two direct self-bootstrap fixes also failed on the
PointMaze navigate all-task exact screen. `qhead_sto_trl_monotone`, which fits
`max(q, Bellman(q), Transitive(q))`, matched the raw self-bootstrap at 0.061
success. `qhead_sto_trl_self_buffered`, which stores fitted-q backups in a
monotone target buffer and performs 5000 final consolidation steps, dropped to
0.000 success. Raising `rank_ce_weight` from 0.05 to 1.0 also dropped raw and
monotone self-bootstrap success to 0.000. The generated-target q-head stayed at
1.000 in the comparable run. See
`results/pointmaze_qhead_self_bootstrap_stabilization.md`.

June 20 projected fresh-iteration check: resetting the q-head and optimizer for
each intermediate stochastic-TRL target also failed, with 0.000 exact success
and 0.509 action agreement on PointMaze navigate. This rules out optimizer-state
carryover as the main bottleneck.

June 20 guided-rank check: adding generated-target action-ranking labels to the
self-bootstrapped value fit also failed. `rank_ce_weight=1.0` stayed at 0.000
exact success; `rank_ce_weight=10.0` reached only 0.119 exact success. This
suggests the self-bootstrapped value-target drift must be controlled, not only
the final action labels.

June 20 reset-final target-buffer check: the positive reset-final result closes
the earlier target-buffered navigate failure without changing the target
generator. It should be described as a stronger target-buffered learned-critic
bridge, not as a solved projected self-bootstrap loop, because the target buffer
is still generated explicitly from the learned high-level transition model.
The real-environment support screen now has the same three-eval-seed rollout
scope as the generated-target q-head rows, but the algorithmic caveat remains:
the target buffer is still explicitly generated from the learned high-level
transition model.
The top-4 bridge variant reduces the all-waypoint dependence, but it still uses
the learned high-level transition matrix for Bellman calibration and waypoint
target generation.

June 20 sampled target-buffer check: replacing the explicit matrix backup with
sampled target generation is also positive in a bounded PointMaze screen. The
new `qhead_sto_trl_sampled_buffered_reset_final` method draws 64 sampled
next-state targets per high-level row and samples 32 bridge candidates per
state-goal pair, keeping only four bridge waypoints per transitive backup before
the reset-final consolidation fit. It reaches 1.000 exact success on both
PointMaze teleport variants. In seed-0 real-environment checks with 10 episodes
per task, it reaches 0.980 on both variants, matching table stochastic TRL and
full Bellman. Its action agreement to full Bellman is lower than the all-target
buffered row, so this is a high-success sampled-target bridge rather than a
replacement for the more thoroughly verified generated-target q-head claim.

June 20 sampled target-network projection check: removing the explicit target
buffer is still negative. With the same 64 sampled next-state targets, 32 bridge
candidates, and four retained bridge waypoints, direct projected updates from
the current q-head snapshot reach only 0.061 exact success on PointMaze navigate.
A heavier 1000-warmup/1000-step-per-iteration fit drops to 0.000. This shows the
new sampled-target success comes from stabilizing target generation in a buffer,
not merely from replacing all-state backups with samples.

June 20 replay/consolidation ablations: the final clean projection is still
necessary. The sampled target-buffer variant without final reset-fit reaches
0.000 exact success on PointMaze navigate. A target-replay variant that
accumulates projected q-head targets and resets the q-head every iteration for a
1000-step fit also reaches 0.000. The positive sampled-buffer result therefore
depends on both monotone target-buffer generation and the final reset
consolidation fit.

For the paper, the safe claim is:

> a joint-action learned critic can carry an operator-generated stochastic
> transitive target on PointMaze cell features and execute it through the
> PointMaze controller across three evaluation seeds, while fully
> self-bootstrapped neural fitted iteration remains a limitation and next step.
