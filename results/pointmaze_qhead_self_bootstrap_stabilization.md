# PointMaze Q-Head Self-Bootstrap Stabilization Diagnostic

Date: 2026-06-19

## Question

The positive PointMaze learned-critic result uses a two-phase procedure: generate
the matched-budget stochastic transitive target from the learned transition
model, then fit a fresh joint-action q-head. This diagnostic checks whether two
small self-bootstrapped stabilizers can close the remaining gap:

- `qhead_sto_trl_monotone`: backs up from the fitted q-head but fits
  `max(q, Bellman(q), Transitive(q))`.
- `qhead_sto_trl_self_buffered`: backs up from the fitted q-head, stores all
  discovered targets in a monotone buffer, and does final consolidation on that
  buffer.
- `qhead_sto_trl_fresh_iter`: backs up from the fitted q-head, then fits each
  intermediate operator target with a freshly initialized q-head/optimizer.
- `qhead_sto_trl_guided_rank`: uses self-bootstrapped value targets but takes
  the action-ranking cross-entropy labels from the internally generated
  stochastic-TRL target.
- `qhead_sto_trl_buffered_reset_final`: generates the same monotone stochastic
  target buffer as the target-buffered variant, then resets the q-head and
  optimizer before the final consolidation fit.
- high action-ranking loss: repeats the raw and monotone self-bootstrap with
  `rank_ce_weight=1.0` instead of `0.05`.

## Setting

- Environment: `pointmaze-teleport-navigate-v0`.
- Transition model: raw-observation MLP trained on dataset cell changes.
- Features: raw-observation state/goal features plus state/goal one-hot IDs.
- Evaluation: exact high-level model rollout over all five tasks.
- Matched budget: 6 stochastic/transitive iterations.
- Final consolidation for self-buffered: 5000 gradient steps.

## Result

| method | success | action agreement to full |
| --- | ---: | ---: |
| qhead self-bootstrap | 0.061 | 0.384 |
| qhead monotone self-bootstrap | 0.061 | 0.384 |
| qhead fitted-q self-buffered | 0.000 | 0.416 |
| qhead fresh projected iteration | 0.000 | 0.509 |
| qhead guided rank, rank_ce=1 | 0.000 | 0.349 |
| qhead guided rank, rank_ce=10 | 0.119 | 0.431 |
| qhead sampled target-network projection | 0.061 | 0.355 |
| qhead sampled target-network projection, heavier fit | 0.000 | 0.335 |
| qhead sampled target buffer without final reset | 0.000 | 0.603 |
| qhead sampled target replay reset-each-iteration | 0.000 | 0.385 |
| qhead target buffer + reset-final | 1.000 | 0.966 |
| qhead top-4 bridge target buffer + reset-final | 1.000 | 0.965 |
| qhead sampled target buffer + reset-final | 1.000 | 0.933 |
| qhead generated stochastic target | 1.000 | 0.966 |
| table stochastic TRL | 1.000 | 0.988 |
| table full Bellman | 1.000 | 1.000 |

Three-evaluation-seed real-environment support with the proportional PointMaze
controller:

| env | top-4 q-head, seeds 0/1/2 | top-4 table/full refs, seeds 0/1/2 | sampled q-head, seed 0 | sampled table/full refs, seed 0 |
| --- | ---: | ---: | ---: | ---: |
| pointmaze-teleport-navigate-v0 | 0.933 | 0.933 | 0.980 | 0.980 |
| pointmaze-teleport-stitch-v0 | 0.933 | 0.933 | 0.980 | 0.980 |

The high action-ranking-loss run was also negative:

| method | rank_ce_weight | success | action agreement to full |
| --- | ---: | ---: | ---: |
| qhead self-bootstrap | 1.0 | 0.000 | 0.354 |
| qhead monotone self-bootstrap | 1.0 | 0.000 | 0.354 |

Raw files:

- `results/pointmaze_qhead_monotone_navigate_all_exact.csv`
- `results/pointmaze_qhead_self_buffered_navigate_all_exact.csv`
- `results/pointmaze_qhead_rank1_navigate_all_exact.csv`
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
- `results/qhead_buffered_reset_final_verification.md`
- `results/qhead_sampled_buffered_verification.md`
- `results/qhead_stabilization_boundary_verification.md`

## Interpretation

Simple monotonicity is not enough: the monotone fitted target follows the same
policy as the raw self-bootstrapped q-head. A fitted-q target buffer is also not
enough and can make the exact policy worse even when its supervised fit loss is
lower. Increasing the action-ranking loss does not rescue the policy and
instead worsens exact success. The successful generated-target q-head therefore
remains a real two-phase result rather than a solved fully self-bootstrapped
fitted-iteration algorithm.

A June 20 projected fitted-iteration check was also negative. It reset the
q-head and optimizer for each intermediate stochastic-TRL operator target, so
optimizer-state carryover was not the only issue. The intermediate targets were
fit to low MSE, but exact model success stayed at 0.000 and action agreement was
only 0.509. The remaining failure is therefore the projected self-bootstrap path
itself: early approximation errors steer later transitive targets into a poor
policy even when each individual projection fits its current target.

A guided-ranking variant was also negative. With generated stochastic-TRL action
labels and self-bootstrapped value targets, `rank_ce_weight=1.0` stayed at 0.000
success. Raising the rank weight to 10.0 increased success only to 0.119, still
far below the 1.000 generated-target q-head. This suggests that action-ranking
guidance alone is insufficient once the self-bootstrapped value targets drift.

A June 20 reset-final target-buffer check was positive. The old target-buffered
q-head with warm-started final consolidation reached only 0.200 success on
PointMaze navigate, solving task 5 but not tasks 1-4. Resetting the q-head and
optimizer before the final 5000-step consolidation fit raises navigate all-task
exact success to 1.000, matching the generated-target q-head and table
stochastic TRL. The same reset-final variant also reaches 1.000 on PointMaze
stitch. This shows that the generated stochastic target buffer is useful and
that the previous buffered failure was final-fit hysteresis, not a bad target.
It is still not a fully self-bootstrapped critic: the buffer is generated by
explicit target iteration over the learned high-level transition model rather
than by backing up from the projected q-head itself.

The same reset-final q-head also survives a real-environment check with
evaluation seeds 0, 1, and 2 and 10 episodes per task on both PointMaze
teleport variants, reaching 0.933 success and matching generated-target
q-head, table stochastic TRL, and full Bellman. This gives the reset-final
target-buffer result the same real-env rollout scope as the existing
generated-target q-head claim.

A less privileged top-K bridge variant also succeeds. Instead of composing over
all 45 possible waypoints, it retrieves only the top 4 candidate bridge states
per state-goal pair by current value composition, then applies the reset-final
target-buffer fit. This K=4 variant reaches 1.000 exact success and 0.933
three-evaluation-seed real-environment success on both PointMaze teleport
variants, again matching generated-target q-head, table stochastic TRL, and
full Bellman.

A sampled target-buffer variant also succeeds in the first bounded screen. It
draws 64 offline next-state samples per high-level state-action row for Bellman
calibration and samples 32 bridge candidates per state-goal pair, retaining only
the top 4 bridge waypoints for each transitive backup. With the same reset-final
consolidation, it reaches 1.000 exact success on both PointMaze teleport
variants. In a seed-0 real-environment support check with 10 episodes per task,
it reaches 0.980 on both variants and matches table stochastic TRL and full
Bellman. This reduces the explicit all-state target-generation dependence, but
it is still a high-level sampled-target diagnostic rather than a fully
self-bootstrapped projected critic.

The direct sampled target-network projection remains negative. It uses the same
64 sampled next-state targets and 32 sampled bridge candidates, but computes each
operator target from the current q-head snapshot and trains the same q-head
directly, without accumulating a target buffer or performing a final reset-fit
to that buffer. On PointMaze navigate exact evaluation, the default fit reaches
only 0.061 success and action agreement 0.355. Increasing warmup to 1000 steps
and per-iteration fitting to 1000 steps drops success to 0.000. This reinforces
the current diagnosis: sampled targets are sufficient when stabilized by a
target buffer, but projected self-bootstrapping still drifts.

Two additional replay/consolidation ablations are also negative. Removing the
final reset-fit from the sampled target-buffer variant drops PointMaze navigate
exact success to 0.000 despite the same sampled target buffer. A target-replay
variant that resets every iteration, accumulates projected q-head targets, and
refits from scratch for 1000 steps at every iteration also reaches 0.000. The
current positive recipe therefore needs both monotone sampled target-buffer
generation and a final clean projection onto that buffer.
