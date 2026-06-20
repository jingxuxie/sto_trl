# Stochastic TRL Paper Readiness Audit

Date: 2026-06-19.

## Current Verified Package

- Main hard-task table: `results/paper_tables/main_hard_task_results.csv`.
- Current fast-hard iteration table:
  `results/paper_tables/current_fast_hard_screen.csv`.
- Claim verifier: `results/main_claim_verification.md` reports `PASS`.
- LaTeX consistency verifier: `paper/stochastic_trl/latex_claim_verification.md`
  reports `PASS`.
- Compiled draft: `paper/stochastic_trl/main.pdf`; the main experiment section
  now includes the verified AntMaze learned high-level module diagnostic.
- Reproduction notes: `REPRODUCE_MAIN_RESULTS.md`.

## Strongest Claims

- Stochastic TRL reaches the 180-sweep Bellman reference at the matched
  6-sweep budget on PointMaze teleport navigate/stitch and AntMaze teleport
  navigate/stitch.
- Headline success is high: 0.901 on both PointMaze rows, 0.947 on AntMaze
  navigate, and 0.960 on AntMaze stitch.
- Matched Bellman remains low: 0.343 on PointMaze, 0.310 on AntMaze navigate,
  and 0.317 on AntMaze stitch.
- AntMaze robustness is covered by three independently trained BC controllers
  for both navigate and stitch.
- The support-TRL ablation shows deterministic support composition is not enough
  on PointMaze stitch.
- The generated hard-task stress table now focuses on tasks 4 and 5. On seed 0,
  PointMaze teleport stitch reaches 0.930 stochastic-TRL success over 50
  episodes per hard task versus 0.420 matched Bellman and 0.480 support TRL.
  AntMaze navigate/stitch reach 0.950/1.000 over 10 episodes per hard task
  versus 0.300/0.350 matched Bellman, matching full Bellman in both rows.
- The current fast-hard iteration table is now verifier-covered. It records the
  optimized exact PointMaze hard-slice proxy, current seed-0 PointMaze real-env
  checks, and current AntMaze hard-task checks with the full-Bellman reference
  included. Stochastic TRL reaches 1.000 on the PointMaze rows and 0.900/1.000
  on AntMaze navigate/stitch hard tasks, matching full Bellman in every row.
- The focused PointMaze task-5 appendix row gives a higher-episode single-task
  check: stochastic TRL 0.908 versus matched Bellman 0.380.
- New PointMaze learned-controller screens on teleport navigate and stitch use
  saved 5k-step full-observation BC controllers and reach 1.000 stochastic-TRL
  success over all five tasks at the matched 6-sweep budget across three
  evaluation seeds. Matched Bellman averages 0.323 on navigate and 0.223 on
  stitch; the 180-sweep Bellman reference reaches 1.000 on both. This is now a
  generated and verified appendix table, not a headline replacement.
- A new PointMaze learned-transition screen trains a tabular-softmax transition
  model from collapsed offline cell changes. On both teleport navigate and
  stitch, stochastic TRL reaches 0.916 success and matches the 180-sweep
  Bellman reference, while matched Bellman reaches 0.408. This is now a
  generated and verified appendix table, but not yet an end-to-end neural
  result.
- A new PointMaze raw-observation MLP transition screen trains a shared
  transition head on XY/action features using the same offline cell-change
  targets. Across three transition seeds, stochastic TRL again reaches 0.916
  success on both teleport navigate and stitch and matches the 180-sweep
  Bellman reference, while matched Bellman averages 0.512 on navigate and
  0.483 on stitch. This is now generated and verified appendix evidence for a
  less privileged learned transition module, but it still uses the cell
  abstraction.
- A PointMaze raw-observation transition plus tie-policy head eval-seed screen
  now aggregates eval seeds 0, 1, and 2 over all tasks. Stochastic TRL reaches
  0.927 on both navigate and stitch, matching full Bellman, while matched
  Bellman reaches 0.417. This strengthens the learned high-level module story
  beyond the original seed-0 screen.
- The same PointMaze raw-observation transition plus tie-policy head now has a
  transition-seed robustness screen. With eval seed 0 fixed, transition seeds
  0, 1, and 2 reach 0.980 stochastic-TRL success on both navigate and stitch,
  matching full Bellman while matched Bellman remains at 0.530.
- A new AntMaze learned-transition screen pairs the learned BC executor with a
  tabular-softmax high-level transition model trained from 20 sampled outcomes
  per row. A three-transition-seed robustness screen at 10 episodes per hard
  task keeps stochastic TRL at 0.950 on AntMaze navigate and 1.000 on AntMaze
  stitch, always matching full Bellman. Matched Bellman remains at 0.300 on
  navigate and averages 0.367 on stitch. This is now a generated and verified
  appendix table, not a headline replacement.
- A path-following diagnostic shows that persistent waypoint tracking is not a
  better default. On AntMaze navigate hard tasks 4 and 5 it reaches 0.917
  stochastic-TRL success across three evaluation seeds while matched Bellman
  falls to 0.000, but the existing greedy hard-task aggregate is slightly
  stronger. The same path mode hurts AntMaze stitch task 5, so greedy remains
  the safer default for the cross-task headline protocol.
- A less privileged AntMaze transition screen fits only observed offline
  jump/teleport rows and keeps local moves at the topology scaffold. On tasks
  4 and 5 with 10 episodes per task, stochastic TRL reaches 0.950 on navigate
  and 1.000 on stitch, matching full Bellman; matched Bellman reaches 0.300 and
  0.400. The direct all-cell-change AntMaze target is a negative boundary: it
  drops stochastic TRL/full Bellman to 0.500 in 5-episode hard-task screens.
- A new AntMaze raw-observation MLP transition screen trains a shared transition
  head on 29D cell-representative observations plus high-level action one-hot
  features for the observed jump-change rows. Across three transition seeds,
  on tasks 4 and 5 with 10 episodes per task, stochastic TRL reaches 0.950 on
  navigate and 1.000 on stitch, matching full Bellman; matched Bellman reaches
  0.300 on both.
- A new raw-observation PointMaze value/control-head diagnostic separates a
  failure mode from a promising representation. Scalar value and single-label
  policy MLP heads drop stitch success to 0.000, even though the same learned
  transition model with table Q-values reaches 0.980. A tie-preserving
  raw-observation policy head recovers 0.980 stochastic-TRL success on both
  navigate and stitch and matches the full-Bellman reference, while matched
  Bellman reaches 0.530. A previous-action-conditioned single-label policy
  head now recovers the same 0.980 success on both PointMaze variants with
  1000 value-head steps; the shared PointMaze evaluator needed the same sticky
  previous-action tie-break for 4D policy scores to avoid drifting out of the
  goal cell after arrival.
- The same tie-preserving high-level policy-head representation now has a
  compact AntMaze hard-task diagnostic with the learned BC executor and topology
  transition model: stochastic TRL reaches 0.925 on navigate and 0.950 on
  stitch on tasks 4 and 5 with 20 episodes per task, while matched Bellman
  reaches 0.350 and 0.400.
- A stronger combined AntMaze learned-module screen replaces the topology
  transition model with a raw-observation MLP jump-change transition head and
  keeps the tie-preserving high-level policy head. Across three transition
  seeds on tasks 4 and 5 with 10 episodes per task, stochastic TRL reaches
  0.950 on navigate and 1.000 on stitch, matching full Bellman; matched
  Bellman reaches 0.300 on both.
- With transition seed 0 fixed, the same combined AntMaze screen reaches 0.933
  on navigate and 0.967 on stitch across three evaluation seeds, matching full
  Bellman while matched Bellman reaches 0.283.
- A previous-action-conditioned single-label high-level policy MLP now fixes
  the sticky-tie failure mode beyond a single rollout seed. With a
  raw-observation MLP jump-change transition head and transition seed 0 fixed,
  it reaches 0.933 on navigate and 0.967 on stitch across evaluation seeds
  0, 1, and 2 with 10 episodes per hard task, staying within 0.02 of full
  Bellman while matched Bellman reaches 0.350 and 0.283. See
  `results/raw_obs_value_policy_diagnostic.md`.
- A bounded AntMaze all-task support screen uses the same raw-observation
  transition plus previous-action high-level policy head and the same saved BC
  controllers. On seed 0 with 3 episodes per task over all five tasks,
  stochastic TRL matches full Bellman at 0.933 on both navigate and stitch,
  while matched Bellman reaches 0.200. A 5-episode rerun remains positive:
  stochastic TRL matches full Bellman at 0.920 on navigate and 0.960 on stitch,
  while matched Bellman reaches 0.360. This is verifier-covered by
  `scripts/verify_antmaze_alltask_prev_policy.py` and summarized in
  `results/antmaze_alltask_prev_policy_summary.md`.
- A new AntMaze joint-action q-head diagnostic is positive on navigate hard
  tasks but exposes a stitch representation boundary. With the same
  raw-observation jump-change transition model and saved BC executors, a
  raw-observation q-head reaches 1.000 stochastic-TRL success on AntMaze
  navigate tasks 4 and 5, matching the learned full-Bellman q-head and beating
  the matched Bellman q-head at 0.333. On AntMaze stitch, the stochastic and
  full-Bellman q-heads both reach only 0.667 because task 5 remains fragile;
  increasing q-head fitting from 5000 to 10000 steps does not improve it.
  Adding previous-action conditioning to the q-head does not rescue stitch:
  stochastic and full-Bellman q-head success both drop to 0.000. This is
  verifier-covered by `scripts/verify_antmaze_qhead_hard_task.py` and
  summarized in `results/antmaze_qhead_hard_task_summary.md`.
- A new neural shortcut phase screen trains a small MLP stochastic-TRL critic
  from offline empirical transitions on stochastic risky-shortcut MDPs with
  safe lengths 16, 32, and 64. Across 12 safe-optimal settings, neural
  stochastic TRL reaches 1.000 exact success and 1.000 correct start-action
  rate with the logarithmic matched sweep budget. Neural Bellman TD reaches
  0.027 exact success under the same budget, and neural support-TRL reaches
  0.093 by over-composing the lucky risky shortcut. The table stochastic-TRL
  and full-Bellman references also reach 1.000. See
  `results/neural_shortcut_phase.md`,
  `results/paper_tables/neural_shortcut_phase.csv`, and
  `results/figures/neural_shortcut_phase.pdf`.
- A new PointMaze joint-action critic diagnostic moves the learned-critic
  bridge beyond the controlled shortcut MDPs. It trains a `Q_theta(s,g)` head
  that outputs all four high-level action values from raw-observation
  state-goal features plus state/goal identifiers, using the learned
  raw-observation transition head underneath. The target is generated
  internally by matched-budget stochastic transitive iteration from the learned
  transition model, then fitted by the q-head. In exact high-level model
  evaluation over all five tasks, the generated-target q-head reaches 1.000
  success on both PointMaze teleport navigate and stitch, matching table
  stochastic TRL and full Bellman. In real PointMaze environment rollouts with
  the proportional topology controller, the same q-head reaches 0.933 all-task
  success on both variants across evaluation seeds 0, 1, and 2, matching table
  stochastic TRL and full Bellman. In the seed-0 hard-task screen it reaches
  1.000 success on tasks 4 and 5 for both variants. The matched-budget q-head
  Bellman TD baseline reaches only 0.066/0.072 exact-model success and
  0.000/0.027 three-seed all-task real-env success, and the raw
  self-bootstrapped stochastic q-head collapses near zero. This is
  verifier-covered by `scripts/verify_qhead_critic_claim.py`,
  `scripts/verify_qhead_multiseed_env_claim.py`,
  `results/qhead_critic_claim_verification.md`, and
  `results/qhead_multiseed_env_claim_verification.md`.

## Claim Boundaries

- Continuous-control results are topology-planner diagnostics with learned
  executors, not end-to-end neural stochastic TRL.
- The new neural shortcut screen supports the learned value-function claim on
  controlled finite MDPs. It is not yet an OGBench raw-observation critic or a
  full actor-critic implementation.
- The PointMaze q-head critic screen is a two-phase generated-target diagnostic
  over the learned high-level cell MDP. It shows that the scalar value-head
  failure is not an unavoidable function-approximation barrier, and it avoids
  relying on an externally precomputed table target in the reported q-head row.
  The new real-env rows still use the topology controller, so they strengthen
  the learned-critic execution evidence without becoming an end-to-end actor
  result. It does not yet solve fully self-bootstrapped neural stochastic fitted
  iteration, so do not claim an end-to-end learned stochastic-TRL critic for
  OGBench.
- A target-buffered q-head variant with final consolidation is a useful
  intermediate diagnostic but not yet a headline result. The original
  warm-started target-iteration protocol solved PointMaze stitch but only
  navigate task 5. A reset-final variant now generates the same target buffer,
  then resets the q-head and optimizer before final consolidation; it reaches
  1.000 exact success on both PointMaze navigate and stitch all-task screens.
  A three-evaluation-seed real-environment support check reaches 0.933 on both
  variants, matching generated-target q-head, table stochastic TRL, and full
  Bellman. A top-4 bridge waypoint variant, which composes over only four
  retrieved waypoints per state-goal pair, preserves the same exact and
  real-environment success. This supports the target-buffered learned-critic
  route, while still relying on explicit target iteration over the learned
  high-level transition model.
- A sampled target-buffer variant reduces that explicit target-generation
  dependence in a bounded PointMaze support screen. It samples 64 offline
  next-state targets per high-level state-action row for Bellman calibration and
  samples 32 bridge candidates per state-goal pair, retaining four bridge
  waypoints per transitive backup before reset-final consolidation. It reaches
  1.000 exact success on both PointMaze teleport variants and 0.980 seed-0
  real-environment success on both variants with 10 episodes per task, matching
  table stochastic TRL and full Bellman in those rows. Its action agreement to
  full Bellman is lower than the all-target reset-final row, and this is still a
  high-level sampled-target diagnostic rather than fully self-bootstrapped
  neural fitted iteration. It is verifier-covered by
  `scripts/verify_qhead_sampled_buffered.py` and
  `results/qhead_sampled_buffered_verification.md`.
- A direct sampled target-network projection check is negative. With the same 64
  sampled next-state targets, 32 bridge candidates, and four retained bridge
  waypoints, but without accumulating a target buffer or doing reset-final
  consolidation, PointMaze navigate exact success reaches only 0.061. Increasing
  warmup and per-iteration fitting to 1000 steps drops success to 0.000. This is
  now verifier-covered by `scripts/verify_qhead_stabilization_boundaries.py` and
  supports the current boundary: sampled targets are useful, but projected
  self-bootstrapping still drifts.
- Two further online replay/consolidation ablations are negative. Removing the
  final reset-fit from the sampled target-buffer variant drops PointMaze navigate
  exact success to 0.000. A target-replay variant that accumulates projected
  q-head targets and resets every iteration for a 1000-step fit also reaches
  0.000. The positive recipe currently needs both monotone sampled target-buffer
  generation and a final clean projection onto that buffer.
- Two simple self-bootstrap stabilizers are also negative on the PointMaze
  navigate failure case. Adding `max(q, backup)` monotonicity leaves success at
  0.061, and a fitted-q monotone target buffer drops to 0.000 while the
  generated-target q-head remains at 1.000. Increasing the action-ranking loss
  to 1.0 also drops success to 0.000. A projected fresh-iteration variant that
  resets the q-head/optimizer for each intermediate target also reaches 0.000,
  so optimizer-state carryover is not the main failure mode. A guided-rank
  variant that uses generated-target action labels reaches only 0.119 even with
  `rank_ce_weight=10.0`, so action-label guidance alone is insufficient. The
  positive reset-final target-buffer result shows that final projection
  hysteresis can be fixed, but not that fully projected self-bootstrapping is
  solved. See
  `results/pointmaze_qhead_self_bootstrap_stabilization.md`.
- Do not claim the raw-observation value/control head is generally solved.
  The tie-preserving PointMaze and AntMaze policy-head screens are positive
  diagnostics over the high-level abstraction. The previous-action-conditioned
  policy head is now positive on PointMaze in a single-seed all-task screen and
  on AntMaze in a three-eval-seed hard-task screen. The scalar value and
  previous-action-free single-label policy heads remain failure boundaries. The
  AntMaze raw-observation q-head is positive on navigate hard tasks but does not
  yet solve stitch task 5, even with 10000 fitting steps; a previous-action
  q-head variant also fails on stitch with 0.000 stochastic/full success.
- GCFBC is not main evidence. After waypoint fixes it reaches 0.600 on one
  AntMaze task-3 screen after 20k updates, below the verified MSE BC executor.
  A 10k-update GPU hard-task screen on AntMaze navigate tasks 4 and 5 reaches
  0.000 success, so the TRL-paper-style flow controller is still a controller
  bottleneck rather than a replacement for the verified executor.
- The official TRL neural-codepath PointMaze screens are not main evidence.
  Clean direct-actor screens in the official runner remain low: GCFBC peaks at
  0.080 success, deterministic MSEBC peaks at 0.120, and existing TRL/log/relax
  screens stay in the 0.000-0.100 range. This suggests the official neural loop
  is currently actor/controller-limited before it isolates the stochastic
  critic question; see `results/neural_codepath_diagnostic.md`.
- The direct-actor boundary is not a contradiction of the learned-controller
  PointMaze result: the explicit stochastic-TRL waypoint executor reaches
  1.000 on both PointMaze teleport variants over three evaluation seeds. The
  takeaway is that long-horizon waypoint structure is currently essential for
  learned execution. The generated source table is
  `results/paper_tables/controller_execution_isolation.csv`.
- The empirical graph planner is a useful less-privileged PointMaze navigate
  diagnostic, but not a complete replacement for the topology scaffold. On
  `pointmaze-teleport-stitch-v0`, the same independent-seed graph protocol
  reaches only 0.144 stochastic-TRL success, 0.168 matched Bellman success, and
  0.078 full-Bellman success. Because the 220-sweep reference also fails, treat
  this as a graph/executor boundary, not an algorithmic contradiction of the
  high-success topology stitch result. A follow-up task-4 executor sweep found
  no simple rescue: the best 220-sweep Bellman executor was `transition_value`
  at only 0.133 mean success over three seeds. Replacing the proportional
  waypoint tracker with the saved PointMaze full-goal BC controller also scored
  0.000 on a 10-episode seed-0 task-4 check, so the current failure is mainly
  the empirical graph path/representation rather than the local controller.
- The method should be described as horizon-efficient matching of a long-sweep
  Bellman reference, not as beating full Bellman.

## Remaining High-Impact Work

1. Turn the positive PointMaze generated-target q-head diagnostic into a robust
   self-bootstrapped stochastic-TRL critic. The current q-head architecture can
   represent and execute the generated target; raw self-bootstrapping collapses,
   and simple monotone/self-buffered/ranking-weight/projected
   fresh-iteration/guided-rank fitted updates do not fix the navigate failure
   case. A reset-final target buffer now solves PointMaze navigate/stitch in the
   exact high-level model and in a three-evaluation-seed real-environment
   support screen, including with only four retrieved bridge waypoints per
   state-goal pair. A sampled target-buffer variant now shows that sampled
   offline next-state and bridge targets can preserve high success on PointMaze.
   Direct sampled target-network projection, no-final sampled buffering, and
   reset-each-iteration target replay all failed, so the next step is a stronger
   stabilization mechanism: conservative route consistency, route-supervised
   ranking, or a representation change that prevents projected target drift
   without an explicit final table.
2. Extend the raw-observation MLP transition screens beyond the current
   cell-abstraction setting, ideally to a differentiable auxiliary loss when
   time permits.
3. Improve the AntMaze stitch q-head representation. Previous-action
   conditioning with the current q-head loss failed, so the next plausible
   directions are a route-consistency auxiliary objective, a different
   conservative target, or a stronger action-ranking loss tied to successful
   routes.
4. Convert the current article-style draft into the exact target venue template
   once the target venue is chosen.

## Completion Status

The package is strong enough for a serious draft and internal review. The new
neural shortcut phase screen closes the smallest learned-critic gap. The full
research objective is not yet complete because the benchmark learned-critic
story should still move beyond controlled finite MDPs and high-level
cell-abstraction screens.
