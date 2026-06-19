# Stochastic TRL Paper Readiness Audit

Date: 2026-06-19.

## Current Verified Package

- Main hard-task table: `results/paper_tables/main_hard_task_results.csv`.
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

## Claim Boundaries

- Continuous-control results are topology-planner diagnostics with learned
  executors, not end-to-end neural stochastic TRL.
- Do not claim the raw-observation value/control head is generally solved.
  The tie-preserving PointMaze and AntMaze policy-head screens are positive
  diagnostics over the high-level abstraction. The previous-action-conditioned
  policy head is now positive on PointMaze in a single-seed all-task screen and
  on AntMaze in a three-eval-seed hard-task screen. The scalar value and
  previous-action-free single-label policy heads remain failure boundaries.
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

1. Extend the raw-observation MLP transition screens beyond the current
   cell-abstraction setting, ideally to a differentiable auxiliary loss when
   time permits.
2. If a short neural/controller screen is promising, run a bounded longer run
   with progress logging; otherwise keep it as negative/ablation evidence.
3. Convert the current article-style draft into the exact target venue template
   once the target venue is chosen.

## Completion Status

The package is strong enough for a serious draft and internal review. The full
research objective is not yet complete because the paper would be substantially
stronger with a raw-observation module that goes beyond the current
cell-abstraction screens.
