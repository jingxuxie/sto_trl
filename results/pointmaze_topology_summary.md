# PointMaze Topology-Level Planner Summary

Environment: `pointmaze-teleport-navigate-v0`.

This diagnostic infers a coarse cell topology and stochastic teleport
transitions from the offline dataset, then compares planning updates on the
induced stochastic cell MDP. It is not yet a fully neural controller result; its
purpose is to test whether the stochastic transitive backup can solve the
difficult teleport tasks when the low-level executor is not the bottleneck.

Files:

- Raw result: `results/pointmaze_topology_dataset_5seed_ep50.csv`
- Summary JSON: `results/pointmaze_topology_dataset_5seed_ep50_summary.json`
- Paper table: `results/paper_tables/pointmaze_topology_5seed.csv`
- Paired stats: `results/paper_tables/pointmaze_topology_paired_stats.csv`
- Runner: `scripts/run_pointmaze_topology_planner.py`

Setting:

- Topology states: `45` free maze cells.
- Actions: four compass moves.
- Free cells and stochastic teleport jumps are inferred from offline dataset
  transitions.
- Matched budget: `ceil(log2(45)) = 6` sweeps.
- Full Bellman reference: `180` sweeps.
- Evaluation: five seeds, 50 episodes per task, five fixed OGBench tasks.

Main result:

| method | sweeps | mean success | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bellman matched | 6 | 0.343 | 0.000 | 0.460 | 0.440 | 0.464 | 0.352 |
| Stochastic TRL | 6 | 0.901 | 0.892 | 0.912 | 0.876 | 0.908 | 0.916 |
| Bellman full | 180 | 0.901 | 0.892 | 0.912 | 0.876 | 0.908 | 0.916 |

Paired seed-level improvement:

- Stochastic TRL minus matched Bellman: `+0.558`, 95% CI `[0.501, 0.614]`.
- The improvement is positive for all five seeds.
- Stochastic TRL exactly matches the 180-sweep Bellman reference in this
  topology-level diagnostic.

Interpretation: stochastic TRL can reach high success on the difficult
PointMaze teleport tasks with a logarithmic planning budget when execution is
handled by a reliable dataset-inferred topology-level scaffold. The next
publishability step is to make the local controller/topology module more
standard and portable.

Focused single-task appendix check:

- Raw result: `results/pointmaze_stitch_task5_ep100_seed01234.csv`
- Paper table: `results/paper_tables/pointmaze_stitch_task5_ep100_seed01234.csv`
- Setting: `pointmaze-teleport-stitch-v0`, task 5, five seeds, 100 episodes per
  seed.

| method | sweeps | mean success |
| --- | ---: | ---: |
| Bellman matched | 6 | 0.380 |
| Stochastic TRL | 6 | 0.908 |
| Bellman full | 180 | 0.908 |

This focused row is appendix evidence for high absolute success on one
long-horizon task. It is source-backed by `scripts/verify_main_claims.py`, but
the all-task 5-seed table remains the main PointMaze claim.

Learned local-controller screen:

- Runner: `scripts/run_antmaze_bc_topology_planner.py` reused on
  PointMaze teleport environments.
- Navigate saved policy: `results/policies/pointmaze_navigate_fullgoal_bc_5k.pkl`.
- Navigate raw result:
  `results/pointmaze_bc_topology_navigate_fullgoal_5k_ep20_seed0.csv`.
- Navigate seed-1/2 raw result:
  `results/pointmaze_bc_topology_navigate_fullgoal_5k_ep20_seed12.csv`.
- Stitch saved policy: `results/policies/pointmaze_stitch_fullgoal_bc_5k.pkl`.
- Stitch raw result: `results/pointmaze_bc_topology_stitch_fullgoal_5k_ep20_seed0.csv`.
- Stitch seed-1/2 raw result:
  `results/pointmaze_bc_topology_stitch_fullgoal_5k_ep20_seed12.csv`.
- Setting: full-observation goal-conditioned BC controller, 5k optimizer
  updates, body-nearest `k=16` waypoint observations, seeds 0, 1, and 2,
  20 episodes per task.

| env | Bellman matched | Stochastic TRL | Bellman full | improvement |
| --- | ---: | ---: | ---: | ---: |
| navigate | 0.323 | 1.000 | 1.000 | 0.677 |
| stitch | 0.223 | 1.000 | 1.000 | 0.777 |

The generated table is
`results/paper_tables/pointmaze_learned_controller_ep20_seed012.csv`, and
`scripts/verify_main_claims.py` checks the raw metadata and values.

Interpretation: this is now a verified multi-eval-seed appendix row that
directly addresses the prior PointMaze executor caveat. The stochastic-TRL
planning signal survives on both PointMaze teleport navigate and stitch when
the low-level executor is a learned goal-conditioned BC controller rather than
the proportional topology executor.

Learned transition-module screen:

- Summary: `results/pointmaze_learned_transition_summary.md`.
- Runner: `scripts/run_pointmaze_learned_transition_planner.py`.
- Main finite-sample setting: learned tabular-softmax transition model trained
  from 20 sampled high-level outcomes per `(state, action)` row.
- Result: on both PointMaze teleport navigate and stitch, stochastic TRL reaches
  `0.916` success and matches the 180-sweep Bellman reference, while matched
  Bellman reaches `0.424`.

This is a learned transition/value-module diagnostic, not an end-to-end neural
TRL replacement. The raw continuous-action `dataset_counts` target is currently
negative because the quick discrete action-inference target changes the MDP.
