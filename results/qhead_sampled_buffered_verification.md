# Q-Head Sampled-Target Buffer Verification

Status: PASS

## navigate exact sampled targets

- source: `results/pointmaze_qhead_sampled_buffered_nav_exact_s64_c32_k4.csv`
- sampled q-head success: 1.000
- table stochastic success: 1.000
- full Bellman success: 1.000
- action agreement to full: 0.933
- sampled next states per row: 64
- sampled bridge candidates per state-goal pair: 32
- retained bridge waypoints per backup: 4

## stitch exact sampled targets

- source: `results/pointmaze_qhead_sampled_buffered_stitch_exact_s64_c32_k4.csv`
- sampled q-head success: 1.000
- table stochastic success: 1.000
- full Bellman success: 1.000
- action agreement to full: 0.940
- sampled next states per row: 64
- sampled bridge candidates per state-goal pair: 32
- retained bridge waypoints per backup: 4

## navigate real env seed0 sampled targets

- source: `results/pointmaze_qhead_sampled_buffered_nav_env_ep10_seed0_s64_c32_k4.csv`
- sampled q-head success: 0.980
- table stochastic success: 0.980
- full Bellman success: 0.980
- action agreement to full: 0.933
- sampled next states per row: 64
- sampled bridge candidates per state-goal pair: 32
- retained bridge waypoints per backup: 4

## stitch real env seed0 sampled targets

- source: `results/pointmaze_qhead_sampled_buffered_stitch_env_ep10_seed0_s64_c32_k4.csv`
- sampled q-head success: 0.980
- table stochastic success: 0.980
- full Bellman success: 0.980
- action agreement to full: 0.940
- sampled next states per row: 64
- sampled bridge candidates per state-goal pair: 32
- retained bridge waypoints per backup: 4

