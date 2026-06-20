# Q-Head Multi-Seed Real-Environment Verification

Status: PASS
Checked environments: 2

## pointmaze-teleport-navigate-v0

- source: `results/pointmaze_qhead_target_fit_navigate_all_env_ep10_seed012.csv`
- qhead Bellman TD: 0.000
- generated-target qhead: 0.933
- table matched Bellman: 0.393
- table stochastic TRL: 0.933
- table full Bellman: 0.933
- qhead action agreement to full: 0.966

## pointmaze-teleport-stitch-v0

- source: `results/pointmaze_qhead_target_fit_stitch_all_env_ep10_seed012.csv`
- qhead Bellman TD: 0.027
- generated-target qhead: 0.933
- table matched Bellman: 0.393
- table stochastic TRL: 0.933
- table full Bellman: 0.933
- qhead action agreement to full: 0.959
