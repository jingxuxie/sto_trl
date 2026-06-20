# Q-Head Critic Claim Verification

Status: PASS
Checked rows: 6

## Exact model all tasks: pointmaze-teleport-navigate-v0

- qhead Bellman TD success: 0.066
- self-bootstrapped qhead success: 0.000
- generated-target qhead critic success: 1.000
- table matched Bellman success: 0.476
- table stochastic TRL success: 1.000
- table full Bellman success: 1.000
- qhead action agreement to full Bellman: 0.966

## Exact model all tasks: pointmaze-teleport-stitch-v0

- qhead Bellman TD success: 0.072
- self-bootstrapped qhead success: 0.000
- generated-target qhead critic success: 1.000
- table matched Bellman success: 0.481
- table stochastic TRL success: 1.000
- table full Bellman success: 1.000
- qhead action agreement to full Bellman: 0.959

## Real env all tasks: pointmaze-teleport-navigate-v0

- qhead Bellman TD success: 0.000
- generated-target qhead critic success: 0.980
- table matched Bellman success: 0.520
- table stochastic TRL success: 0.980
- table full Bellman success: 0.980
- qhead action agreement to full Bellman: 0.966

## Real env all tasks: pointmaze-teleport-stitch-v0

- qhead Bellman TD success: 0.040
- generated-target qhead critic success: 0.980
- table matched Bellman success: 0.520
- table stochastic TRL success: 0.980
- table full Bellman success: 0.980
- qhead action agreement to full Bellman: 0.959

## Real env hard tasks 4,5: pointmaze-teleport-navigate-v0

- qhead Bellman TD success: 0.000
- generated-target qhead critic success: 1.000
- table matched Bellman success: 0.500
- table stochastic TRL success: 1.000
- table full Bellman success: 1.000
- qhead action agreement to full Bellman: 0.966

## Real env hard tasks 4,5: pointmaze-teleport-stitch-v0

- qhead Bellman TD success: 0.125
- generated-target qhead critic success: 1.000
- table matched Bellman success: 0.500
- table stochastic TRL success: 1.000
- table full Bellman success: 1.000
- qhead action agreement to full Bellman: 0.959
