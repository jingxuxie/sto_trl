# Evaluation Speed Rejection Verification

Status: PASS

- JAX reference: success 1.000, eval 4.53s, action 0.354 ms/call
- JAX fused: success 0.900, eval 6.08s, action 0.530 ms/call
- JAX vs fused single-step max action difference: 6.85e-07
- cap800 stochastic: success 1.000, eval 4.86s
- cap800 matched Bellman: success 0.400

## Sources

- `results/antmaze_stitch_prev_policy_hard_task45_ep5_seed0_jax_reference.csv`
- `results/antmaze_stitch_prev_policy_hard_task45_ep5_seed0_jax_fused.csv`
- `results/antmaze_stitch_prev_policy_hard_task45_ep5_seed0_cap800.csv`
- `results/antmaze_stitch_prev_policy_hard_task45_ep5_seed0_cap800_matched.csv`
