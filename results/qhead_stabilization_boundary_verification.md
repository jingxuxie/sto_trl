# Q-Head Stabilization Boundary Verification

Status: PASS

- raw self-bootstrap: `results/pointmaze_qhead_monotone_navigate_all_exact.csv` method `qhead_sto_trl` success 0.061, action agreement 0.384
- monotone self-bootstrap: `results/pointmaze_qhead_monotone_navigate_all_exact.csv` method `qhead_sto_trl_monotone` success 0.061, action agreement 0.384
- self-buffered target: `results/pointmaze_qhead_self_buffered_navigate_all_exact.csv` method `qhead_sto_trl_self_buffered` success 0.000, action agreement 0.416
- high-rank raw self-bootstrap: `results/pointmaze_qhead_rank1_navigate_all_exact.csv` method `qhead_sto_trl` success 0.000, action agreement 0.354
- fresh projected iteration: `results/pointmaze_qhead_fresh_iter_navigate_all_exact.csv` method `qhead_sto_trl_fresh_iter` success 0.000, action agreement 0.509
- guided rank projection: `results/pointmaze_qhead_guided_rank_navigate_all_exact.csv` method `qhead_sto_trl_guided_rank` success 0.000, action agreement 0.349
- guided rank projection high weight: `results/pointmaze_qhead_guided_rank10_navigate_all_exact.csv` method `qhead_sto_trl_guided_rank` success 0.119, action agreement 0.431
- sampled target-network projection: `results/pointmaze_qhead_sampled_target_net_nav_exact_s64_c32_k4.csv` method `qhead_sto_trl_sampled_target_net` success 0.061, action agreement 0.355
- sampled target-network projection heavy fit: `results/pointmaze_qhead_sampled_target_net_nav_exact_s64_c32_k4_heavy.csv` method `qhead_sto_trl_sampled_target_net` success 0.000, action agreement 0.335
- sampled target buffer without final reset: `results/pointmaze_qhead_sampled_buffered_nav_exact_s64_c32_k4_no_final.csv` method `qhead_sto_trl_sampled_buffered_reset_final` success 0.000, action agreement 0.603
- sampled target replay reset-each-iteration: `results/pointmaze_qhead_sampled_target_replay_nav_exact_s64_c32_k4.csv` method `qhead_sto_trl_sampled_target_replay` success 0.000, action agreement 0.385
- buffered reset-final q-head navigate: `results/pointmaze_qhead_buffered_reset_final_navigate_all_exact.csv` method `qhead_sto_trl_buffered_reset_final` success 1.000, action agreement 0.966
- buffered reset-final q-head stitch: `results/pointmaze_qhead_buffered_reset_final_stitch_all_exact.csv` method `qhead_sto_trl_buffered_reset_final` success 1.000, action agreement 0.959
- generated target q-head: `results/pointmaze_qhead_guided_rank_navigate_all_exact.csv` method `qhead_sto_trl_target_fit` success 1.000, action agreement 0.966
- table stochastic TRL: `results/pointmaze_qhead_guided_rank10_navigate_all_exact.csv` method `table_sto_trl_matched` success 1.000, action agreement 0.988
