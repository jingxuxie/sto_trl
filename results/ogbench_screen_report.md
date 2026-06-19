# OGBench PointMaze Screen

| method | train_step | eval_step | eval_overall_success | best_eval_step | best_eval_overall_success | eval_episode_success | eval_return | eval_length | train_q_loss | train_actor_loss | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gcfbc | 10000 | 10000 | 0.0000 | 5000 | 0.0800 |  |  |  |  | 1.1189 | results/ogbench_screen_exp/dummy/pointmaze_gcfbc_10k_actor_diag_cpu/sd000_20260619_104221 |
| msebc | 10000 | 10000 | 0.0800 | 5000 | 0.1200 |  |  |  |  | 0.4029 | results/ogbench_screen_exp/dummy/pointmaze_msebc_10k_actor_diag_cpu/sd000_20260619_104622 |
| msebc | 10000 | 10000 | 0.1200 | 10000 | 0.1200 |  |  |  |  | 0.3964 | results/ogbench_screen_exp/dummy/pointmaze_msebc_10k_ln256_actor_diag_cpu/sd000_20260619_104714 |
| trl_log_relax_mc | 60000 | 50000 | 0.1000 | 50000 | 0.1000 |  |  |  | 0.0000 | 0.9635 | results/ogbench_screen_exp/dummy/pointmaze_teleport_100k_frs_relax_mc/sd000_20260618_204314 |
| trl_log_relax_neg | 10000 | 10000 | 0.0000 | 10000 | 0.0000 |  |  |  | 0.0000 | 1.0924 | results/ogbench_screen_exp/dummy/pointmaze_teleport_10k_frs_neg03_screen/sd000_20260618_203150 |
| trl_log_relax_neg | 10000 | 10000 | 0.1000 | 10000 | 0.1000 |  |  |  | 0.0000 | 1.0929 | results/ogbench_screen_exp/dummy/pointmaze_teleport_10k_frs_neg_screen/sd000_20260618_202952 |
| trl |  | 10000 | 0.1000 | 10000 | 0.1000 |  |  |  | 0.0852 |  | results/ogbench_screen_exp/dummy/pointmaze_teleport_10k_frs_screen/sd000_20260618_202232 |
| trl_log_relax_mc | 10000 | 10000 | 0.1000 | 10000 | 0.1000 |  |  |  | 0.0002 | 1.0186 | results/ogbench_screen_exp/dummy/pointmaze_teleport_10k_frs_screen/sd000_20260618_202358 |
| trl_log_relax_td | 10000 | 10000 | 0.0400 | 10000 | 0.0400 |  |  |  | 0.0000 | 1.0936 | results/ogbench_screen_exp/dummy/pointmaze_teleport_10k_frs_td/sd000_20260618_210313 |
| trl_log_relax_tdmax | 10000 | 10000 | 0.0000 | 10000 | 0.0000 |  |  |  | 0.0000 | 1.0925 | results/ogbench_screen_exp/dummy/pointmaze_teleport_10k_frs_tdmax/sd000_20260618_211449 |
| trl |  | 10000 | 0.1000 | 10000 | 0.1000 |  |  |  |  |  | results/ogbench_screen_exp/dummy/pointmaze_teleport_10k_screen/sd000_20260618_201344 |
| trl_log | 10000 | 10000 | 0.0000 | 10000 | 0.0000 |  |  |  | 0.2295 | 1.0673 | results/ogbench_screen_exp/dummy/pointmaze_teleport_10k_screen/sd000_20260618_201501 |
| trl_log_mc | 10000 | 10000 | 0.0000 | 10000 | 0.0000 |  |  |  | 0.2436 | 1.0675 | results/ogbench_screen_exp/dummy/pointmaze_teleport_10k_screen/sd000_20260618_201624 |
| trl_log_relax_mc | 10000 | 10000 | 0.0000 | 10000 | 0.0000 |  |  |  | 0.0002 | -0.9145 | results/ogbench_screen_exp/dummy/pointmaze_teleport_10k_screen/sd000_20260618_202022 |
| trl |  | 30000 | 0.0000 | 30000 | 0.0000 |  |  |  | 0.3107 |  | results/ogbench_screen_exp/dummy/pointmaze_teleport_30k_frs_compare/sd000_20260618_203411 |
| trl_log_relax_mc | 30000 | 30000 | 0.1000 | 30000 | 0.1000 |  |  |  | 0.0003 | 0.9285 | results/ogbench_screen_exp/dummy/pointmaze_teleport_30k_frs_compare/sd000_20260618_203651 |
| trl_log_relax_neg | 30000 | 30000 | 0.0000 | 30000 | 0.0000 |  |  |  | 0.0007 | 0.9284 | results/ogbench_screen_exp/dummy/pointmaze_teleport_30k_frs_compare/sd000_20260618_203924 |
| trl |  | 3000 | 0.0000 | 3000 | 0.0000 |  |  |  |  |  | results/ogbench_screen_exp/dummy/pointmaze_teleport_3k_screen/sd000_20260618_200813 |
| trl_log | 3000 | 3000 | 0.0000 | 3000 | 0.0000 |  |  |  | 18289.7380 | 1.0739 | results/ogbench_screen_exp/dummy/pointmaze_teleport_3k_screen/sd000_20260618_200907 |
| trl_log_mc | 3000 | 3000 | 0.0000 | 3000 | 0.0000 |  |  |  | 7315.6600 | 1.0844 | results/ogbench_screen_exp/dummy/pointmaze_teleport_3k_screen/sd000_20260618_201002 |
