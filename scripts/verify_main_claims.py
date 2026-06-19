#!/usr/bin/env python3
"""Verify headline stochastic TRL paper claims against raw result CSVs."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLE_DIR = RESULTS / "paper_tables"

MAIN_TABLE = TABLE_DIR / "main_hard_task_results.csv"
HARD_STRESS_TABLE = TABLE_DIR / "hard_task_stress_seed0.csv"
POINTMAZE_SINGLE_TASK_TABLE = TABLE_DIR / "pointmaze_stitch_task5_ep100_seed01234.csv"
POINTMAZE_LEARNED_TRANSITION_TABLE = TABLE_DIR / "pointmaze_learned_transition.csv"
POINTMAZE_BC_CONTROLLER_TABLE = TABLE_DIR / "pointmaze_learned_controller_ep20_seed012.csv"
CONTROLLER_EXECUTION_ISOLATION_TABLE = TABLE_DIR / "controller_execution_isolation.csv"
FAST_EVAL_PROFILE_TABLE = TABLE_DIR / "fast_eval_profile.csv"
ANTMAZE_SUPPORT_ABLATION_TABLE = TABLE_DIR / "antmaze_support_ablation_ep5_seed0_task45.csv"
POINTMAZE_TIE_POLICY_HEAD_TABLE = TABLE_DIR / "pointmaze_tie_policy_head_ep20_seed0.csv"
POINTMAZE_PREV_POLICY_HEAD_TABLE = TABLE_DIR / "pointmaze_rawobs_transition_prev_policy_head_ep20_seed0.csv"
POINTMAZE_TIE_POLICY_HEAD_EVAL_SEED_TABLE = TABLE_DIR / "pointmaze_tie_policy_head_ep20_evalseed012.csv"
POINTMAZE_TIE_POLICY_HEAD_TRANSITION_SEED_TABLE = TABLE_DIR / "pointmaze_tie_policy_head_ep20_tseed012.csv"
ANTMAZE_TIE_POLICY_HEAD_TABLE = TABLE_DIR / "antmaze_tie_policy_head_hard_tasks_ep20_seed0.csv"
ANTMAZE_RAWOBS_TIE_POLICY_HEAD_TABLE = TABLE_DIR / "antmaze_rawobs_transition_tie_policy_head_ep10_tseed012.csv"
ANTMAZE_RAWOBS_TIE_POLICY_EVAL_SEED_TABLE = TABLE_DIR / "antmaze_rawobs_transition_tie_policy_head_ep10_evalseed012.csv"
ANTMAZE_RAWOBS_PREV_POLICY_EVAL_SEED_TABLE = (
    TABLE_DIR / "antmaze_rawobs_transition_prev_policy_head_ep10_evalseed012.csv"
)
ANTMAZE_LEARNED_TRANSITION_TABLE = TABLE_DIR / "antmaze_learned_transition_robustness.csv"
REPORT = RESULTS / "main_claim_verification.md"
SUPPORT_ABLATION = RESULTS / "pointmaze_topology_stitch_support_baseline_5seed_ep50.csv"
POINTMAZE_SINGLE_TASK_SOURCE = RESULTS / "pointmaze_stitch_task5_ep100_seed01234.csv"
POINTMAZE_TIE_POLICY_HEAD_SPECS = [
    (
        "pointmaze-teleport-navigate-v0",
        RESULTS / "pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv",
    ),
    (
        "pointmaze-teleport-stitch-v0",
        RESULTS / "pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv",
    ),
]
POINTMAZE_PREV_POLICY_HEAD_SPECS = [
    (
        "pointmaze-teleport-navigate-v0",
        RESULTS / "pointmaze_navigate_rawobs_mlp_transition_prev_policy_ep20_seed0.csv",
    ),
    (
        "pointmaze-teleport-stitch-v0",
        RESULTS / "pointmaze_stitch_rawobs_mlp_transition_prev_policy_ep20_seed0.csv",
    ),
]
POINTMAZE_TIE_POLICY_HEAD_EVAL_SEED_SPECS = [
    {
        "env": "pointmaze-teleport-navigate-v0",
        "sources": [
            RESULTS / "pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv",
            RESULTS / "pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_evalseed12.csv",
        ],
    },
    {
        "env": "pointmaze-teleport-stitch-v0",
        "sources": [
            RESULTS / "pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv",
            RESULTS / "pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_evalseed12.csv",
        ],
    },
]
POINTMAZE_TIE_POLICY_HEAD_TRANSITION_SEED_SPECS = [
    {
        "env": "pointmaze-teleport-navigate-v0",
        "sources": [
            RESULTS / "pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv",
            RESULTS / "pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_tseed1.csv",
            RESULTS / "pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_tseed2.csv",
        ],
    },
    {
        "env": "pointmaze-teleport-stitch-v0",
        "sources": [
            RESULTS / "pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv",
            RESULTS / "pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_tseed1.csv",
            RESULTS / "pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_tseed2.csv",
        ],
    },
]
ANTMAZE_TIE_POLICY_HEAD_SPECS = [
    {
        "env": "antmaze-teleport-navigate-v0",
        "controller": "50k full-goal BC + body-nearest k16",
        "bc_steps": "50000",
        "source": RESULTS / "antmaze_navigate_tie_policy_head_hard_tasks_ep20_seed0.csv",
        "min_sto": 0.90,
        "max_matched": 0.40,
        "min_improvement": 0.55,
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "controller": "20k full-goal BC + body-nearest k16",
        "bc_steps": "20000",
        "source": RESULTS / "antmaze_stitch_tie_policy_head_hard_tasks_ep20_seed0.csv",
        "min_sto": 0.95,
        "max_matched": 0.45,
        "min_improvement": 0.50,
    },
]
ANTMAZE_RAWOBS_TIE_POLICY_HEAD_SPECS = [
    {
        "env": "antmaze-teleport-navigate-v0",
        "controller": "50k full-goal BC + body-nearest k16",
        "bc_steps": "50000",
        "sources": [
            RESULTS / "antmaze_navigate_rawobs_transition_tie_policy_head_ep10_seed0.csv",
            RESULTS / "antmaze_navigate_rawobs_transition_tie_policy_head_ep10_tseed1.csv",
            RESULTS / "antmaze_navigate_rawobs_transition_tie_policy_head_ep10_tseed2.csv",
        ],
        "expected_transition_seeds": {"0", "1", "2"},
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "controller": "20k full-goal BC + body-nearest k16",
        "bc_steps": "20000",
        "sources": [
            RESULTS / "antmaze_stitch_rawobs_transition_tie_policy_head_ep10_seed0.csv",
            RESULTS / "antmaze_stitch_rawobs_transition_tie_policy_head_ep10_tseed1.csv",
            RESULTS / "antmaze_stitch_rawobs_transition_tie_policy_head_ep10_tseed2.csv",
        ],
        "expected_transition_seeds": {"0", "1", "2"},
    },
]
ANTMAZE_RAWOBS_TIE_POLICY_EVAL_SEED_SPECS = [
    {
        "env": "antmaze-teleport-navigate-v0",
        "controller": "50k full-goal BC + body-nearest k16",
        "bc_steps": "50000",
        "sources": [
            RESULTS / "antmaze_navigate_rawobs_transition_tie_policy_head_ep10_seed0.csv",
            RESULTS / "antmaze_navigate_rawobs_transition_tie_policy_head_ep10_tseed0_evalseed12.csv",
        ],
        "min_sto": 0.93,
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "controller": "20k full-goal BC + body-nearest k16",
        "bc_steps": "20000",
        "sources": [
            RESULTS / "antmaze_stitch_rawobs_transition_tie_policy_head_ep10_seed0.csv",
            RESULTS / "antmaze_stitch_rawobs_transition_tie_policy_head_ep10_tseed0_evalseed12.csv",
        ],
        "min_sto": 0.95,
    },
]
ANTMAZE_RAWOBS_PREV_POLICY_EVAL_SEED_SPECS = [
    {
        "env": "antmaze-teleport-navigate-v0",
        "controller": "50k full-goal BC + body-nearest k16",
        "bc_steps": "50000",
        "episodes": "10",
        "sources": [
            RESULTS / "antmaze_navigate_rawobs_transition_prev_policy_head_ep10_seed012.csv",
        ],
        "min_sto": 0.93,
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "controller": "20k full-goal BC + body-nearest k16",
        "bc_steps": "20000",
        "episodes": "10",
        "sources": [
            RESULTS / "antmaze_stitch_rawobs_transition_prev_policy_head_ep10_seed012.csv",
        ],
        "min_sto": 0.95,
    },
]
POINTMAZE_BC_CONTROLLER_SPECS = [
    {
        "env": "pointmaze-teleport-navigate-v0",
        "controller": "5k full-goal BC + body-nearest k16",
        "paths": [
            RESULTS / "pointmaze_bc_topology_navigate_fullgoal_5k_ep20_seed0.csv",
            RESULTS / "pointmaze_bc_topology_navigate_fullgoal_5k_ep20_seed12.csv",
        ],
        "max_matched": 0.35,
        "min_improvement": 0.65,
    },
    {
        "env": "pointmaze-teleport-stitch-v0",
        "controller": "5k full-goal BC + body-nearest k16",
        "paths": [
            RESULTS / "pointmaze_bc_topology_stitch_fullgoal_5k_ep20_seed0.csv",
            RESULTS / "pointmaze_bc_topology_stitch_fullgoal_5k_ep20_seed12.csv",
        ],
        "max_matched": 0.35,
        "min_improvement": 0.65,
    },
]
POINTMAZE_DIRECT_ACTOR_SPECS = [
    {
        "execution_path": "GCFBC direct final-goal actor",
        "env": "pointmaze-teleport-navigate-v0",
        "controller_or_actor": "GCFBC 128,128",
        "eval_setting": "seed 0, 5 episodes/task",
        "path": RESULTS
        / "ogbench_screen_exp"
        / "dummy"
        / "pointmaze_gcfbc_10k_actor_diag_cpu"
        / "sd000_20260619_104221"
        / "eval.csv",
    },
    {
        "execution_path": "MSEBC direct final-goal actor",
        "env": "pointmaze-teleport-navigate-v0",
        "controller_or_actor": "MSEBC LN 256,256",
        "eval_setting": "seed 0, 5 episodes/task",
        "path": RESULTS
        / "ogbench_screen_exp"
        / "dummy"
        / "pointmaze_msebc_10k_ln256_actor_diag_cpu"
        / "sd000_20260619_104714"
        / "eval.csv",
    },
]
FAST_EVAL_PROFILE_SPECS = [
    {
        "screen": "antmaze navigate hard slice",
        "role": "recommended fast screen",
        "path": RESULTS / "antmaze_navigate_fast_profile_repeat1_ep5_seed0_task45.csv",
        "min_sto": 0.90,
        "min_improvement": 0.40,
    },
    {
        "screen": "antmaze stitch hard slice",
        "role": "recommended fast screen",
        "path": RESULTS / "antmaze_stitch_fast_profile_repeat1_ep5_seed0_task45.csv",
        "min_sto": 0.99,
        "min_improvement": 0.35,
    },
    {
        "screen": "antmaze stitch hard slice",
        "role": "two-episode baseline",
        "path": RESULTS / "antmaze_stitch_fast_profile_repeat1_ep2_seed0_task45.csv",
    },
    {
        "screen": "antmaze stitch hard slice",
        "role": "action-repeat ablation",
        "path": RESULTS / "antmaze_stitch_fast_profile_repeat2_ep2_seed0_task45.csv",
    },
]
ANTMAZE_SUPPORT_ABLATION_SPECS = [
    {
        "env": "antmaze-teleport-navigate-v0",
        "controller": "50k full-goal BC + body-nearest k16",
        "path": RESULTS / "antmaze_navigate_support_ablation_ep5_seed0_task45.csv",
        "min_sto": 0.90,
        "min_sto_minus_support": 0.50,
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "controller": "20k full-goal BC + body-nearest k16",
        "path": RESULTS / "antmaze_stitch_support_ablation_ep5_seed0_task45.csv",
        "min_sto": 0.90,
        "min_sto_minus_support": 0.30,
    },
]
POINTMAZE_LEARNED_TRANSITION_SPECS = [
    {
        "env": "pointmaze-teleport-navigate-v0",
        "transition_model": "dataset cell-change softmax",
        "raw_transition_model": "table_softmax",
        "sources": [RESULTS / "pointmaze_learned_transition_navigate_cellchanges_1k_ep50_seed0.csv"],
        "transition_steps": "1000",
        "expected_transition_seeds": {"0"},
        "max_matched": 0.45,
        "min_improvement": 0.45,
    },
    {
        "env": "pointmaze-teleport-stitch-v0",
        "transition_model": "dataset cell-change softmax",
        "raw_transition_model": "table_softmax",
        "sources": [RESULTS / "pointmaze_learned_transition_stitch_cellchanges_1k_ep50_seed0.csv"],
        "transition_steps": "1000",
        "expected_transition_seeds": {"0"},
        "max_matched": 0.45,
        "min_improvement": 0.45,
    },
    {
        "env": "pointmaze-teleport-navigate-v0",
        "transition_model": "raw-observation MLP cell-change",
        "raw_transition_model": "raw_obs_mlp",
        "sources": [
            RESULTS / "pointmaze_navigate_rawobs_mlp_cellchanges_ep50_seed0.csv",
            RESULTS / "pointmaze_navigate_rawobs_mlp_cellchanges_ep50_tseed1.csv",
            RESULTS / "pointmaze_navigate_rawobs_mlp_cellchanges_ep50_tseed2.csv",
        ],
        "transition_steps": "2000",
        "expected_transition_seeds": {"0", "1", "2"},
        "max_matched": 0.55,
        "min_improvement": 0.40,
    },
    {
        "env": "pointmaze-teleport-stitch-v0",
        "transition_model": "raw-observation MLP cell-change",
        "raw_transition_model": "raw_obs_mlp",
        "sources": [
            RESULTS / "pointmaze_stitch_rawobs_mlp_cellchanges_ep50_seed0.csv",
            RESULTS / "pointmaze_stitch_rawobs_mlp_cellchanges_ep50_tseed1.csv",
            RESULTS / "pointmaze_stitch_rawobs_mlp_cellchanges_ep50_tseed2.csv",
        ],
        "transition_steps": "2000",
        "expected_transition_seeds": {"0", "1", "2"},
        "max_matched": 0.55,
        "min_improvement": 0.40,
    },
]
ANTMAZE_LEARNED_TRANSITION_SPECS = [
    {
        "env": "antmaze-teleport-navigate-v0",
        "controller": "50k BC",
        "bc_steps": "50000",
        "transition_model": "table softmax, 20 samples/row",
        "raw_transition_model": "learned_softmax",
        "raw_transition_source": "topology_samples",
        "samples_per_row": "20",
        "transition_steps": "1000",
        "expected_transition_seeds": {"0", "1", "2"},
        "min_sto": 0.95,
        "max_matched": 0.40,
        "min_improvement": 0.55,
        "paths": [
            RESULTS / "antmaze_navigate_learned_transition_samples20_ep10_tseed0.csv",
            RESULTS / "antmaze_navigate_learned_transition_samples20_ep10_tseed1.csv",
            RESULTS / "antmaze_navigate_learned_transition_samples20_ep10_tseed2.csv",
        ],
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "controller": "20k BC",
        "bc_steps": "20000",
        "transition_model": "table softmax, 20 samples/row",
        "raw_transition_model": "learned_softmax",
        "raw_transition_source": "topology_samples",
        "samples_per_row": "20",
        "transition_steps": "1000",
        "expected_transition_seeds": {"0", "1", "2"},
        "min_sto": 0.95,
        "max_matched": 0.40,
        "min_improvement": 0.55,
        "paths": [
            RESULTS / "antmaze_stitch_learned_transition_samples20_ep10_tseed0.csv",
            RESULTS / "antmaze_stitch_learned_transition_samples20_ep10_tseed1.csv",
            RESULTS / "antmaze_stitch_learned_transition_samples20_ep10_tseed2.csv",
        ],
    },
    {
        "env": "antmaze-teleport-navigate-v0",
        "controller": "50k BC",
        "bc_steps": "50000",
        "transition_model": "raw-observation MLP jump-change",
        "raw_transition_model": "raw_obs_mlp",
        "raw_transition_source": "dataset_jump_changes",
        "transition_steps": "1000",
        "expected_transition_seeds": {"0", "1", "2"},
        "min_sto": 0.95,
        "max_matched": 0.40,
        "min_improvement": 0.55,
        "paths": [
            RESULTS / "antmaze_navigate_rawobs_mlp_jumpchanges_ep10_seed0.csv",
            RESULTS / "antmaze_navigate_rawobs_mlp_jumpchanges_ep10_tseed1.csv",
            RESULTS / "antmaze_navigate_rawobs_mlp_jumpchanges_ep10_tseed2.csv",
        ],
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "controller": "20k BC",
        "bc_steps": "20000",
        "transition_model": "raw-observation MLP jump-change",
        "raw_transition_model": "raw_obs_mlp",
        "raw_transition_source": "dataset_jump_changes",
        "transition_steps": "1000",
        "expected_transition_seeds": {"0", "1", "2"},
        "min_sto": 0.95,
        "max_matched": 0.40,
        "min_improvement": 0.55,
        "paths": [
            RESULTS / "antmaze_stitch_rawobs_mlp_jumpchanges_ep10_seed0.csv",
            RESULTS / "antmaze_stitch_rawobs_mlp_jumpchanges_ep10_tseed1.csv",
            RESULTS / "antmaze_stitch_rawobs_mlp_jumpchanges_ep10_tseed2.csv",
        ],
    },
]
ANTMAZE_BCSEED1_SPECS = [
    (
        "antmaze-teleport-navigate-v0",
        RESULTS / "antmaze_bc_topology_navigate_fullgoal_50k_bcseed1_ep20_seed012_bodyk16_cpu.csv",
        "50000",
    ),
    (
        "antmaze-teleport-stitch-v0",
        RESULTS / "antmaze_bc_topology_stitch_fullgoal_20k_bcseed1_ep20_seed012_bodyk16_cpu.csv",
        "20000",
    ),
]
ANTMAZE_STITCH_CONTROLLER_SEED_SPECS = [
    (
        "0",
        RESULTS / "antmaze_bc_topology_stitch_fullgoal_20k_ep20_seed012_bodyk16_cpu.csv",
    ),
    (
        "1",
        RESULTS / "antmaze_bc_topology_stitch_fullgoal_20k_bcseed1_ep20_seed012_bodyk16_cpu.csv",
    ),
    (
        "2",
        RESULTS / "antmaze_bc_topology_stitch_fullgoal_20k_bcseed2_ep20_seed012_bodyk16_cpu.csv",
    ),
]
ANTMAZE_NAVIGATE_CONTROLLER_SEED_SPECS = [
    (
        "0",
        RESULTS / "antmaze_bc_topology_navigate_fullgoal_50k_ep20_seed012_bodyk16_cpu.csv",
    ),
    (
        "1",
        RESULTS / "antmaze_bc_topology_navigate_fullgoal_50k_bcseed1_ep20_seed012_bodyk16_cpu.csv",
    ),
    (
        "2",
        RESULTS / "antmaze_bc_topology_navigate_fullgoal_50k_bcseed2_ep20_seed012_bodyk16_cpu.csv",
    ),
]

MAIN_SPECS = [
    {
        "env": "pointmaze-teleport-navigate-v0",
        "source": RESULTS / "pointmaze_topology_dataset_5seed_ep50.csv",
        "executor": "dataset topology scaffold",
        "eval_setting": "5 eval seeds, 50 episodes/task",
        "controller_steps": "0",
    },
    {
        "env": "pointmaze-teleport-stitch-v0",
        "source": RESULTS / "pointmaze_topology_stitch_5seed_ep50.csv",
        "executor": "dataset topology scaffold",
        "eval_setting": "5 eval seeds, 50 episodes/task",
        "controller_steps": "0",
    },
    {
        "env": "antmaze-teleport-navigate-v0",
        "source": RESULTS / "antmaze_bc_topology_navigate_fullgoal_50k_ep20_seed012_bodyk16_cpu.csv",
        "executor": "full-goal BC + body-nearest k16",
        "eval_setting": "3 eval seeds, 20 episodes/task",
        "controller_steps": "50000",
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "source": RESULTS / "antmaze_bc_topology_stitch_fullgoal_20k_ep20_seed012_bodyk16_cpu.csv",
        "executor": "full-goal BC + body-nearest k16",
        "eval_setting": "3 eval seeds, 20 episodes/task",
        "controller_steps": "20000",
    },
]
HARD_STRESS_SPECS = [
    {
        "env": "pointmaze-teleport-stitch-v0",
        "task_scope": "tasks 4,5",
        "eval_setting": "seed 0, 50 episodes/task",
        "controller_steps": "0",
        "source": RESULTS / "pointmaze_stitch_hard_task45_ep50_seed0_fastfocus.csv",
        "allow_support": True,
    },
    {
        "env": "antmaze-teleport-navigate-v0",
        "task_scope": "tasks 4,5",
        "eval_setting": "seed 0, 10 episodes/task",
        "controller_steps": "50000",
        "source": RESULTS / "antmaze_navigate_hard_task45_ep10_seed0_fastfocus.csv",
        "allow_support": False,
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "task_scope": "tasks 4,5",
        "eval_setting": "seed 0, 10 episodes/task",
        "controller_steps": "20000",
        "source": RESULTS / "antmaze_stitch_hard_task45_ep10_seed0_fastfocus.csv",
        "allow_support": False,
    },
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals)


def sample_stdev(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) <= 1:
        return 0.0
    mu = mean(vals)
    return math.sqrt(sum((x - mu) ** 2 for x in vals) / (len(vals) - 1))


def t_critical_95(n: int) -> float:
    return {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def assert_equal(errors: list[str], got: object, expected: object, label: str) -> None:
    if got != expected:
        errors.append(f"{label}: got {got!r}, expected {expected!r}")


def assert_close(errors: list[str], got: float, expected: float, label: str, tol: float = 5e-4) -> None:
    if abs(got - expected) > tol:
        errors.append(f"{label}: got {got:.6f}, expected {expected:.6f}")


def summarize_source(path: Path) -> dict[str, dict[str, float]]:
    return summarize_rows(read_rows(path))


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    summary: dict[str, dict[str, float]] = {}
    for method, rows in grouped.items():
        summary[method] = {
            "success": mean(float(row["overall_success"]) for row in rows),
            "sweeps": float(rows[0]["iters"]),
        }
    return summary


def distinct_values(rows: list[dict[str, str]], key: str) -> set[str]:
    return {row.get(key, "") for row in rows if key in row}


def verify_main_table(errors: list[str]) -> list[dict[str, str]]:
    main_rows = read_rows(MAIN_TABLE)
    by_env = {row["env"]: row for row in main_rows}
    verified: list[dict[str, str]] = []
    for spec in MAIN_SPECS:
        env = spec["env"]
        source = spec["source"]
        if not source.exists():
            errors.append(f"{env}: missing source {source.relative_to(ROOT)}")
            continue
        if env not in by_env:
            errors.append(f"{env}: missing row in {MAIN_TABLE.relative_to(ROOT)}")
            continue
        row = by_env[env]
        source_rows = read_rows(source)
        summary = summarize_rows(source_rows)
        required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
        if not required.issubset(summary):
            errors.append(f"{env}: source missing methods {sorted(required - set(summary))}")
            continue
        if spec["controller_steps"] != "0":
            raw_bc_steps = distinct_values(source_rows, "bc_steps")
            if raw_bc_steps != {spec["controller_steps"]}:
                errors.append(
                    f"{env}: raw bc_steps metadata {sorted(raw_bc_steps)} does not match "
                    f"controller_steps {spec['controller_steps']}"
                )

        matched = summary["bellman_matched"]["success"]
        sto = summary["sto_trl_matched"]["success"]
        full = summary["bellman_full"]["success"]
        matched_sweeps = int(summary["bellman_matched"]["sweeps"])
        sto_sweeps = int(summary["sto_trl_matched"]["sweeps"])
        full_sweeps = int(summary["bellman_full"]["sweeps"])

        assert_equal(errors, row["executor"], spec["executor"], f"{env} executor")
        assert_equal(errors, row["eval_setting"], spec["eval_setting"], f"{env} eval_setting")
        assert_equal(errors, row["controller_steps"], spec["controller_steps"], f"{env} controller_steps")
        assert_equal(errors, int(row["matched_sweeps"]), matched_sweeps, f"{env} matched_sweeps")
        assert_equal(errors, sto_sweeps, matched_sweeps, f"{env} stochastic TRL sweeps")
        assert_equal(errors, int(row["full_sweeps"]), full_sweeps, f"{env} full_sweeps")
        assert_close(errors, float(row["bellman_matched"]), round(matched, 3), f"{env} matched table value")
        assert_close(errors, float(row["stochastic_trl"]), round(sto, 3), f"{env} stochastic TRL table value")
        assert_close(errors, float(row["bellman_full"]), round(full, 3), f"{env} full Bellman table value")
        assert_close(errors, float(row["sto_minus_matched"]), round(sto - matched, 3), f"{env} improvement")

        if matched_sweeps != 6 or sto_sweeps != 6:
            errors.append(f"{env}: matched-budget sweeps are not 6")
        if full_sweeps != 180:
            errors.append(f"{env}: full Bellman reference sweeps are not 180")
        if sto < 0.90:
            errors.append(f"{env}: stochastic TRL success below 0.90 ({sto:.3f})")
        if matched > 0.40:
            errors.append(f"{env}: matched Bellman success above 0.40 ({matched:.3f})")
        if sto - matched < 0.50:
            errors.append(f"{env}: stochastic TRL improvement below 0.50 ({sto - matched:.3f})")
        if abs(sto - full) > 1e-12 or row["sto_equals_full"] != "True":
            errors.append(f"{env}: stochastic TRL does not exactly match full Bellman")

        verified.append(
            {
                "env": env,
                "source": str(source.relative_to(ROOT)),
                "matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "full": fmt(full),
                "improvement": fmt(sto - matched),
            }
        )
    return verified


def verify_hard_task_stress_table(errors: list[str]) -> list[dict[str, str]]:
    if not HARD_STRESS_TABLE.exists():
        errors.append(f"missing hard-task stress table {HARD_STRESS_TABLE.relative_to(ROOT)}")
        return []
    table_rows = read_rows(HARD_STRESS_TABLE)
    by_env = {row["env"]: row for row in table_rows}
    verified: list[dict[str, str]] = []
    for spec in HARD_STRESS_SPECS:
        env = spec["env"]
        source = spec["source"]
        if not source.exists():
            errors.append(f"{env}: missing hard-task source {source.relative_to(ROOT)}")
            continue
        if env not in by_env:
            errors.append(f"{env}: missing row in {HARD_STRESS_TABLE.relative_to(ROOT)}")
            continue
        row = by_env[env]
        summary = summarize_source(source)
        required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
        if not required.issubset(summary):
            errors.append(f"{env}: hard-task source missing methods {sorted(required - set(summary))}")
            continue
        matched = summary["bellman_matched"]["success"]
        sto = summary["sto_trl_matched"]["success"]
        full = summary["bellman_full"]["success"]
        support = summary.get("support_trl_matched", {}).get("success")

        assert_equal(errors, row["task_scope"], spec["task_scope"], f"{env} hard-task task_scope")
        assert_equal(errors, row["eval_setting"], spec["eval_setting"], f"{env} hard-task eval_setting")
        assert_equal(errors, row["controller_steps"], spec["controller_steps"], f"{env} hard-task controller_steps")
        assert_equal(errors, row["source"], str(source.relative_to(ROOT)), f"{env} hard-task source")
        assert_close(errors, float(row["bellman_matched"]), round(matched, 3), f"{env} hard-task matched")
        assert_close(errors, float(row["stochastic_trl"]), round(sto, 3), f"{env} hard-task stochastic TRL")
        assert_close(errors, float(row["bellman_full"]), round(full, 3), f"{env} hard-task full Bellman")
        assert_close(errors, float(row["sto_minus_matched"]), round(sto - matched, 3), f"{env} hard-task improvement")

        if spec["allow_support"]:
            if support is None:
                errors.append(f"{env}: hard-task source missing support TRL row")
            elif row["support_trl"] == "":
                errors.append(f"{env}: hard-task table missing support TRL value")
            else:
                assert_close(errors, float(row["support_trl"]), round(support, 3), f"{env} hard-task support")
                if not (matched < support < sto):
                    errors.append(
                        f"{env}: expected matched < support < stochastic in hard-task stress, "
                        f"got {matched:.3f}, {support:.3f}, {sto:.3f}"
                    )
        elif row["support_trl"] != "":
            errors.append(f"{env}: hard-task support TRL should be blank")

        if sto < 0.90:
            errors.append(f"{env}: hard-task stochastic TRL success below 0.90 ({sto:.3f})")
        if sto - matched < 0.40:
            errors.append(f"{env}: hard-task improvement below 0.40 ({sto - matched:.3f})")
        if abs(sto - full) > 1e-12:
            errors.append(f"{env}: hard-task stochastic TRL does not match full Bellman")

        verified.append(
            {
                "env": env,
                "task_scope": str(spec["task_scope"]),
                "matched": fmt(matched),
                "support_trl": "" if support is None else fmt(float(support)),
                "stochastic_trl": fmt(sto),
                "full": fmt(full),
                "improvement": fmt(sto - matched),
            }
        )
    return verified


def verify_pointmaze_single_task(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    if not POINTMAZE_SINGLE_TASK_SOURCE.exists():
        errors.append(f"PointMaze task-5: missing source {POINTMAZE_SINGLE_TASK_SOURCE.relative_to(ROOT)}")
        return verified
    if not POINTMAZE_SINGLE_TASK_TABLE.exists():
        errors.append(f"PointMaze task-5: missing table {POINTMAZE_SINGLE_TASK_TABLE.relative_to(ROOT)}")
        return verified

    rows = read_rows(POINTMAZE_SINGLE_TASK_SOURCE)
    table_rows = read_rows(POINTMAZE_SINGLE_TASK_TABLE)
    if len(table_rows) != 1:
        errors.append(f"PointMaze task-5: expected one generated table row, got {len(table_rows)}")
        return verified
    table = table_rows[0]

    raw_seeds = distinct_values(rows, "seed")
    raw_episodes = distinct_values(rows, "episodes_per_task")
    raw_tasks = distinct_values(rows, "task_ids")
    raw_topology = distinct_values(rows, "topology_source")
    if raw_seeds != {"0", "1", "2", "3", "4"}:
        errors.append(f"PointMaze task-5: raw seeds {sorted(raw_seeds)} do not match 0..4")
    if raw_episodes != {"100"}:
        errors.append(f"PointMaze task-5: raw episodes_per_task {sorted(raw_episodes)} do not match 100")
    if raw_tasks != {"5"}:
        errors.append(f"PointMaze task-5: raw task_ids {sorted(raw_tasks)} do not match 5")
    if raw_topology and raw_topology != {"dataset"}:
        errors.append(f"PointMaze task-5: raw topology_source {sorted(raw_topology)} does not match dataset")

    summary = summarize_rows(rows)
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    if not required.issubset(summary):
        errors.append(f"PointMaze task-5: source missing methods {sorted(required - set(summary))}")
        return verified

    matched = summary["bellman_matched"]["success"]
    sto = summary["sto_trl_matched"]["success"]
    full = summary["bellman_full"]["success"]
    matched_sweeps = int(summary["bellman_matched"]["sweeps"])
    sto_sweeps = int(summary["sto_trl_matched"]["sweeps"])
    full_sweeps = int(summary["bellman_full"]["sweeps"])

    assert_equal(errors, table["env"], "pointmaze-teleport-stitch-v0", "PointMaze task-5 env")
    assert_equal(errors, table["task_id"], "5", "PointMaze task-5 task_id")
    assert_equal(errors, table["eval_setting"], "5 eval seeds, 100 episodes", "PointMaze task-5 eval_setting")
    assert_equal(errors, int(table["matched_sweeps"]), matched_sweeps, "PointMaze task-5 matched_sweeps")
    assert_equal(errors, sto_sweeps, matched_sweeps, "PointMaze task-5 stochastic sweeps")
    assert_equal(errors, int(table["full_sweeps"]), full_sweeps, "PointMaze task-5 full_sweeps")
    assert_close(errors, float(table["bellman_matched"]), round(matched, 3), "PointMaze task-5 matched")
    assert_close(errors, float(table["stochastic_trl"]), round(sto, 3), "PointMaze task-5 stochastic")
    assert_close(errors, float(table["bellman_full"]), round(full, 3), "PointMaze task-5 full")

    if matched_sweeps != 6 or sto_sweeps != 6:
        errors.append("PointMaze task-5: matched and stochastic TRL sweeps are not 6")
    if full_sweeps != 180:
        errors.append(f"PointMaze task-5: full Bellman sweeps are not 180 ({full_sweeps})")
    if sto < 0.90:
        errors.append(f"PointMaze task-5: stochastic TRL success below 0.90 ({sto:.3f})")
    if matched > 0.40:
        errors.append(f"PointMaze task-5: matched Bellman success above 0.40 ({matched:.3f})")
    if sto - matched < 0.50:
        errors.append(f"PointMaze task-5: improvement below 0.50 ({sto - matched:.3f})")
    if abs(sto - full) > 1e-12:
        errors.append("PointMaze task-5: stochastic TRL does not match full Bellman")

    verified.append(
        {
            "env": table["env"],
            "task_id": table["task_id"],
            "n_seeds": str(len(raw_seeds)),
            "episodes_per_seed": "100",
            "matched": fmt(matched),
            "stochastic_trl": fmt(sto),
            "full": fmt(full),
            "improvement": fmt(sto - matched),
        }
    )
    return verified


def verify_pointmaze_learned_transition(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    if not POINTMAZE_LEARNED_TRANSITION_TABLE.exists():
        errors.append(
            f"PointMaze learned transition: missing table "
            f"{POINTMAZE_LEARNED_TRANSITION_TABLE.relative_to(ROOT)}"
        )
        return verified

    table_rows = read_rows(POINTMAZE_LEARNED_TRANSITION_TABLE)
    by_key = {(row["env"], row["transition_model"]): row for row in table_rows}
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for spec in POINTMAZE_LEARNED_TRANSITION_SPECS:
        env = str(spec["env"])
        transition_model = str(spec["transition_model"])
        sources = list(spec["sources"])
        key = (env, transition_model)
        if key not in by_key:
            errors.append(f"{env}: missing learned-transition table row for {transition_model}")
            continue
        row = by_key[key]
        summaries: list[dict[str, dict[str, float]]] = []
        transition_seeds: list[str] = []
        oracle_l1: list[float] = []
        oracle_top1: list[float] = []
        source_labels = [str(source.relative_to(ROOT)) for source in sources]
        for source in sources:
            if not source.exists():
                errors.append(f"{env}: missing learned-transition source {source.relative_to(ROOT)}")
                continue
            source_rows = read_rows(source)
            raw_episodes = distinct_values(source_rows, "episodes_per_task")
            raw_seeds = distinct_values(source_rows, "seed")
            raw_tasks = distinct_values(source_rows, "task_ids")
            raw_transition_model = distinct_values(source_rows, "transition_model")
            raw_transition_source = distinct_values(source_rows, "transition_target_source")
            raw_transition_steps = distinct_values(source_rows, "transition_steps")
            if raw_episodes != {"50"}:
                errors.append(f"{env}: raw learned-transition episodes {sorted(raw_episodes)} do not match 50")
            if raw_seeds != {"0"}:
                errors.append(f"{env}: raw learned-transition eval seeds {sorted(raw_seeds)} do not match seed 0")
            if raw_tasks != {"all"}:
                errors.append(f"{env}: raw learned-transition task_ids {sorted(raw_tasks)} do not match all")
            if raw_transition_model != {str(spec["raw_transition_model"])}:
                errors.append(
                    f"{env}: raw learned-transition model {sorted(raw_transition_model)} "
                    f"does not match {spec['raw_transition_model']}"
                )
            if raw_transition_source != {"dataset_cell_changes"}:
                errors.append(
                    f"{env}: raw learned-transition target source {sorted(raw_transition_source)} "
                    "does not match dataset_cell_changes"
                )
            if raw_transition_steps != {str(spec["transition_steps"])}:
                errors.append(
                    f"{env}: raw learned-transition steps {sorted(raw_transition_steps)} "
                    f"do not match {spec['transition_steps']}"
                )

            summary = summarize_rows(source_rows)
            if not required.issubset(summary):
                errors.append(f"{env}: learned-transition source missing methods {sorted(required - set(summary))}")
                continue
            by_method = {source_row["method"]: source_row for source_row in source_rows}
            summaries.append(summary)
            transition_seeds.append(str(int(float(by_method["sto_trl_matched"]["transition_seed"]))))
            oracle_l1.append(float(by_method["sto_trl_matched"]["transition_oracle_l1"]))
            oracle_top1.append(float(by_method["sto_trl_matched"]["transition_oracle_top1"]))

        if len(summaries) != len(sources):
            continue

        def method_mean(method: str) -> float:
            return mean(summary[method]["success"] for summary in summaries)

        matched = method_mean("bellman_matched")
        sto = method_mean("sto_trl_matched")
        full = method_mean("bellman_full")
        matched_sweeps = int(summaries[0]["bellman_matched"]["sweeps"])
        sto_sweeps = int(summaries[0]["sto_trl_matched"]["sweeps"])
        full_sweeps = int(summaries[0]["bellman_full"]["sweeps"])
        oracle_l1_range = f"{fmt(min(oracle_l1))}-{fmt(max(oracle_l1))}"
        oracle_top1_range = f"{fmt(min(oracle_top1))}-{fmt(max(oracle_top1))}"
        expected_transition_seeds = set(str(seed) for seed in spec["expected_transition_seeds"])
        eval_setting = (
            "seed 0, 50 episodes/task"
            if len(summaries) == 1
            else f"seed 0, {len(summaries)} transition seeds, 50 episodes/task"
        )

        assert_equal(errors, row["transition_model"], transition_model, f"{env} LT model")
        assert_equal(errors, row["eval_setting"], eval_setting, f"{env} LT eval_setting")
        assert_equal(errors, row["transition_seeds"], ",".join(transition_seeds), f"{env} LT transition_seeds")
        assert_equal(errors, row["sources"], ";".join(source_labels), f"{env} LT sources")
        assert_equal(errors, int(row["matched_sweeps"]), matched_sweeps, f"{env} LT matched_sweeps")
        assert_equal(errors, sto_sweeps, matched_sweeps, f"{env} LT stochastic sweeps")
        assert_equal(errors, int(row["full_sweeps"]), full_sweeps, f"{env} LT full_sweeps")
        assert_close(errors, float(row["bellman_matched"]), round(matched, 3), f"{env} LT matched")
        assert_close(errors, float(row["stochastic_trl"]), round(sto, 3), f"{env} LT stochastic")
        assert_close(errors, float(row["bellman_full"]), round(full, 3), f"{env} LT full")
        assert_close(errors, float(row["sto_minus_matched"]), round(sto - matched, 3), f"{env} LT improvement")
        assert_equal(errors, row["oracle_l1_range"], oracle_l1_range, f"{env} LT oracle_l1_range")
        assert_equal(errors, row["oracle_top1_range"], oracle_top1_range, f"{env} LT oracle_top1_range")

        if set(transition_seeds) != expected_transition_seeds:
            errors.append(
                f"{env}: learned-transition seeds {transition_seeds} "
                f"do not match {sorted(expected_transition_seeds)}"
            )
        if matched_sweeps != 6 or sto_sweeps != 6:
            errors.append(f"{env}: learned-transition matched and stochastic sweeps are not 6")
        if full_sweeps != 180:
            errors.append(f"{env}: learned-transition full Bellman sweeps are not 180 ({full_sweeps})")
        if sto < 0.90:
            errors.append(f"{env}: learned-transition stochastic TRL success below 0.90 ({sto:.3f})")
        max_matched = float(spec["max_matched"])
        min_improvement = float(spec["min_improvement"])
        if matched > max_matched:
            errors.append(
                f"{env}: learned-transition matched Bellman success above "
                f"{max_matched:.2f} ({matched:.3f})"
            )
        if sto - matched < min_improvement:
            errors.append(
                f"{env}: learned-transition improvement below "
                f"{min_improvement:.2f} ({sto - matched:.3f})"
            )
        if abs(sto - full) > 1e-12:
            errors.append(f"{env}: learned-transition stochastic TRL does not match full Bellman")

        verified.append(
            {
                "env": env,
                "transition_model": transition_model,
                "transition_seeds": ",".join(transition_seeds),
                "matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "full": fmt(full),
                "improvement": fmt(sto - matched),
                "oracle_l1_range": oracle_l1_range,
                "oracle_top1_range": oracle_top1_range,
            }
        )
    return verified


def verify_pointmaze_bc_controller(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    if not POINTMAZE_BC_CONTROLLER_TABLE.exists():
        errors.append(
            f"PointMaze learned-controller: missing table "
            f"{POINTMAZE_BC_CONTROLLER_TABLE.relative_to(ROOT)}"
        )
        return verified

    table_rows = read_rows(POINTMAZE_BC_CONTROLLER_TABLE)
    by_env = {row["env"]: row for row in table_rows}
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for spec in POINTMAZE_BC_CONTROLLER_SPECS:
        env = str(spec["env"])
        controller = str(spec["controller"])
        paths = list(spec["paths"])
        if env not in by_env:
            errors.append(f"{env}: missing PointMaze learned-controller table row")
            continue
        row = by_env[env]
        source_rows: list[dict[str, str]] = []
        source_labels = [str(path.relative_to(ROOT)) for path in paths]
        for path in paths:
            if not path.exists():
                errors.append(f"{env}: missing learned-controller source {path.relative_to(ROOT)}")
                continue
            source_rows.extend(read_rows(path))
        if not source_rows:
            continue

        raw_episodes = distinct_values(source_rows, "episodes_per_task")
        raw_eval_seeds = distinct_values(source_rows, "seed")
        raw_tasks = distinct_values(source_rows, "task_ids")
        raw_bc_steps = distinct_values(source_rows, "bc_steps")
        raw_bc_seed = distinct_values(source_rows, "bc_seed")
        raw_goal_rep = distinct_values(source_rows, "goal_representation")
        raw_goal_mode = distinct_values(source_rows, "goal_candidate_mode")
        raw_goal_k = distinct_values(source_rows, "goal_candidates_per_state")
        raw_backend = distinct_values(source_rows, "policy_eval_backend")
        raw_lookahead = distinct_values(source_rows, "waypoint_lookahead")
        raw_action_repeat = distinct_values(source_rows, "eval_action_repeat")
        raw_transition_model = distinct_values(source_rows, "transition_model")
        if raw_episodes != {"20"}:
            errors.append(f"{env}: learned-controller episodes {sorted(raw_episodes)} do not match 20")
        if raw_eval_seeds != {"0", "1", "2"}:
            errors.append(f"{env}: learned-controller eval seeds {sorted(raw_eval_seeds)} do not match 0,1,2")
        if raw_tasks != {"all"}:
            errors.append(f"{env}: learned-controller task_ids {sorted(raw_tasks)} do not match all")
        if raw_bc_steps != {"5000"} or raw_bc_seed != {"0"}:
            errors.append(
                f"{env}: learned-controller BC metadata steps/seeds "
                f"{sorted(raw_bc_steps)}/{sorted(raw_bc_seed)} do not match 5000/0"
            )
        if raw_goal_rep != {"full"} or raw_goal_mode != {"body_nearest"} or raw_goal_k != {"16"}:
            errors.append(
                f"{env}: learned-controller goal metadata "
                f"{sorted(raw_goal_rep)}/{sorted(raw_goal_mode)}/{sorted(raw_goal_k)} "
                "does not match full/body_nearest/16"
            )
        if raw_backend != {"jax"}:
            errors.append(f"{env}: learned-controller backend {sorted(raw_backend)} does not match jax")
        if raw_lookahead != {"1"} or raw_action_repeat != {"1"}:
            errors.append(
                f"{env}: learned-controller lookahead/action-repeat "
                f"{sorted(raw_lookahead)}/{sorted(raw_action_repeat)} does not match 1/1"
            )
        if raw_transition_model != {"topology"}:
            errors.append(f"{env}: learned-controller transition_model {sorted(raw_transition_model)} does not match topology")

        summary = summarize_rows(source_rows)
        if not required.issubset(summary):
            errors.append(f"{env}: learned-controller source missing methods {sorted(required - set(summary))}")
            continue

        matched = summary["bellman_matched"]["success"]
        sto = summary["sto_trl_matched"]["success"]
        full = summary["bellman_full"]["success"]
        matched_sweeps = int(summary["bellman_matched"]["sweeps"])
        sto_sweeps = int(summary["sto_trl_matched"]["sweeps"])
        full_sweeps = int(summary["bellman_full"]["sweeps"])
        eval_seeds = ",".join(str(seed) for seed in sorted(raw_eval_seeds, key=lambda x: int(float(x))))

        assert_equal(errors, row["controller"], controller, f"{env} learned-controller controller")
        assert_equal(errors, row["eval_setting"], "3 eval seeds, 20 episodes/task", f"{env} learned-controller eval_setting")
        assert_equal(errors, row["eval_seeds"], eval_seeds, f"{env} learned-controller eval_seeds")
        assert_equal(errors, row["controller_steps"], "5000", f"{env} learned-controller controller_steps")
        assert_equal(errors, row["sources"], ";".join(source_labels), f"{env} learned-controller sources")
        assert_equal(errors, int(row["matched_sweeps"]), matched_sweeps, f"{env} learned-controller matched_sweeps")
        assert_equal(errors, sto_sweeps, matched_sweeps, f"{env} learned-controller stochastic sweeps")
        assert_equal(errors, int(row["full_sweeps"]), full_sweeps, f"{env} learned-controller full_sweeps")
        assert_close(errors, float(row["bellman_matched"]), round(matched, 3), f"{env} learned-controller matched")
        assert_close(errors, float(row["stochastic_trl"]), round(sto, 3), f"{env} learned-controller stochastic")
        assert_close(errors, float(row["bellman_full"]), round(full, 3), f"{env} learned-controller full")
        assert_close(errors, float(row["sto_minus_matched"]), round(sto - matched, 3), f"{env} learned-controller improvement")

        if matched_sweeps != 6 or sto_sweeps != 6:
            errors.append(f"{env}: learned-controller matched and stochastic sweeps are not 6")
        if full_sweeps != 180:
            errors.append(f"{env}: learned-controller full Bellman sweeps are not 180 ({full_sweeps})")
        if sto < 0.99:
            errors.append(f"{env}: learned-controller stochastic TRL success below 0.99 ({sto:.3f})")
        if matched > float(spec["max_matched"]):
            errors.append(f"{env}: learned-controller matched Bellman above threshold ({matched:.3f})")
        if sto - matched < float(spec["min_improvement"]):
            errors.append(f"{env}: learned-controller improvement below threshold ({sto - matched:.3f})")
        if abs(sto - full) > 1e-12:
            errors.append(f"{env}: learned-controller stochastic TRL does not match full Bellman")

        verified.append(
            {
                "env": env,
                "controller": controller,
                "eval_seeds": eval_seeds,
                "matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "full": fmt(full),
                "improvement": fmt(sto - matched),
            }
        )
    return verified


def verify_controller_execution_isolation(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    if not CONTROLLER_EXECUTION_ISOLATION_TABLE.exists():
        errors.append(
            f"Controller execution isolation: missing table "
            f"{CONTROLLER_EXECUTION_ISOLATION_TABLE.relative_to(ROOT)}"
        )
        return verified

    table_rows = read_rows(CONTROLLER_EXECUTION_ISOLATION_TABLE)
    by_key = {(row["execution_path"], row["env"]): row for row in table_rows}
    best_direct = 0.0

    for spec in POINTMAZE_DIRECT_ACTOR_SPECS:
        execution_path = str(spec["execution_path"])
        env = str(spec["env"])
        path = Path(spec["path"])
        key = (execution_path, env)
        if key not in by_key:
            errors.append(f"{execution_path}: missing controller-isolation table row")
            continue
        table = by_key[key]
        if not path.exists():
            errors.append(f"{execution_path}: missing direct actor source {path.relative_to(ROOT)}")
            continue
        source_rows = read_rows(path)
        values = [float(row["evaluation/overall_success"]) for row in source_rows]
        if not values:
            errors.append(f"{execution_path}: direct actor source has no eval rows")
            continue
        best = max(values)
        final = values[-1]
        best_direct = max(best_direct, best)

        assert_equal(errors, table["controller_or_actor"], spec["controller_or_actor"], execution_path)
        assert_equal(errors, table["eval_setting"], spec["eval_setting"], execution_path)
        assert_equal(errors, table["source"], str(path.relative_to(ROOT)), execution_path)
        assert_close(errors, float(table["best_success"]), round(best, 3), f"{execution_path} best")
        assert_close(errors, float(table["final_success"]), round(final, 3), f"{execution_path} final")
        if best > 0.15 or final > 0.15:
            errors.append(f"{execution_path}: direct actor exceeds 0.15 success ({best:.3f}/{final:.3f})")

        verified.append(
            {
                "execution_path": execution_path,
                "env": env,
                "best_success": fmt(best),
                "final_success": fmt(final),
                "source": str(path.relative_to(ROOT)),
            }
        )

    if not POINTMAZE_BC_CONTROLLER_TABLE.exists():
        errors.append(
            f"Controller execution isolation: missing learned-controller table "
            f"{POINTMAZE_BC_CONTROLLER_TABLE.relative_to(ROOT)}"
        )
        return verified

    for learned_row in read_rows(POINTMAZE_BC_CONTROLLER_TABLE):
        execution_path = "Stochastic TRL + learned waypoint BC"
        env = learned_row["env"]
        key = (execution_path, env)
        if key not in by_key:
            errors.append(f"{env}: missing learned waypoint controller-isolation table row")
            continue
        table = by_key[key]
        success = float(learned_row["stochastic_trl"])
        assert_equal(errors, table["controller_or_actor"], learned_row["controller"], f"{env} waypoint controller")
        assert_equal(errors, table["eval_setting"], learned_row["eval_setting"], f"{env} waypoint controller")
        assert_equal(errors, table["source"], learned_row["sources"], f"{env} waypoint controller")
        assert_close(errors, float(table["best_success"]), round(success, 3), f"{env} waypoint best")
        assert_close(errors, float(table["final_success"]), round(success, 3), f"{env} waypoint final")
        if success < 0.99:
            errors.append(f"{env}: waypoint executor stochastic TRL below 0.99 ({success:.3f})")
        if env == "pointmaze-teleport-navigate-v0" and success - best_direct < 0.85:
            errors.append(
                f"{env}: waypoint/direct actor gap below 0.85 ({success - best_direct:.3f})"
            )

        verified.append(
            {
                "execution_path": execution_path,
                "env": env,
                "best_success": fmt(success),
                "final_success": fmt(success),
                "source": learned_row["sources"],
            }
        )
    return verified


def verify_fast_eval_profile(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    if not FAST_EVAL_PROFILE_TABLE.exists():
        errors.append(f"Fast eval profile: missing table {FAST_EVAL_PROFILE_TABLE.relative_to(ROOT)}")
        return verified

    table_rows = read_rows(FAST_EVAL_PROFILE_TABLE)
    by_source = {row["source"]: row for row in table_rows}
    stitch_ep2_baseline: float | None = None
    stitch_ep2_repeat2: float | None = None

    for spec in FAST_EVAL_PROFILE_SPECS:
        path = Path(spec["path"])
        source = str(path.relative_to(ROOT))
        if source not in by_source:
            errors.append(f"{spec['screen']} {spec['role']}: missing fast-profile table row")
            continue
        table = by_source[source]
        if not path.exists():
            errors.append(f"{spec['screen']} {spec['role']}: missing fast-profile source {source}")
            continue
        source_rows = read_rows(path)
        by_method = {row["method"]: row for row in source_rows}
        if "bellman_matched" not in by_method or "sto_trl_matched" not in by_method:
            errors.append(f"{source}: missing matched or stochastic fast-profile rows")
            continue
        matched = by_method["bellman_matched"]
        sto = by_method["sto_trl_matched"]
        matched_success = float(matched["overall_success"])
        sto_success = float(sto["overall_success"])
        improvement = sto_success - matched_success

        assert_equal(errors, table["screen"], spec["screen"], f"{source} screen")
        assert_equal(errors, table["role"], spec["role"], f"{source} role")
        assert_equal(errors, table["env"], sto["env"], f"{source} env")
        assert_equal(errors, table["task_ids"], sto["task_ids"], f"{source} task_ids")
        assert_equal(errors, table["seed"], sto["seed"], f"{source} seed")
        assert_equal(errors, table["episodes_per_task"], sto["episodes_per_task"], f"{source} episodes")
        assert_equal(errors, table["eval_action_repeat"], sto["eval_action_repeat"], f"{source} action repeat")
        assert_close(errors, float(table["matched_success"]), round(matched_success, 3), f"{source} matched")
        assert_close(errors, float(table["stochastic_trl"]), round(sto_success, 3), f"{source} stochastic")
        assert_close(errors, float(table["sto_minus_matched"]), round(improvement, 3), f"{source} improvement")
        assert_close(errors, float(table["setup_seconds"]), round(float(sto["setup_seconds"]), 2), f"{source} setup")
        assert_close(
            errors,
            float(table["matched_eval_seconds"]),
            round(float(matched["eval_seconds"]), 2),
            f"{source} matched eval seconds",
        )
        assert_close(
            errors,
            float(table["sto_eval_seconds"]),
            round(float(sto["eval_seconds"]), 2),
            f"{source} stochastic eval seconds",
        )

        if spec["role"] == "recommended fast screen":
            if table["eval_action_repeat"] != "1":
                errors.append(f"{source}: recommended fast screen does not use action repeat 1")
            if sto_success < float(spec["min_sto"]):
                errors.append(f"{source}: stochastic fast-profile success below threshold ({sto_success:.3f})")
            if improvement < float(spec["min_improvement"]):
                errors.append(f"{source}: fast-profile improvement below threshold ({improvement:.3f})")
        if spec["role"] == "two-episode baseline":
            stitch_ep2_baseline = sto_success
        if spec["role"] == "action-repeat ablation":
            stitch_ep2_repeat2 = sto_success

        verified.append(
            {
                "screen": str(spec["screen"]),
                "role": str(spec["role"]),
                "matched": fmt(matched_success),
                "stochastic_trl": fmt(sto_success),
                "improvement": fmt(improvement),
                "sto_eval_seconds": f"{float(sto['eval_seconds']):.2f}",
            }
        )

    if stitch_ep2_baseline is not None and stitch_ep2_repeat2 is not None:
        if stitch_ep2_repeat2 >= stitch_ep2_baseline:
            errors.append(
                "fast-profile action-repeat ablation is not lower than the repeat-1 baseline; "
                "re-check the unsafe-action-repeat conclusion"
            )
    return verified


def verify_antmaze_support_ablation(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    if not ANTMAZE_SUPPORT_ABLATION_TABLE.exists():
        errors.append(
            f"AntMaze support ablation: missing table "
            f"{ANTMAZE_SUPPORT_ABLATION_TABLE.relative_to(ROOT)}"
        )
        return verified

    table_rows = read_rows(ANTMAZE_SUPPORT_ABLATION_TABLE)
    by_source = {row["source"]: row for row in table_rows}
    required = {"bellman_matched", "support_trl_matched", "sto_trl_matched", "bellman_full"}
    for spec in ANTMAZE_SUPPORT_ABLATION_SPECS:
        path = Path(spec["path"])
        source = str(path.relative_to(ROOT))
        if source not in by_source:
            errors.append(f"{spec['env']}: missing AntMaze support-ablation table row")
            continue
        table = by_source[source]
        if not path.exists():
            errors.append(f"{spec['env']}: missing AntMaze support-ablation source {source}")
            continue
        source_rows = read_rows(path)
        by_method = {row["method"]: row for row in source_rows}
        if not required.issubset(by_method):
            errors.append(f"{source}: missing support-ablation methods {sorted(required - set(by_method))}")
            continue

        matched = float(by_method["bellman_matched"]["overall_success"])
        support = float(by_method["support_trl_matched"]["overall_success"])
        sto = float(by_method["sto_trl_matched"]["overall_success"])
        full = float(by_method["bellman_full"]["overall_success"])
        support_minus_matched = support - matched
        sto_minus_support = sto - support
        sto_row = by_method["sto_trl_matched"]

        assert_equal(errors, table["env"], spec["env"], f"{source} env")
        assert_equal(errors, table["controller"], spec["controller"], f"{source} controller")
        assert_close(errors, float(table["bellman_matched"]), round(matched, 3), f"{source} matched")
        assert_close(errors, float(table["support_trl"]), round(support, 3), f"{source} support")
        assert_close(errors, float(table["stochastic_trl"]), round(sto, 3), f"{source} stochastic")
        assert_close(errors, float(table["bellman_full"]), round(full, 3), f"{source} full")
        assert_close(
            errors,
            float(table["support_minus_matched"]),
            round(support_minus_matched, 3),
            f"{source} support-minus-matched",
        )
        assert_close(
            errors,
            float(table["sto_minus_support"]),
            round(sto_minus_support, 3),
            f"{source} sto-minus-support",
        )

        if sto_row["policy_eval_backend"] != "jax":
            errors.append(f"{source}: support ablation must use JAX policy backend")
        if sto_row["task_ids"] != "4,5":
            errors.append(f"{source}: support ablation must use task IDs 4,5")
        if sto_row["episodes_per_task"] != "5":
            errors.append(f"{source}: support ablation must use 5 episodes per task")
        if sto_row["eval_action_repeat"] != "1":
            errors.append(f"{source}: support ablation must use action repeat 1")
        if int(float(sto_row["iters"])) != 6 or int(float(by_method["support_trl_matched"]["iters"])) != 6:
            errors.append(f"{source}: support ablation expected matched/support sweeps 6")
        if int(float(by_method["bellman_full"]["iters"])) != 180:
            errors.append(f"{source}: support ablation expected full Bellman sweeps 180")
        if sto < float(spec["min_sto"]):
            errors.append(f"{source}: stochastic TRL support-ablation success below threshold ({sto:.3f})")
        if abs(support_minus_matched) > 1e-12:
            errors.append(
                f"{source}: support-only TRL should match matched Bellman in this diagnostic "
                f"({support:.3f} vs {matched:.3f})"
            )
        if sto_minus_support < float(spec["min_sto_minus_support"]):
            errors.append(f"{source}: stochastic TRL gap over support below threshold ({sto_minus_support:.3f})")
        if full + 1e-12 < sto:
            errors.append(f"{source}: full Bellman is below stochastic TRL in support ablation")

        verified.append(
            {
                "env": str(spec["env"]),
                "matched": fmt(matched),
                "support_trl": fmt(support),
                "stochastic_trl": fmt(sto),
                "full": fmt(full),
                "sto_minus_support": fmt(sto_minus_support),
            }
        )
    return verified


def verify_pointmaze_tie_policy_head(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    if not POINTMAZE_TIE_POLICY_HEAD_TABLE.exists():
        errors.append(
            f"PointMaze tie-policy head: missing table "
            f"{POINTMAZE_TIE_POLICY_HEAD_TABLE.relative_to(ROOT)}"
        )
        return verified

    table_rows = read_rows(POINTMAZE_TIE_POLICY_HEAD_TABLE)
    by_env = {row["env"]: row for row in table_rows}
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for env, source in POINTMAZE_TIE_POLICY_HEAD_SPECS:
        if not source.exists():
            errors.append(f"{env}: missing tie-policy source {source.relative_to(ROOT)}")
            continue
        if env not in by_env:
            errors.append(f"{env}: missing generated tie-policy table row")
            continue
        table = by_env[env]
        source_rows = read_rows(source)

        raw_env = distinct_values(source_rows, "env")
        raw_eval_seed = distinct_values(source_rows, "seed")
        raw_episodes = distinct_values(source_rows, "episodes_per_task")
        raw_tasks = distinct_values(source_rows, "task_ids")
        raw_topology = distinct_values(source_rows, "topology_source")
        raw_transition_model = distinct_values(source_rows, "transition_model")
        raw_transition_source = distinct_values(source_rows, "transition_target_source")
        raw_transition_steps = distinct_values(source_rows, "transition_steps")
        raw_value_model = distinct_values(source_rows, "value_model")
        raw_value_steps = distinct_values(source_rows, "value_steps")
        if raw_env != {env}:
            errors.append(f"{env}: raw tie-policy env {sorted(raw_env)} does not match")
        if raw_eval_seed != {"0"}:
            errors.append(f"{env}: raw tie-policy eval seed {sorted(raw_eval_seed)} does not match 0")
        if raw_episodes != {"20"}:
            errors.append(f"{env}: raw tie-policy episodes {sorted(raw_episodes)} do not match 20")
        if raw_tasks != {"all"}:
            errors.append(f"{env}: raw tie-policy task_ids {sorted(raw_tasks)} do not match all")
        if raw_topology != {"dataset"}:
            errors.append(f"{env}: raw tie-policy topology {sorted(raw_topology)} does not match dataset")
        if raw_transition_model != {"raw_obs_mlp"} or raw_transition_source != {"dataset_cell_changes"}:
            errors.append(
                f"{env}: raw tie-policy transition metadata "
                f"{sorted(raw_transition_model)}/{sorted(raw_transition_source)} "
                "does not match raw_obs_mlp/dataset_cell_changes"
            )
        if raw_transition_steps != {"2000"}:
            errors.append(f"{env}: raw tie-policy transition steps {sorted(raw_transition_steps)} do not match 2000")
        if raw_value_model != {"raw_obs_tie_policy_mlp"} or raw_value_steps != {"3000"}:
            errors.append(
                f"{env}: raw tie-policy value metadata "
                f"{sorted(raw_value_model)}/{sorted(raw_value_steps)} "
                "does not match raw_obs_tie_policy_mlp/3000"
            )

        summary = summarize_rows(source_rows)
        if not required.issubset(summary):
            errors.append(f"{env}: tie-policy source missing methods {sorted(required - set(summary))}")
            continue
        by_method = {row["method"]: row for row in source_rows}

        matched = summary["bellman_matched"]["success"]
        sto = summary["sto_trl_matched"]["success"]
        full = summary["bellman_full"]["success"]
        matched_sweeps = int(summary["bellman_matched"]["sweeps"])
        sto_sweeps = int(summary["sto_trl_matched"]["sweeps"])
        full_sweeps = int(summary["bellman_full"]["sweeps"])
        sto_row = by_method["sto_trl_matched"]
        transition_top1 = float(sto_row["transition_top1"])
        value_action_agreement = float(sto_row["value_action_agreement"])

        assert_equal(errors, table["transition_model"], "raw-observation MLP cell-change", f"{env} tie transition")
        assert_equal(errors, table["control_head"], "raw-observation tie-policy MLP", f"{env} tie head")
        assert_equal(errors, table["eval_setting"], "seed 0, all tasks, 20 episodes/task", f"{env} tie eval")
        assert_equal(errors, table["eval_seed"], "0", f"{env} tie eval seed")
        assert_equal(errors, int(table["matched_sweeps"]), matched_sweeps, f"{env} tie matched sweeps")
        assert_equal(errors, sto_sweeps, matched_sweeps, f"{env} tie stochastic sweeps")
        assert_equal(errors, int(table["full_sweeps"]), full_sweeps, f"{env} tie full sweeps")
        assert_equal(errors, table["source"], str(source.relative_to(ROOT)), f"{env} tie source")
        assert_close(errors, float(table["bellman_matched"]), round(matched, 3), f"{env} tie matched")
        assert_close(errors, float(table["stochastic_trl"]), round(sto, 3), f"{env} tie stochastic")
        assert_close(errors, float(table["bellman_full"]), round(full, 3), f"{env} tie full")
        assert_close(errors, float(table["sto_minus_matched"]), round(sto - matched, 3), f"{env} tie gain")
        assert_close(errors, float(table["transition_top1"]), round(transition_top1, 3), f"{env} tie transition top1")
        assert_close(
            errors,
            float(table["value_action_agreement"]),
            round(value_action_agreement, 3),
            f"{env} tie action agreement",
        )

        if matched_sweeps != 6 or sto_sweeps != 6:
            errors.append(f"{env}: tie-policy matched and stochastic TRL sweeps are not 6")
        if full_sweeps != 180:
            errors.append(f"{env}: tie-policy full Bellman sweeps are not 180 ({full_sweeps})")
        if sto < 0.95:
            errors.append(f"{env}: tie-policy stochastic TRL success below 0.95 ({sto:.3f})")
        if matched > 0.60:
            errors.append(f"{env}: tie-policy matched Bellman success above 0.60 ({matched:.3f})")
        if sto - matched < 0.40:
            errors.append(f"{env}: tie-policy improvement below 0.40 ({sto - matched:.3f})")
        if abs(sto - full) > 1e-12:
            errors.append(f"{env}: tie-policy stochastic TRL does not match full Bellman")
        if value_action_agreement < 0.95:
            errors.append(f"{env}: tie-policy action agreement below 0.95 ({value_action_agreement:.3f})")

        verified.append(
            {
                "env": env,
                "matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "full": fmt(full),
                "improvement": fmt(sto - matched),
                "transition_top1": fmt(transition_top1),
                "value_action_agreement": fmt(value_action_agreement),
            }
        )
    return verified


def verify_pointmaze_prev_policy_head(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    if not POINTMAZE_PREV_POLICY_HEAD_TABLE.exists():
        errors.append(
            f"PointMaze previous-action policy head: missing table "
            f"{POINTMAZE_PREV_POLICY_HEAD_TABLE.relative_to(ROOT)}"
        )
        return verified

    table_rows = read_rows(POINTMAZE_PREV_POLICY_HEAD_TABLE)
    by_env = {row["env"]: row for row in table_rows}
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for env, source in POINTMAZE_PREV_POLICY_HEAD_SPECS:
        if not source.exists():
            errors.append(f"{env}: missing previous-action policy source {source.relative_to(ROOT)}")
            continue
        if env not in by_env:
            errors.append(f"{env}: missing generated previous-action policy table row")
            continue
        table = by_env[env]
        source_rows = read_rows(source)

        raw_env = distinct_values(source_rows, "env")
        raw_eval_seed = distinct_values(source_rows, "seed")
        raw_episodes = distinct_values(source_rows, "episodes_per_task")
        raw_tasks = distinct_values(source_rows, "task_ids")
        raw_topology = distinct_values(source_rows, "topology_source")
        raw_transition_model = distinct_values(source_rows, "transition_model")
        raw_transition_source = distinct_values(source_rows, "transition_target_source")
        raw_transition_steps = distinct_values(source_rows, "transition_steps")
        raw_value_model = distinct_values(source_rows, "value_model")
        raw_value_steps = distinct_values(source_rows, "value_steps")
        if raw_env != {env}:
            errors.append(f"{env}: raw previous-action policy env {sorted(raw_env)} does not match")
        if raw_eval_seed != {"0"}:
            errors.append(f"{env}: raw previous-action policy eval seed {sorted(raw_eval_seed)} does not match 0")
        if raw_episodes != {"20"}:
            errors.append(f"{env}: raw previous-action policy episodes {sorted(raw_episodes)} do not match 20")
        if raw_tasks != {"all"}:
            errors.append(f"{env}: raw previous-action policy task_ids {sorted(raw_tasks)} do not match all")
        if raw_topology != {"dataset"}:
            errors.append(f"{env}: raw previous-action policy topology {sorted(raw_topology)} does not match dataset")
        if raw_transition_model != {"raw_obs_mlp"} or raw_transition_source != {"dataset_cell_changes"}:
            errors.append(
                f"{env}: raw previous-action policy transition metadata "
                f"{sorted(raw_transition_model)}/{sorted(raw_transition_source)} "
                "does not match raw_obs_mlp/dataset_cell_changes"
            )
        if raw_transition_steps != {"2000"}:
            errors.append(
                f"{env}: raw previous-action policy transition steps "
                f"{sorted(raw_transition_steps)} do not match 2000"
            )
        if raw_value_model != {"raw_obs_prev_policy_mlp"} or raw_value_steps != {"1000"}:
            errors.append(
                f"{env}: raw previous-action policy value metadata "
                f"{sorted(raw_value_model)}/{sorted(raw_value_steps)} "
                "does not match raw_obs_prev_policy_mlp/1000"
            )

        summary = summarize_rows(source_rows)
        if not required.issubset(summary):
            errors.append(f"{env}: previous-action policy source missing methods {sorted(required - set(summary))}")
            continue
        by_method = {row["method"]: row for row in source_rows}

        matched = summary["bellman_matched"]["success"]
        sto = summary["sto_trl_matched"]["success"]
        full = summary["bellman_full"]["success"]
        matched_sweeps = int(summary["bellman_matched"]["sweeps"])
        sto_sweeps = int(summary["sto_trl_matched"]["sweeps"])
        full_sweeps = int(summary["bellman_full"]["sweeps"])
        sto_row = by_method["sto_trl_matched"]
        transition_top1 = float(sto_row["transition_top1"])
        value_action_agreement = float(sto_row["value_action_agreement"])

        assert_equal(errors, table["transition_model"], "raw-observation MLP cell-change", f"{env} prev transition")
        assert_equal(errors, table["control_head"], "previous-action policy MLP", f"{env} prev head")
        assert_equal(errors, table["eval_setting"], "seed 0, all tasks, 20 episodes/task", f"{env} prev eval")
        assert_equal(errors, table["eval_seed"], "0", f"{env} prev eval seed")
        assert_equal(errors, table["value_steps"], "1000", f"{env} prev value steps")
        assert_equal(errors, int(table["matched_sweeps"]), matched_sweeps, f"{env} prev matched sweeps")
        assert_equal(errors, sto_sweeps, matched_sweeps, f"{env} prev stochastic sweeps")
        assert_equal(errors, int(table["full_sweeps"]), full_sweeps, f"{env} prev full sweeps")
        assert_equal(errors, table["source"], str(source.relative_to(ROOT)), f"{env} prev source")
        assert_close(errors, float(table["bellman_matched"]), round(matched, 3), f"{env} prev matched")
        assert_close(errors, float(table["stochastic_trl"]), round(sto, 3), f"{env} prev stochastic")
        assert_close(errors, float(table["bellman_full"]), round(full, 3), f"{env} prev full")
        assert_close(errors, float(table["sto_minus_matched"]), round(sto - matched, 3), f"{env} prev gain")
        assert_close(errors, float(table["transition_top1"]), round(transition_top1, 3), f"{env} prev transition top1")
        assert_close(
            errors,
            float(table["value_action_agreement"]),
            round(value_action_agreement, 3),
            f"{env} prev action agreement",
        )

        if matched_sweeps != 6 or sto_sweeps != 6:
            errors.append(f"{env}: previous-action policy matched and stochastic TRL sweeps are not 6")
        if full_sweeps != 180:
            errors.append(f"{env}: previous-action policy full Bellman sweeps are not 180 ({full_sweeps})")
        if sto < 0.95:
            errors.append(f"{env}: previous-action policy stochastic TRL success below 0.95 ({sto:.3f})")
        if matched > 0.60:
            errors.append(f"{env}: previous-action policy matched Bellman success above 0.60 ({matched:.3f})")
        if sto - matched < 0.40:
            errors.append(f"{env}: previous-action policy improvement below 0.40 ({sto - matched:.3f})")
        if abs(sto - full) > 1e-12:
            errors.append(f"{env}: previous-action policy stochastic TRL does not match full Bellman")
        if value_action_agreement < 0.99:
            errors.append(
                f"{env}: previous-action policy action agreement below 0.99 "
                f"({value_action_agreement:.3f})"
            )

        verified.append(
            {
                "env": env,
                "matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "full": fmt(full),
                "improvement": fmt(sto - matched),
                "transition_top1": fmt(transition_top1),
                "value_action_agreement": fmt(value_action_agreement),
            }
        )
    return verified


def verify_pointmaze_tie_policy_head_eval_seed(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    if not POINTMAZE_TIE_POLICY_HEAD_EVAL_SEED_TABLE.exists():
        errors.append(
            f"PointMaze tie-policy eval-seed: missing table "
            f"{POINTMAZE_TIE_POLICY_HEAD_EVAL_SEED_TABLE.relative_to(ROOT)}"
        )
        return verified

    table_rows = read_rows(POINTMAZE_TIE_POLICY_HEAD_EVAL_SEED_TABLE)
    by_env = {row["env"]: row for row in table_rows}
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for spec in POINTMAZE_TIE_POLICY_HEAD_EVAL_SEED_SPECS:
        env = str(spec["env"])
        sources = list(spec["sources"])
        if env not in by_env:
            errors.append(f"{env}: missing generated tie-policy eval-seed table row")
            continue
        table = by_env[env]
        source_rows: list[dict[str, str]] = []
        source_labels = [str(path.relative_to(ROOT)) for path in sources]
        for source in sources:
            if not source.exists():
                errors.append(f"{env}: missing tie-policy eval-seed source {source.relative_to(ROOT)}")
                continue
            source_rows.extend(read_rows(source))
        if not source_rows:
            continue

        raw_env = distinct_values(source_rows, "env")
        raw_eval_seeds = distinct_values(source_rows, "seed")
        raw_episodes = distinct_values(source_rows, "episodes_per_task")
        raw_tasks = distinct_values(source_rows, "task_ids")
        raw_topology = distinct_values(source_rows, "topology_source")
        raw_transition_model = distinct_values(source_rows, "transition_model")
        raw_transition_source = distinct_values(source_rows, "transition_target_source")
        raw_transition_steps = distinct_values(source_rows, "transition_steps")
        raw_value_model = distinct_values(source_rows, "value_model")
        raw_value_steps = distinct_values(source_rows, "value_steps")
        if raw_env != {env}:
            errors.append(f"{env}: raw tie-policy eval-seed env {sorted(raw_env)} does not match")
        if raw_eval_seeds != {"0", "1", "2"}:
            errors.append(f"{env}: raw tie-policy eval seeds {sorted(raw_eval_seeds)} do not match 0,1,2")
        if raw_episodes != {"20"} or raw_tasks != {"all"}:
            errors.append(
                f"{env}: raw tie-policy eval-seed episodes/task_ids "
                f"{sorted(raw_episodes)}/{sorted(raw_tasks)} do not match 20/all"
            )
        if raw_topology != {"dataset"}:
            errors.append(f"{env}: raw tie-policy eval-seed topology {sorted(raw_topology)} does not match dataset")
        if raw_transition_model != {"raw_obs_mlp"} or raw_transition_source != {"dataset_cell_changes"}:
            errors.append(
                f"{env}: raw tie-policy eval-seed transition metadata "
                f"{sorted(raw_transition_model)}/{sorted(raw_transition_source)} "
                "does not match raw_obs_mlp/dataset_cell_changes"
            )
        if raw_transition_steps != {"2000"}:
            errors.append(
                f"{env}: raw tie-policy eval-seed transition steps "
                f"{sorted(raw_transition_steps)} do not match 2000"
            )
        if raw_value_model != {"raw_obs_tie_policy_mlp"} or raw_value_steps != {"3000"}:
            errors.append(
                f"{env}: raw tie-policy eval-seed value metadata "
                f"{sorted(raw_value_model)}/{sorted(raw_value_steps)} "
                "does not match raw_obs_tie_policy_mlp/3000"
            )

        by_method: dict[str, list[dict[str, str]]] = {}
        for row in source_rows:
            by_method.setdefault(row["method"], []).append(row)
        if not required.issubset(by_method):
            errors.append(f"{env}: tie-policy eval-seed source missing methods {sorted(required - set(by_method))}")
            continue

        def mean_method(method: str) -> float:
            return mean(float(row["overall_success"]) for row in by_method[method])

        matched = mean_method("bellman_matched")
        sto = mean_method("sto_trl_matched")
        full = mean_method("bellman_full")
        matched_sweeps = int(float(by_method["bellman_matched"][0]["iters"]))
        sto_sweeps = int(float(by_method["sto_trl_matched"][0]["iters"]))
        full_sweeps = int(float(by_method["bellman_full"][0]["iters"]))
        eval_seeds = ",".join(str(seed) for seed in sorted(raw_eval_seeds, key=lambda x: int(float(x))))
        transition_top1 = min(float(row["transition_top1"]) for row in by_method["sto_trl_matched"])
        value_action_agreement = min(float(row["value_action_agreement"]) for row in by_method["sto_trl_matched"])

        assert_equal(errors, table["transition_model"], "raw-observation MLP cell-change", f"{env} tie eval transition")
        assert_equal(errors, table["control_head"], "raw-observation tie-policy MLP", f"{env} tie eval head")
        assert_equal(
            errors,
            table["eval_setting"],
            "3 eval seeds, all tasks, 20 episodes/task",
            f"{env} tie eval setting",
        )
        assert_equal(errors, table["eval_seeds"], eval_seeds, f"{env} tie eval seeds")
        assert_equal(errors, int(table["matched_sweeps"]), matched_sweeps, f"{env} tie eval matched sweeps")
        assert_equal(errors, sto_sweeps, matched_sweeps, f"{env} tie eval stochastic sweeps")
        assert_equal(errors, int(table["full_sweeps"]), full_sweeps, f"{env} tie eval full sweeps")
        assert_equal(errors, table["sources"], ";".join(source_labels), f"{env} tie eval sources")
        assert_close(errors, float(table["bellman_matched"]), round(matched, 3), f"{env} tie eval matched")
        assert_close(errors, float(table["stochastic_trl"]), round(sto, 3), f"{env} tie eval stochastic")
        assert_close(errors, float(table["bellman_full"]), round(full, 3), f"{env} tie eval full")
        assert_close(errors, float(table["sto_minus_matched"]), round(sto - matched, 3), f"{env} tie eval gain")
        assert_close(errors, float(table["transition_top1"]), round(transition_top1, 3), f"{env} tie eval transition")
        assert_close(
            errors,
            float(table["value_action_agreement_min"]),
            round(value_action_agreement, 3),
            f"{env} tie eval action agreement",
        )

        if matched_sweeps != 6 or sto_sweeps != 6:
            errors.append(f"{env}: tie-policy eval matched and stochastic TRL sweeps are not 6")
        if full_sweeps != 180:
            errors.append(f"{env}: tie-policy eval full Bellman sweeps are not 180 ({full_sweeps})")
        if sto < 0.90:
            errors.append(f"{env}: tie-policy eval stochastic TRL success below 0.90 ({sto:.3f})")
        if matched > 0.45:
            errors.append(f"{env}: tie-policy eval matched Bellman success above 0.45 ({matched:.3f})")
        if sto - matched < 0.45:
            errors.append(f"{env}: tie-policy eval improvement below 0.45 ({sto - matched:.3f})")
        if abs(sto - full) > 1e-12:
            errors.append(f"{env}: tie-policy eval stochastic TRL does not match full Bellman")
        if value_action_agreement < 0.95:
            errors.append(
                f"{env}: tie-policy eval action agreement below 0.95 ({value_action_agreement:.3f})"
            )

        verified.append(
            {
                "env": env,
                "eval_seeds": eval_seeds,
                "matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "full": fmt(full),
                "improvement": fmt(sto - matched),
                "transition_top1": fmt(transition_top1),
                "value_action_agreement_min": fmt(value_action_agreement),
            }
        )
    return verified


def verify_pointmaze_tie_policy_head_transition_seed(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    if not POINTMAZE_TIE_POLICY_HEAD_TRANSITION_SEED_TABLE.exists():
        errors.append(
            f"PointMaze tie-policy transition-seed: missing table "
            f"{POINTMAZE_TIE_POLICY_HEAD_TRANSITION_SEED_TABLE.relative_to(ROOT)}"
        )
        return verified

    table_rows = read_rows(POINTMAZE_TIE_POLICY_HEAD_TRANSITION_SEED_TABLE)
    by_env = {row["env"]: row for row in table_rows}
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for spec in POINTMAZE_TIE_POLICY_HEAD_TRANSITION_SEED_SPECS:
        env = str(spec["env"])
        sources = list(spec["sources"])
        if env not in by_env:
            errors.append(f"{env}: missing generated tie-policy transition-seed table row")
            continue
        table = by_env[env]
        source_rows: list[dict[str, str]] = []
        source_labels = [str(path.relative_to(ROOT)) for path in sources]
        for source in sources:
            if not source.exists():
                errors.append(f"{env}: missing tie-policy transition-seed source {source.relative_to(ROOT)}")
                continue
            source_rows.extend(read_rows(source))
        if not source_rows:
            continue

        raw_env = distinct_values(source_rows, "env")
        raw_eval_seed = distinct_values(source_rows, "seed")
        raw_transition_seeds = distinct_values(source_rows, "transition_seed")
        raw_episodes = distinct_values(source_rows, "episodes_per_task")
        raw_tasks = distinct_values(source_rows, "task_ids")
        raw_topology = distinct_values(source_rows, "topology_source")
        raw_transition_model = distinct_values(source_rows, "transition_model")
        raw_transition_source = distinct_values(source_rows, "transition_target_source")
        raw_transition_steps = distinct_values(source_rows, "transition_steps")
        raw_value_model = distinct_values(source_rows, "value_model")
        raw_value_steps = distinct_values(source_rows, "value_steps")
        if raw_env != {env}:
            errors.append(f"{env}: raw tie-policy transition-seed env {sorted(raw_env)} does not match")
        if raw_eval_seed != {"0"}:
            errors.append(f"{env}: raw tie-policy transition-seed eval seed {sorted(raw_eval_seed)} does not match 0")
        if raw_transition_seeds != {"0", "1", "2"}:
            errors.append(
                f"{env}: raw tie-policy transition seeds {sorted(raw_transition_seeds)} do not match 0,1,2"
            )
        if raw_episodes != {"20"} or raw_tasks != {"all"}:
            errors.append(
                f"{env}: raw tie-policy transition-seed episodes/task_ids "
                f"{sorted(raw_episodes)}/{sorted(raw_tasks)} do not match 20/all"
            )
        if raw_topology != {"dataset"}:
            errors.append(
                f"{env}: raw tie-policy transition-seed topology {sorted(raw_topology)} does not match dataset"
            )
        if raw_transition_model != {"raw_obs_mlp"} or raw_transition_source != {"dataset_cell_changes"}:
            errors.append(
                f"{env}: raw tie-policy transition-seed transition metadata "
                f"{sorted(raw_transition_model)}/{sorted(raw_transition_source)} "
                "does not match raw_obs_mlp/dataset_cell_changes"
            )
        if raw_transition_steps != {"2000"}:
            errors.append(
                f"{env}: raw tie-policy transition-seed transition steps "
                f"{sorted(raw_transition_steps)} do not match 2000"
            )
        if raw_value_model != {"raw_obs_tie_policy_mlp"} or raw_value_steps != {"3000"}:
            errors.append(
                f"{env}: raw tie-policy transition-seed value metadata "
                f"{sorted(raw_value_model)}/{sorted(raw_value_steps)} "
                "does not match raw_obs_tie_policy_mlp/3000"
            )

        by_method: dict[str, list[dict[str, str]]] = {}
        for row in source_rows:
            by_method.setdefault(row["method"], []).append(row)
        if not required.issubset(by_method):
            errors.append(
                f"{env}: tie-policy transition-seed source missing methods {sorted(required - set(by_method))}"
            )
            continue

        def mean_method(method: str) -> float:
            return mean(float(row["overall_success"]) for row in by_method[method])

        matched = mean_method("bellman_matched")
        sto = mean_method("sto_trl_matched")
        full = mean_method("bellman_full")
        matched_sweeps = int(float(by_method["bellman_matched"][0]["iters"]))
        sto_sweeps = int(float(by_method["sto_trl_matched"][0]["iters"]))
        full_sweeps = int(float(by_method["bellman_full"][0]["iters"]))
        transition_seeds = ",".join(
            str(seed) for seed in sorted(raw_transition_seeds, key=lambda x: int(float(x)))
        )
        transition_top1 = min(float(row["transition_top1"]) for row in by_method["sto_trl_matched"])
        value_action_agreement = min(float(row["value_action_agreement"]) for row in by_method["sto_trl_matched"])

        assert_equal(errors, table["transition_model"], "raw-observation MLP cell-change", f"{env} tie tseed transition")
        assert_equal(errors, table["control_head"], "raw-observation tie-policy MLP", f"{env} tie tseed head")
        assert_equal(
            errors,
            table["eval_setting"],
            "seed 0, 3 transition seeds, all tasks, 20 episodes/task",
            f"{env} tie tseed setting",
        )
        assert_equal(errors, table["eval_seed"], "0", f"{env} tie tseed eval seed")
        assert_equal(errors, table["transition_seeds"], transition_seeds, f"{env} tie tseed seeds")
        assert_equal(errors, int(table["matched_sweeps"]), matched_sweeps, f"{env} tie tseed matched sweeps")
        assert_equal(errors, sto_sweeps, matched_sweeps, f"{env} tie tseed stochastic sweeps")
        assert_equal(errors, int(table["full_sweeps"]), full_sweeps, f"{env} tie tseed full sweeps")
        assert_equal(errors, table["sources"], ";".join(source_labels), f"{env} tie tseed sources")
        assert_close(errors, float(table["bellman_matched"]), round(matched, 3), f"{env} tie tseed matched")
        assert_close(errors, float(table["stochastic_trl"]), round(sto, 3), f"{env} tie tseed stochastic")
        assert_close(errors, float(table["bellman_full"]), round(full, 3), f"{env} tie tseed full")
        assert_close(errors, float(table["sto_minus_matched"]), round(sto - matched, 3), f"{env} tie tseed gain")
        assert_close(errors, float(table["transition_top1_min"]), round(transition_top1, 3), f"{env} tie tseed transition")
        assert_close(
            errors,
            float(table["value_action_agreement_min"]),
            round(value_action_agreement, 3),
            f"{env} tie tseed action agreement",
        )

        if matched_sweeps != 6 or sto_sweeps != 6:
            errors.append(f"{env}: tie-policy transition-seed matched and stochastic TRL sweeps are not 6")
        if full_sweeps != 180:
            errors.append(f"{env}: tie-policy transition-seed full Bellman sweeps are not 180 ({full_sweeps})")
        if sto < 0.95:
            errors.append(f"{env}: tie-policy transition-seed stochastic TRL success below 0.95 ({sto:.3f})")
        if matched > 0.60:
            errors.append(f"{env}: tie-policy transition-seed matched Bellman success above 0.60 ({matched:.3f})")
        if sto - matched < 0.40:
            errors.append(f"{env}: tie-policy transition-seed improvement below 0.40 ({sto - matched:.3f})")
        if abs(sto - full) > 1e-12:
            errors.append(f"{env}: tie-policy transition-seed stochastic TRL does not match full Bellman")
        if value_action_agreement < 0.95:
            errors.append(
                f"{env}: tie-policy transition-seed action agreement below 0.95 ({value_action_agreement:.3f})"
            )

        verified.append(
            {
                "env": env,
                "transition_seeds": transition_seeds,
                "matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "full": fmt(full),
                "improvement": fmt(sto - matched),
                "transition_top1_min": fmt(transition_top1),
                "value_action_agreement_min": fmt(value_action_agreement),
            }
        )
    return verified


def verify_antmaze_tie_policy_head(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    if not ANTMAZE_TIE_POLICY_HEAD_TABLE.exists():
        errors.append(
            f"AntMaze tie-policy head: missing table "
            f"{ANTMAZE_TIE_POLICY_HEAD_TABLE.relative_to(ROOT)}"
        )
        return verified

    table_rows = read_rows(ANTMAZE_TIE_POLICY_HEAD_TABLE)
    by_env = {row["env"]: row for row in table_rows}
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for spec in ANTMAZE_TIE_POLICY_HEAD_SPECS:
        env = str(spec["env"])
        source = Path(spec["source"])
        if not source.exists():
            errors.append(f"{env}: missing AntMaze tie-policy source {source.relative_to(ROOT)}")
            continue
        if env not in by_env:
            errors.append(f"{env}: missing generated AntMaze tie-policy table row")
            continue
        table = by_env[env]
        source_rows = read_rows(source)

        raw_env = distinct_values(source_rows, "env")
        raw_bc_steps = distinct_values(source_rows, "bc_steps")
        raw_eval_seed = distinct_values(source_rows, "seed")
        raw_episodes = distinct_values(source_rows, "episodes_per_task")
        raw_tasks = distinct_values(source_rows, "task_ids")
        raw_backend = distinct_values(source_rows, "policy_eval_backend")
        raw_goal_mode = distinct_values(source_rows, "goal_candidate_mode")
        raw_goal_k = distinct_values(source_rows, "goal_candidates_per_state")
        raw_transition_model = distinct_values(source_rows, "transition_model")
        raw_transition_source = distinct_values(source_rows, "transition_target_source")
        raw_value_model = distinct_values(source_rows, "value_model")
        raw_value_steps = distinct_values(source_rows, "value_steps")
        if raw_env != {env}:
            errors.append(f"{env}: raw AntMaze tie-policy env {sorted(raw_env)} does not match")
        if raw_bc_steps != {str(spec["bc_steps"])}:
            errors.append(f"{env}: raw AntMaze tie-policy bc_steps {sorted(raw_bc_steps)} does not match")
        if raw_eval_seed != {"0"}:
            errors.append(f"{env}: raw AntMaze tie-policy eval seed {sorted(raw_eval_seed)} does not match 0")
        if raw_episodes != {"20"}:
            errors.append(f"{env}: raw AntMaze tie-policy episodes {sorted(raw_episodes)} do not match 20")
        if raw_tasks != {"4,5"}:
            errors.append(f"{env}: raw AntMaze tie-policy task_ids {sorted(raw_tasks)} do not match 4,5")
        if raw_backend != {"jax"}:
            errors.append(f"{env}: raw AntMaze tie-policy backend {sorted(raw_backend)} does not match jax")
        if raw_goal_mode != {"body_nearest"} or raw_goal_k != {"16"}:
            errors.append(
                f"{env}: raw AntMaze tie-policy goal candidates mode/k "
                f"{sorted(raw_goal_mode)}/{sorted(raw_goal_k)} do not match body_nearest/16"
            )
        if raw_transition_model != {"topology"} or raw_transition_source != {"topology_samples"}:
            errors.append(
                f"{env}: raw AntMaze tie-policy transition metadata "
                f"{sorted(raw_transition_model)}/{sorted(raw_transition_source)} "
                "does not match topology/topology_samples"
            )
        if raw_value_model != {"raw_obs_tie_policy_mlp"} or raw_value_steps != {"2000"}:
            errors.append(
                f"{env}: raw AntMaze tie-policy value metadata "
                f"{sorted(raw_value_model)}/{sorted(raw_value_steps)} "
                "does not match raw_obs_tie_policy_mlp/2000"
            )

        summary = summarize_rows(source_rows)
        if not required.issubset(summary):
            errors.append(f"{env}: AntMaze tie-policy source missing methods {sorted(required - set(summary))}")
            continue
        by_method = {row["method"]: row for row in source_rows}

        matched = summary["bellman_matched"]["success"]
        sto = summary["sto_trl_matched"]["success"]
        full = summary["bellman_full"]["success"]
        matched_sweeps = int(summary["bellman_matched"]["sweeps"])
        sto_sweeps = int(summary["sto_trl_matched"]["sweeps"])
        full_sweeps = int(summary["bellman_full"]["sweeps"])
        sto_row = by_method["sto_trl_matched"]
        value_action_agreement = float(sto_row["value_action_agreement"])

        assert_equal(errors, table["controller"], str(spec["controller"]), f"{env} AntMaze tie controller")
        assert_equal(errors, table["control_head"], "raw-observation tie-policy MLP", f"{env} AntMaze tie head")
        assert_equal(errors, table["eval_setting"], "seed 0, tasks 4-5, 20 episodes/task", f"{env} AntMaze tie eval")
        assert_equal(errors, table["eval_seed"], "0", f"{env} AntMaze tie eval seed")
        assert_equal(errors, table["task_ids"], "4,5", f"{env} AntMaze tie task_ids")
        assert_equal(errors, table["controller_steps"], str(spec["bc_steps"]), f"{env} AntMaze tie bc_steps")
        assert_equal(errors, int(table["matched_sweeps"]), matched_sweeps, f"{env} AntMaze tie matched sweeps")
        assert_equal(errors, sto_sweeps, matched_sweeps, f"{env} AntMaze tie stochastic sweeps")
        assert_equal(errors, int(table["full_sweeps"]), full_sweeps, f"{env} AntMaze tie full sweeps")
        assert_equal(errors, table["value_steps"], "2000", f"{env} AntMaze tie value steps")
        assert_equal(errors, table["source"], str(source.relative_to(ROOT)), f"{env} AntMaze tie source")
        assert_close(errors, float(table["bellman_matched"]), round(matched, 3), f"{env} AntMaze tie matched")
        assert_close(errors, float(table["stochastic_trl"]), round(sto, 3), f"{env} AntMaze tie stochastic")
        assert_close(errors, float(table["bellman_full"]), round(full, 3), f"{env} AntMaze tie full")
        assert_close(errors, float(table["sto_minus_matched"]), round(sto - matched, 3), f"{env} AntMaze tie gain")
        assert_close(
            errors,
            float(table["value_action_agreement"]),
            round(value_action_agreement, 3),
            f"{env} AntMaze tie action agreement",
        )

        if matched_sweeps != 6 or sto_sweeps != 6:
            errors.append(f"{env}: AntMaze tie-policy matched and stochastic TRL sweeps are not 6")
        if full_sweeps != 180:
            errors.append(f"{env}: AntMaze tie-policy full Bellman sweeps are not 180 ({full_sweeps})")
        if sto < float(spec["min_sto"]) - 5e-4:
            errors.append(f"{env}: AntMaze tie-policy stochastic TRL success below {spec['min_sto']:.2f} ({sto:.3f})")
        if matched > float(spec["max_matched"]):
            errors.append(
                f"{env}: AntMaze tie-policy matched Bellman success above "
                f"{spec['max_matched']:.2f} ({matched:.3f})"
            )
        if sto - matched < float(spec["min_improvement"]):
            errors.append(
                f"{env}: AntMaze tie-policy improvement below "
                f"{spec['min_improvement']:.2f} ({sto - matched:.3f})"
            )
        if value_action_agreement < 0.99:
            errors.append(f"{env}: AntMaze tie-policy action agreement below 0.99 ({value_action_agreement:.3f})")

        verified.append(
            {
                "env": env,
                "controller": str(spec["controller"]),
                "matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "full": fmt(full),
                "improvement": fmt(sto - matched),
                "value_action_agreement": fmt(value_action_agreement),
            }
        )
    return verified


def verify_antmaze_rawobs_tie_policy_head(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    if not ANTMAZE_RAWOBS_TIE_POLICY_HEAD_TABLE.exists():
        errors.append(
            f"AntMaze raw-observation transition tie-policy head: missing table "
            f"{ANTMAZE_RAWOBS_TIE_POLICY_HEAD_TABLE.relative_to(ROOT)}"
        )
        return verified

    table_rows = read_rows(ANTMAZE_RAWOBS_TIE_POLICY_HEAD_TABLE)
    by_env = {row["env"]: row for row in table_rows}
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for spec in ANTMAZE_RAWOBS_TIE_POLICY_HEAD_SPECS:
        env = str(spec["env"])
        paths = list(spec["sources"])
        if env not in by_env:
            errors.append(f"{env}: missing generated AntMaze raw-observation tie-policy table row")
            continue
        table = by_env[env]

        summaries: list[dict[str, dict[str, float]]] = []
        transition_seeds: list[str] = []
        transition_oracle_l1: list[float] = []
        transition_oracle_top1: list[float] = []
        value_action_agreement: list[float] = []
        source_labels = [str(path.relative_to(ROOT)) for path in paths]
        for source in paths:
            if not source.exists():
                errors.append(f"{env}: missing AntMaze raw-observation tie-policy source {source.relative_to(ROOT)}")
                continue
            source_rows = read_rows(source)

            raw_env = distinct_values(source_rows, "env")
            raw_bc_steps = distinct_values(source_rows, "bc_steps")
            raw_eval_seed = distinct_values(source_rows, "seed")
            raw_episodes = distinct_values(source_rows, "episodes_per_task")
            raw_tasks = distinct_values(source_rows, "task_ids")
            raw_backend = distinct_values(source_rows, "policy_eval_backend")
            raw_goal_mode = distinct_values(source_rows, "goal_candidate_mode")
            raw_goal_k = distinct_values(source_rows, "goal_candidates_per_state")
            raw_transition_model = distinct_values(source_rows, "transition_model")
            raw_transition_source = distinct_values(source_rows, "transition_target_source")
            raw_transition_steps = distinct_values(source_rows, "transition_steps")
            raw_value_model = distinct_values(source_rows, "value_model")
            raw_value_steps = distinct_values(source_rows, "value_steps")
            if raw_env != {env}:
                errors.append(f"{env}: raw AntMaze raw-observation tie-policy env {sorted(raw_env)} does not match")
            if raw_bc_steps != {str(spec["bc_steps"])}:
                errors.append(
                    f"{env}: raw AntMaze raw-observation tie-policy bc_steps {sorted(raw_bc_steps)} does not match"
                )
            if raw_eval_seed != {"0"}:
                errors.append(
                    f"{env}: raw AntMaze raw-observation tie-policy eval seed {sorted(raw_eval_seed)} does not match 0"
                )
            if raw_episodes != {"10"}:
                errors.append(
                    f"{env}: raw AntMaze raw-observation tie-policy episodes {sorted(raw_episodes)} do not match 10"
                )
            if raw_tasks != {"4,5"}:
                errors.append(
                    f"{env}: raw AntMaze raw-observation tie-policy task_ids {sorted(raw_tasks)} do not match 4,5"
                )
            if raw_backend != {"jax"}:
                errors.append(
                    f"{env}: raw AntMaze raw-observation tie-policy backend {sorted(raw_backend)} does not match jax"
                )
            if raw_goal_mode != {"body_nearest"} or raw_goal_k != {"16"}:
                errors.append(
                    f"{env}: raw AntMaze raw-observation tie-policy goal candidates mode/k "
                    f"{sorted(raw_goal_mode)}/{sorted(raw_goal_k)} do not match body_nearest/16"
                )
            if raw_transition_model != {"raw_obs_mlp"} or raw_transition_source != {"dataset_jump_changes"}:
                errors.append(
                    f"{env}: raw AntMaze raw-observation tie-policy transition metadata "
                    f"{sorted(raw_transition_model)}/{sorted(raw_transition_source)} "
                    "does not match raw_obs_mlp/dataset_jump_changes"
                )
            if raw_transition_steps != {"1000"}:
                errors.append(
                    f"{env}: raw AntMaze raw-observation tie-policy transition steps "
                    f"{sorted(raw_transition_steps)} do not match 1000"
                )
            if raw_value_model != {"raw_obs_tie_policy_mlp"} or raw_value_steps != {"2000"}:
                errors.append(
                    f"{env}: raw AntMaze raw-observation tie-policy value metadata "
                    f"{sorted(raw_value_model)}/{sorted(raw_value_steps)} "
                    "does not match raw_obs_tie_policy_mlp/2000"
                )

            summary = summarize_rows(source_rows)
            if not required.issubset(summary):
                errors.append(
                    f"{env}: AntMaze raw-observation tie-policy source missing methods "
                    f"{sorted(required - set(summary))}"
                )
                continue
            by_method = {row["method"]: row for row in source_rows}
            sto_row = by_method["sto_trl_matched"]
            summaries.append(summary)
            transition_seeds.append(str(int(float(sto_row["transition_seed"]))))
            transition_oracle_l1.append(float(sto_row["transition_oracle_l1"]))
            transition_oracle_top1.append(float(sto_row["transition_oracle_top1"]))
            value_action_agreement.append(float(sto_row["value_action_agreement"]))

        if len(summaries) != len(paths):
            continue

        def method_mean(method: str) -> float:
            return mean(summary[method]["success"] for summary in summaries)

        matched = method_mean("bellman_matched")
        sto = method_mean("sto_trl_matched")
        full = method_mean("bellman_full")
        matched_sweeps = int(summaries[0]["bellman_matched"]["sweeps"])
        sto_sweeps = int(summaries[0]["sto_trl_matched"]["sweeps"])
        full_sweeps = int(summaries[0]["bellman_full"]["sweeps"])
        transition_oracle_l1_max = max(transition_oracle_l1)
        transition_oracle_top1_min = min(transition_oracle_top1)
        value_action_agreement_min = min(value_action_agreement)

        assert_equal(errors, table["controller"], str(spec["controller"]), f"{env} rawobs tie controller")
        assert_equal(errors, table["transition_model"], "raw-observation MLP jump-change", f"{env} rawobs tie transition")
        assert_equal(errors, table["control_head"], "raw-observation tie-policy MLP", f"{env} rawobs tie head")
        assert_equal(errors, table["eval_setting"], "seed 0, tasks 4-5, 10 episodes/task", f"{env} rawobs tie eval")
        assert_equal(errors, table["eval_seed"], "0", f"{env} rawobs tie eval seed")
        assert_equal(errors, table["transition_seeds"], ",".join(transition_seeds), f"{env} rawobs tie transition seeds")
        assert_equal(errors, table["task_ids"], "4,5", f"{env} rawobs tie task_ids")
        assert_equal(errors, table["controller_steps"], str(spec["bc_steps"]), f"{env} rawobs tie bc_steps")
        assert_equal(errors, int(table["matched_sweeps"]), matched_sweeps, f"{env} rawobs tie matched sweeps")
        assert_equal(errors, sto_sweeps, matched_sweeps, f"{env} rawobs tie stochastic sweeps")
        assert_equal(errors, int(table["full_sweeps"]), full_sweeps, f"{env} rawobs tie full sweeps")
        assert_equal(errors, table["sources"], ";".join(source_labels), f"{env} rawobs tie sources")
        assert_close(errors, float(table["bellman_matched"]), round(matched, 3), f"{env} rawobs tie matched")
        assert_close(errors, float(table["stochastic_trl"]), round(sto, 3), f"{env} rawobs tie stochastic")
        assert_close(errors, float(table["bellman_full"]), round(full, 3), f"{env} rawobs tie full")
        assert_close(errors, float(table["sto_minus_matched"]), round(sto - matched, 3), f"{env} rawobs tie gain")
        assert_close(
            errors,
            float(table["transition_oracle_l1_max"]),
            round(transition_oracle_l1_max, 3),
            f"{env} rawobs tie transition oracle l1 max",
        )
        assert_close(
            errors,
            float(table["transition_oracle_top1_min"]),
            round(transition_oracle_top1_min, 3),
            f"{env} rawobs tie transition oracle top1 min",
        )
        assert_close(
            errors,
            float(table["value_action_agreement_min"]),
            round(value_action_agreement_min, 3),
            f"{env} rawobs tie action agreement min",
        )

        expected_transition_seeds = set(str(seed) for seed in spec["expected_transition_seeds"])
        if set(transition_seeds) != expected_transition_seeds:
            errors.append(
                f"{env}: AntMaze raw-observation tie-policy transition seeds "
                f"{transition_seeds} do not match {sorted(expected_transition_seeds)}"
            )
        if matched_sweeps != 6 or sto_sweeps != 6:
            errors.append(f"{env}: AntMaze raw-observation tie-policy matched and stochastic TRL sweeps are not 6")
        if full_sweeps != 180:
            errors.append(f"{env}: AntMaze raw-observation tie-policy full Bellman sweeps are not 180 ({full_sweeps})")
        if sto < 0.95 - 5e-4:
            errors.append(f"{env}: AntMaze raw-observation tie-policy stochastic TRL success below 0.95 ({sto:.3f})")
        if matched > 0.35:
            errors.append(f"{env}: AntMaze raw-observation tie-policy matched Bellman success above 0.35 ({matched:.3f})")
        if sto - matched < 0.60:
            errors.append(f"{env}: AntMaze raw-observation tie-policy improvement below 0.60 ({sto - matched:.3f})")
        if abs(sto - full) > 1e-12:
            errors.append(f"{env}: AntMaze raw-observation tie-policy stochastic TRL does not match full Bellman")
        if transition_oracle_top1_min < 0.95 or transition_oracle_l1_max > 0.001:
            errors.append(
                f"{env}: AntMaze raw-observation tie-policy transition fit weak "
                f"(top1_min={transition_oracle_top1_min:.3f}, l1_max={transition_oracle_l1_max:.6f})"
            )
        if value_action_agreement_min < 0.99:
            errors.append(
                f"{env}: AntMaze raw-observation tie-policy action agreement below 0.99 "
                f"({value_action_agreement_min:.3f})"
            )

        verified.append(
            {
                "env": env,
                "controller": str(spec["controller"]),
                "transition_seeds": ",".join(transition_seeds),
                "matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "full": fmt(full),
                "improvement": fmt(sto - matched),
                "transition_oracle_l1_max": fmt(transition_oracle_l1_max),
                "transition_oracle_top1_min": fmt(transition_oracle_top1_min),
                "value_action_agreement_min": fmt(value_action_agreement_min),
            }
        )
    return verified


def verify_antmaze_rawobs_tie_policy_eval_seed(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    if not ANTMAZE_RAWOBS_TIE_POLICY_EVAL_SEED_TABLE.exists():
        errors.append(
            f"AntMaze raw-observation transition tie-policy eval-seed: missing table "
            f"{ANTMAZE_RAWOBS_TIE_POLICY_EVAL_SEED_TABLE.relative_to(ROOT)}"
        )
        return verified

    table_rows = read_rows(ANTMAZE_RAWOBS_TIE_POLICY_EVAL_SEED_TABLE)
    by_env = {row["env"]: row for row in table_rows}
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for spec in ANTMAZE_RAWOBS_TIE_POLICY_EVAL_SEED_SPECS:
        env = str(spec["env"])
        paths = list(spec["sources"])
        if env not in by_env:
            errors.append(f"{env}: missing generated AntMaze raw-observation eval-seed table row")
            continue
        table = by_env[env]
        rows: list[dict[str, str]] = []
        source_labels = [str(path.relative_to(ROOT)) for path in paths]
        for source in paths:
            if not source.exists():
                errors.append(f"{env}: missing AntMaze raw-observation eval-seed source {source.relative_to(ROOT)}")
                continue
            rows.extend(read_rows(source))
        if not rows:
            continue

        raw_env = distinct_values(rows, "env")
        raw_bc_steps = distinct_values(rows, "bc_steps")
        raw_eval_seed = distinct_values(rows, "seed")
        raw_episodes = distinct_values(rows, "episodes_per_task")
        raw_tasks = distinct_values(rows, "task_ids")
        raw_backend = distinct_values(rows, "policy_eval_backend")
        raw_goal_mode = distinct_values(rows, "goal_candidate_mode")
        raw_goal_k = distinct_values(rows, "goal_candidates_per_state")
        raw_transition_seed = distinct_values(rows, "transition_seed")
        raw_transition_model = distinct_values(rows, "transition_model")
        raw_transition_source = distinct_values(rows, "transition_target_source")
        raw_transition_steps = distinct_values(rows, "transition_steps")
        raw_value_model = distinct_values(rows, "value_model")
        raw_value_steps = distinct_values(rows, "value_steps")
        if raw_env != {env}:
            errors.append(f"{env}: raw AntMaze raw-observation eval-seed env {sorted(raw_env)} does not match")
        if raw_bc_steps != {str(spec["bc_steps"])}:
            errors.append(f"{env}: raw AntMaze raw-observation eval-seed bc_steps {sorted(raw_bc_steps)} does not match")
        if raw_eval_seed != {"0", "1", "2"}:
            errors.append(f"{env}: raw AntMaze raw-observation eval seeds {sorted(raw_eval_seed)} do not match 0,1,2")
        if raw_episodes != {"10"}:
            errors.append(f"{env}: raw AntMaze raw-observation eval-seed episodes {sorted(raw_episodes)} do not match 10")
        if raw_tasks != {"4,5"}:
            errors.append(f"{env}: raw AntMaze raw-observation eval-seed task_ids {sorted(raw_tasks)} do not match 4,5")
        if raw_backend != {"jax"}:
            errors.append(f"{env}: raw AntMaze raw-observation eval-seed backend {sorted(raw_backend)} does not match jax")
        if raw_goal_mode != {"body_nearest"} or raw_goal_k != {"16"}:
            errors.append(
                f"{env}: raw AntMaze raw-observation eval-seed goal candidates mode/k "
                f"{sorted(raw_goal_mode)}/{sorted(raw_goal_k)} do not match body_nearest/16"
            )
        if raw_transition_seed != {"0"}:
            errors.append(f"{env}: raw AntMaze raw-observation eval-seed transition seed {sorted(raw_transition_seed)} does not match 0")
        if raw_transition_model != {"raw_obs_mlp"} or raw_transition_source != {"dataset_jump_changes"}:
            errors.append(
                f"{env}: raw AntMaze raw-observation eval-seed transition metadata "
                f"{sorted(raw_transition_model)}/{sorted(raw_transition_source)} "
                "does not match raw_obs_mlp/dataset_jump_changes"
            )
        if raw_transition_steps != {"1000"}:
            errors.append(f"{env}: raw AntMaze raw-observation eval-seed transition steps {sorted(raw_transition_steps)} do not match 1000")
        if raw_value_model != {"raw_obs_tie_policy_mlp"} or raw_value_steps != {"2000"}:
            errors.append(
                f"{env}: raw AntMaze raw-observation eval-seed value metadata "
                f"{sorted(raw_value_model)}/{sorted(raw_value_steps)} does not match raw_obs_tie_policy_mlp/2000"
            )

        by_method: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_method.setdefault(row["method"], []).append(row)
        if not required.issubset(by_method):
            errors.append(f"{env}: AntMaze raw-observation eval-seed source missing methods {sorted(required - set(by_method))}")
            continue

        def mean_method(method: str) -> float:
            return mean(float(row["overall_success"]) for row in by_method[method])

        matched = mean_method("bellman_matched")
        sto = mean_method("sto_trl_matched")
        full = mean_method("bellman_full")
        first = by_method["sto_trl_matched"][0]
        eval_seeds = ",".join(
            str(int(float(seed))) for seed in sorted(raw_eval_seed, key=lambda x: int(float(x)))
        )
        matched_sweeps = int(float(by_method["bellman_matched"][0]["iters"]))
        sto_sweeps = int(float(by_method["sto_trl_matched"][0]["iters"]))
        full_sweeps = int(float(by_method["bellman_full"][0]["iters"]))
        transition_oracle_top1 = float(first["transition_oracle_top1"])
        value_action_agreement_min = min(float(row["value_action_agreement"]) for row in by_method["sto_trl_matched"])

        assert_equal(errors, table["controller"], str(spec["controller"]), f"{env} rawobs eval controller")
        assert_equal(errors, table["transition_model"], "raw-observation MLP jump-change", f"{env} rawobs eval transition")
        assert_equal(errors, table["control_head"], "raw-observation tie-policy MLP", f"{env} rawobs eval head")
        assert_equal(errors, table["eval_setting"], "3 eval seeds, tasks 4-5, 10 episodes/task", f"{env} rawobs eval setting")
        assert_equal(errors, table["eval_seeds"], eval_seeds, f"{env} rawobs eval seeds")
        assert_equal(errors, table["transition_seed"], "0", f"{env} rawobs eval transition seed")
        assert_equal(errors, table["task_ids"], "4,5", f"{env} rawobs eval task ids")
        assert_equal(errors, table["controller_steps"], str(spec["bc_steps"]), f"{env} rawobs eval bc_steps")
        assert_equal(errors, int(table["matched_sweeps"]), matched_sweeps, f"{env} rawobs eval matched sweeps")
        assert_equal(errors, sto_sweeps, matched_sweeps, f"{env} rawobs eval stochastic sweeps")
        assert_equal(errors, int(table["full_sweeps"]), full_sweeps, f"{env} rawobs eval full sweeps")
        assert_equal(errors, table["sources"], ";".join(source_labels), f"{env} rawobs eval sources")
        assert_close(errors, float(table["bellman_matched"]), round(matched, 3), f"{env} rawobs eval matched")
        assert_close(errors, float(table["stochastic_trl"]), round(sto, 3), f"{env} rawobs eval stochastic")
        assert_close(errors, float(table["bellman_full"]), round(full, 3), f"{env} rawobs eval full")
        assert_close(errors, float(table["sto_minus_matched"]), round(sto - matched, 3), f"{env} rawobs eval gain")
        assert_close(errors, float(table["transition_oracle_top1"]), round(transition_oracle_top1, 3), f"{env} rawobs eval transition top1")
        assert_close(
            errors,
            float(table["value_action_agreement_min"]),
            round(value_action_agreement_min, 3),
            f"{env} rawobs eval action agreement min",
        )

        if matched_sweeps != 6 or sto_sweeps != 6:
            errors.append(f"{env}: AntMaze raw-observation eval-seed matched and stochastic sweeps are not 6")
        if full_sweeps != 180:
            errors.append(f"{env}: AntMaze raw-observation eval-seed full Bellman sweeps are not 180 ({full_sweeps})")
        if sto < float(spec["min_sto"]) - 5e-4:
            errors.append(f"{env}: AntMaze raw-observation eval-seed stochastic TRL below {spec['min_sto']:.2f} ({sto:.3f})")
        if matched > 0.30:
            errors.append(f"{env}: AntMaze raw-observation eval-seed matched Bellman above 0.30 ({matched:.3f})")
        if sto - matched < 0.60:
            errors.append(f"{env}: AntMaze raw-observation eval-seed improvement below 0.60 ({sto - matched:.3f})")
        if abs(sto - full) > 1e-12:
            errors.append(f"{env}: AntMaze raw-observation eval-seed stochastic TRL does not match full Bellman")
        if transition_oracle_top1 < 0.99 or value_action_agreement_min < 0.99:
            errors.append(
                f"{env}: AntMaze raw-observation eval-seed fit weak "
                f"(transition_top1={transition_oracle_top1:.3f}, action_agree_min={value_action_agreement_min:.3f})"
            )

        verified.append(
            {
                "env": env,
                "controller": str(spec["controller"]),
                "eval_seeds": eval_seeds,
                "matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "full": fmt(full),
                "improvement": fmt(sto - matched),
                "transition_oracle_top1": fmt(transition_oracle_top1),
                "value_action_agreement_min": fmt(value_action_agreement_min),
            }
        )
    return verified


def verify_antmaze_rawobs_prev_policy_eval_seed(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    if not ANTMAZE_RAWOBS_PREV_POLICY_EVAL_SEED_TABLE.exists():
        errors.append(
            f"AntMaze raw-observation transition previous-action policy eval-seed: missing table "
            f"{ANTMAZE_RAWOBS_PREV_POLICY_EVAL_SEED_TABLE.relative_to(ROOT)}"
        )
        return verified

    table_rows = read_rows(ANTMAZE_RAWOBS_PREV_POLICY_EVAL_SEED_TABLE)
    by_env = {row["env"]: row for row in table_rows}
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for spec in ANTMAZE_RAWOBS_PREV_POLICY_EVAL_SEED_SPECS:
        env = str(spec["env"])
        paths = list(spec["sources"])
        if env not in by_env:
            errors.append(f"{env}: missing generated AntMaze previous-action eval-seed table row")
            continue
        table = by_env[env]
        rows: list[dict[str, str]] = []
        source_labels = [str(path.relative_to(ROOT)) for path in paths]
        for source in paths:
            if not source.exists():
                errors.append(f"{env}: missing AntMaze previous-action eval-seed source {source.relative_to(ROOT)}")
                continue
            rows.extend(read_rows(source))
        if not rows:
            continue

        raw_env = distinct_values(rows, "env")
        raw_bc_steps = distinct_values(rows, "bc_steps")
        raw_eval_seed = distinct_values(rows, "seed")
        raw_episodes = distinct_values(rows, "episodes_per_task")
        raw_tasks = distinct_values(rows, "task_ids")
        raw_backend = distinct_values(rows, "policy_eval_backend")
        raw_goal_mode = distinct_values(rows, "goal_candidate_mode")
        raw_goal_k = distinct_values(rows, "goal_candidates_per_state")
        raw_transition_seed = distinct_values(rows, "transition_seed")
        raw_transition_model = distinct_values(rows, "transition_model")
        raw_transition_source = distinct_values(rows, "transition_target_source")
        raw_transition_steps = distinct_values(rows, "transition_steps")
        raw_value_model = distinct_values(rows, "value_model")
        raw_value_steps = distinct_values(rows, "value_steps")
        if raw_env != {env}:
            errors.append(f"{env}: raw AntMaze previous-action eval-seed env {sorted(raw_env)} does not match")
        if raw_bc_steps != {str(spec["bc_steps"])}:
            errors.append(f"{env}: raw AntMaze previous-action eval-seed bc_steps {sorted(raw_bc_steps)} does not match")
        if raw_eval_seed != {"0", "1", "2"}:
            errors.append(f"{env}: raw AntMaze previous-action eval seeds {sorted(raw_eval_seed)} do not match 0,1,2")
        if raw_episodes != {str(spec["episodes"])}:
            errors.append(
                f"{env}: raw AntMaze previous-action eval-seed episodes "
                f"{sorted(raw_episodes)} do not match {spec['episodes']}"
            )
        if raw_tasks != {"4,5"}:
            errors.append(f"{env}: raw AntMaze previous-action eval-seed task_ids {sorted(raw_tasks)} do not match 4,5")
        if raw_backend != {"jax"}:
            errors.append(f"{env}: raw AntMaze previous-action eval-seed backend {sorted(raw_backend)} does not match jax")
        if raw_goal_mode != {"body_nearest"} or raw_goal_k != {"16"}:
            errors.append(
                f"{env}: raw AntMaze previous-action eval-seed goal candidates mode/k "
                f"{sorted(raw_goal_mode)}/{sorted(raw_goal_k)} do not match body_nearest/16"
            )
        if raw_transition_seed != {"0"}:
            errors.append(
                f"{env}: raw AntMaze previous-action eval-seed transition seed "
                f"{sorted(raw_transition_seed)} does not match 0"
            )
        if raw_transition_model != {"raw_obs_mlp"} or raw_transition_source != {"dataset_jump_changes"}:
            errors.append(
                f"{env}: raw AntMaze previous-action eval-seed transition metadata "
                f"{sorted(raw_transition_model)}/{sorted(raw_transition_source)} "
                "does not match raw_obs_mlp/dataset_jump_changes"
            )
        if raw_transition_steps != {"2000"}:
            errors.append(
                f"{env}: raw AntMaze previous-action eval-seed transition steps "
                f"{sorted(raw_transition_steps)} do not match 2000"
            )
        if raw_value_model != {"raw_obs_prev_policy_mlp"} or raw_value_steps != {"2000"}:
            errors.append(
                f"{env}: raw AntMaze previous-action eval-seed value metadata "
                f"{sorted(raw_value_model)}/{sorted(raw_value_steps)} "
                "does not match raw_obs_prev_policy_mlp/2000"
            )

        by_method: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_method.setdefault(row["method"], []).append(row)
        if not required.issubset(by_method):
            errors.append(f"{env}: previous-action eval-seed source missing methods {sorted(required - set(by_method))}")
            continue

        def mean_method(method: str) -> float:
            return mean(float(row["overall_success"]) for row in by_method[method])

        matched = mean_method("bellman_matched")
        sto = mean_method("sto_trl_matched")
        full = mean_method("bellman_full")
        first = by_method["sto_trl_matched"][0]
        eval_seeds = ",".join(str(int(float(seed))) for seed in sorted(raw_eval_seed, key=lambda x: int(float(x))))
        matched_sweeps = int(float(by_method["bellman_matched"][0]["iters"]))
        sto_sweeps = int(float(by_method["sto_trl_matched"][0]["iters"]))
        full_sweeps = int(float(by_method["bellman_full"][0]["iters"]))
        transition_oracle_top1 = float(first["transition_oracle_top1"])
        value_action_agreement_min = min(float(row["value_action_agreement"]) for row in by_method["sto_trl_matched"])

        assert_equal(errors, table["controller"], str(spec["controller"]), f"{env} previous-action controller")
        assert_equal(errors, table["transition_model"], "raw-observation MLP jump-change", f"{env} previous-action transition")
        assert_equal(errors, table["control_head"], "previous-action policy MLP", f"{env} previous-action head")
        assert_equal(
            errors,
            table["eval_setting"],
            f"3 eval seeds, tasks 4-5, {spec['episodes']} episodes/task",
            f"{env} previous-action setting",
        )
        assert_equal(errors, table["eval_seeds"], eval_seeds, f"{env} previous-action eval seeds")
        assert_equal(errors, table["transition_seed"], "0", f"{env} previous-action transition seed")
        assert_equal(errors, table["task_ids"], "4,5", f"{env} previous-action task ids")
        assert_equal(errors, table["controller_steps"], str(spec["bc_steps"]), f"{env} previous-action bc_steps")
        assert_equal(errors, int(table["matched_sweeps"]), matched_sweeps, f"{env} previous-action matched sweeps")
        assert_equal(errors, sto_sweeps, matched_sweeps, f"{env} previous-action stochastic sweeps")
        assert_equal(errors, int(table["full_sweeps"]), full_sweeps, f"{env} previous-action full sweeps")
        assert_equal(errors, table["sources"], ";".join(source_labels), f"{env} previous-action sources")
        assert_close(errors, float(table["bellman_matched"]), round(matched, 3), f"{env} previous-action matched")
        assert_close(errors, float(table["stochastic_trl"]), round(sto, 3), f"{env} previous-action stochastic")
        assert_close(errors, float(table["bellman_full"]), round(full, 3), f"{env} previous-action full")
        assert_close(errors, float(table["sto_minus_matched"]), round(sto - matched, 3), f"{env} previous-action gain")
        assert_close(
            errors,
            float(table["transition_oracle_top1"]),
            round(transition_oracle_top1, 3),
            f"{env} previous-action transition top1",
        )
        assert_close(
            errors,
            float(table["value_action_agreement_min"]),
            round(value_action_agreement_min, 3),
            f"{env} previous-action action agreement min",
        )

        if matched_sweeps != 6 or sto_sweeps != 6:
            errors.append(f"{env}: previous-action eval-seed matched and stochastic sweeps are not 6")
        if full_sweeps != 180:
            errors.append(f"{env}: previous-action eval-seed full Bellman sweeps are not 180 ({full_sweeps})")
        if sto < float(spec["min_sto"]):
            errors.append(f"{env}: previous-action eval-seed stochastic TRL below {spec['min_sto']:.2f} ({sto:.3f})")
        if matched > 0.45:
            errors.append(f"{env}: previous-action eval-seed matched Bellman above 0.45 ({matched:.3f})")
        if sto - matched < 0.45:
            errors.append(f"{env}: previous-action eval-seed improvement below 0.45 ({sto - matched:.3f})")
        if abs(sto - full) > 0.02:
            errors.append(
                f"{env}: previous-action eval-seed stochastic TRL/full Bellman gap exceeds 0.02 "
                f"({sto:.3f} vs {full:.3f})"
            )
        if transition_oracle_top1 < 0.95 or value_action_agreement_min < 0.99:
            errors.append(
                f"{env}: previous-action eval-seed fit weak "
                f"(transition_top1={transition_oracle_top1:.3f}, action_agree_min={value_action_agreement_min:.3f})"
            )

        verified.append(
            {
                "env": env,
                "controller": str(spec["controller"]),
                "eval_seeds": eval_seeds,
                "matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "full": fmt(full),
                "improvement": fmt(sto - matched),
                "transition_oracle_top1": fmt(transition_oracle_top1),
                "value_action_agreement_min": fmt(value_action_agreement_min),
            }
        )
    return verified


def verify_antmaze_learned_transition(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    if not ANTMAZE_LEARNED_TRANSITION_TABLE.exists():
        errors.append(
            f"AntMaze learned transition: missing table "
            f"{ANTMAZE_LEARNED_TRANSITION_TABLE.relative_to(ROOT)}"
        )
        return verified

    table_rows = read_rows(ANTMAZE_LEARNED_TRANSITION_TABLE)
    by_key = {(row["env"], row["transition_model"]): row for row in table_rows}
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for spec in ANTMAZE_LEARNED_TRANSITION_SPECS:
        env = str(spec["env"])
        controller = str(spec["controller"])
        transition_model = str(spec["transition_model"])
        expected_bc_steps = str(spec["bc_steps"])
        paths = list(spec["paths"])
        key = (env, transition_model)
        if key not in by_key:
            errors.append(f"{env}: missing AntMaze learned-transition table row for {transition_model}")
            continue
        row = by_key[key]
        summaries: list[dict[str, dict[str, float]]] = []
        oracle_l1: list[float] = []
        oracle_top1: list[float] = []
        transition_seeds: list[str] = []
        source_labels = [str(path.relative_to(ROOT)) for path in paths]
        for path in paths:
            if not path.exists():
                errors.append(f"{env}: missing AntMaze learned-transition source {path.relative_to(ROOT)}")
                continue
            source_rows = read_rows(path)
            raw_bc_steps = distinct_values(source_rows, "bc_steps")
            raw_episodes = distinct_values(source_rows, "episodes_per_task")
            raw_eval_seeds = distinct_values(source_rows, "seed")
            raw_tasks = distinct_values(source_rows, "task_ids")
            raw_transition_model = distinct_values(source_rows, "transition_model")
            raw_transition_source = distinct_values(source_rows, "transition_target_source")
            raw_samples = distinct_values(source_rows, "transition_samples_per_row")
            raw_transition_steps = distinct_values(source_rows, "transition_steps")
            raw_backend = distinct_values(source_rows, "policy_eval_backend")
            raw_goal_mode = distinct_values(source_rows, "goal_candidate_mode")
            raw_goal_k = distinct_values(source_rows, "goal_candidates_per_state")
            if raw_bc_steps != {expected_bc_steps}:
                errors.append(f"{env}: raw bc_steps {sorted(raw_bc_steps)} do not match {expected_bc_steps}")
            if raw_episodes != {"10"}:
                errors.append(f"{env}: raw learned-transition episodes {sorted(raw_episodes)} do not match 10")
            if raw_eval_seeds != {"0"}:
                errors.append(f"{env}: raw learned-transition eval seeds {sorted(raw_eval_seeds)} do not match 0")
            if raw_tasks != {"4,5"}:
                errors.append(f"{env}: raw learned-transition task_ids {sorted(raw_tasks)} do not match 4,5")
            if raw_transition_model != {str(spec["raw_transition_model"])}:
                errors.append(
                    f"{env}: raw learned-transition model {sorted(raw_transition_model)} "
                    f"does not match {spec['raw_transition_model']}"
                )
            if raw_transition_source != {str(spec["raw_transition_source"])}:
                errors.append(
                    f"{env}: raw learned-transition target source {sorted(raw_transition_source)} "
                    f"does not match {spec['raw_transition_source']}"
                )
            expected_samples = spec.get("samples_per_row")
            if expected_samples is not None and raw_samples != {str(expected_samples)}:
                errors.append(
                    f"{env}: raw learned-transition samples/row {sorted(raw_samples)} "
                    f"does not match {expected_samples}"
                )
            if raw_transition_steps != {str(spec["transition_steps"])}:
                errors.append(
                    f"{env}: raw learned-transition steps {sorted(raw_transition_steps)} "
                    f"do not match {spec['transition_steps']}"
                )
            if raw_backend != {"jax"}:
                errors.append(f"{env}: raw learned-transition backend {sorted(raw_backend)} does not match jax")
            if raw_goal_mode != {"body_nearest"} or raw_goal_k != {"16"}:
                errors.append(
                    f"{env}: raw learned-transition goal candidates mode/k "
                    f"{sorted(raw_goal_mode)}/{sorted(raw_goal_k)} do not match body_nearest/16"
                )

            summary = summarize_rows(source_rows)
            if not required.issubset(summary):
                errors.append(f"{env}: learned-transition source missing methods {sorted(required - set(summary))}")
                continue
            by_method = {source_row["method"]: source_row for source_row in source_rows}
            summaries.append(summary)
            transition_seeds.append(str(int(float(by_method["sto_trl_matched"]["transition_seed"]))))
            oracle_l1.append(float(by_method["sto_trl_matched"]["transition_oracle_l1"]))
            oracle_top1.append(float(by_method["sto_trl_matched"]["transition_oracle_top1"]))

        if len(summaries) != len(paths):
            continue

        def method_mean(method: str) -> float:
            return mean(summary[method]["success"] for summary in summaries)

        matched = method_mean("bellman_matched")
        sto = method_mean("sto_trl_matched")
        full = method_mean("bellman_full")
        matched_sweeps = int(summaries[0]["bellman_matched"]["sweeps"])
        sto_sweeps = int(summaries[0]["sto_trl_matched"]["sweeps"])
        full_sweeps = int(summaries[0]["bellman_full"]["sweeps"])
        oracle_l1_range = f"{fmt(min(oracle_l1))}-{fmt(max(oracle_l1))}"
        oracle_top1_range = f"{fmt(min(oracle_top1))}-{fmt(max(oracle_top1))}"

        assert_equal(errors, row["controller"], controller, f"{env} LT controller")
        assert_equal(errors, row["transition_model"], transition_model, f"{env} LT model")
        assert_equal(errors, row["eval_setting"], "seed 0, tasks 4-5, 10 episodes/task", f"{env} LT eval_setting")
        assert_equal(errors, row["transition_seeds"], ",".join(transition_seeds), f"{env} LT transition seeds")
        assert_equal(errors, row["sources"], ";".join(source_labels), f"{env} LT sources")
        assert_equal(errors, int(row["matched_sweeps"]), matched_sweeps, f"{env} LT matched_sweeps")
        assert_equal(errors, sto_sweeps, matched_sweeps, f"{env} LT stochastic sweeps")
        assert_equal(errors, int(row["full_sweeps"]), full_sweeps, f"{env} LT full_sweeps")
        assert_close(errors, float(row["bellman_matched"]), round(matched, 3), f"{env} LT matched")
        assert_close(errors, float(row["stochastic_trl"]), round(sto, 3), f"{env} LT stochastic")
        assert_close(errors, float(row["bellman_full"]), round(full, 3), f"{env} LT full")
        assert_close(errors, float(row["sto_minus_matched"]), round(sto - matched, 3), f"{env} LT improvement")
        assert_equal(errors, row["oracle_l1_range"], oracle_l1_range, f"{env} LT oracle_l1_range")
        assert_equal(errors, row["oracle_top1_range"], oracle_top1_range, f"{env} LT oracle_top1_range")

        expected_transition_seeds = set(str(seed) for seed in spec["expected_transition_seeds"])
        if set(transition_seeds) != expected_transition_seeds:
            errors.append(
                f"{env}: learned-transition seeds {transition_seeds} "
                f"do not match {sorted(expected_transition_seeds)}"
            )
        if matched_sweeps != 6 or sto_sweeps != 6:
            errors.append(f"{env}: learned-transition matched and stochastic sweeps are not 6")
        if full_sweeps != 180:
            errors.append(f"{env}: learned-transition full Bellman sweeps are not 180 ({full_sweeps})")
        min_sto = float(spec["min_sto"])
        max_matched = float(spec["max_matched"])
        min_improvement = float(spec["min_improvement"])
        if sto < min_sto - 5e-4:
            errors.append(f"{env}: learned-transition stochastic TRL success below {min_sto:.2f} ({sto:.3f})")
        if matched > max_matched:
            errors.append(
                f"{env}: learned-transition matched Bellman success above "
                f"{max_matched:.2f} ({matched:.3f})"
            )
        if sto - matched < min_improvement:
            errors.append(
                f"{env}: learned-transition improvement below "
                f"{min_improvement:.2f} ({sto - matched:.3f})"
            )
        if abs(sto - full) > 1e-12:
            errors.append(f"{env}: learned-transition stochastic TRL does not match full Bellman")

        verified.append(
            {
                "env": env,
                "controller": controller,
                "transition_model": transition_model,
                "transition_seeds": ",".join(transition_seeds),
                "matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "full": fmt(full),
                "improvement": fmt(sto - matched),
                "oracle_l1_range": oracle_l1_range,
                "oracle_top1_range": oracle_top1_range,
            }
        )
    return verified


def paired_stats(path: Path) -> dict[str, float | str]:
    rows = read_rows(path)
    by_method_seed = {
        (row["method"], int(float(row["seed"]))): float(row["overall_success"])
        for row in rows
    }
    seeds = sorted(
        seed
        for method, seed in by_method_seed
        if method == "sto_trl_matched" and ("bellman_matched", seed) in by_method_seed
    )
    diffs = [by_method_seed[("sto_trl_matched", seed)] - by_method_seed[("bellman_matched", seed)] for seed in seeds]
    full_diffs = [by_method_seed[("bellman_full", seed)] - by_method_seed[("sto_trl_matched", seed)] for seed in seeds]
    n = len(diffs)
    mu = mean(diffs)
    sd = sample_stdev(diffs)
    sem = sd / math.sqrt(n)
    half_width = t_critical_95(n) * sem
    return {
        "n": n,
        "mean_diff": mu,
        "ci_low": mu - half_width,
        "ci_high": mu + half_width,
        "min_diff": min(diffs),
        "max_diff": max(diffs),
        "max_full_minus_sto": max(abs(x) for x in full_diffs),
    }


def verify_paired_stats(errors: list[str]) -> list[dict[str, str]]:
    specs = [
        ("pointmaze-teleport-navigate-v0", RESULTS / "pointmaze_topology_dataset_5seed_ep50.csv"),
        ("pointmaze-teleport-stitch-v0", RESULTS / "pointmaze_topology_stitch_5seed_ep50.csv"),
    ]
    verified: list[dict[str, str]] = []
    for env, path in specs:
        stats = paired_stats(path)
        if stats["n"] != 5:
            errors.append(f"{env}: expected 5 paired seeds, got {stats['n']}")
        if stats["mean_diff"] < 0.50:
            errors.append(f"{env}: paired improvement below 0.50 ({stats['mean_diff']:.3f})")
        if stats["ci_low"] <= 0.0:
            errors.append(f"{env}: paired 95% CI does not exclude zero ({stats['ci_low']:.3f})")
        if stats["min_diff"] <= 0.0:
            errors.append(f"{env}: at least one seed has nonpositive improvement")
        if stats["max_full_minus_sto"] > 1e-12:
            errors.append(f"{env}: full Bellman differs from stochastic TRL in paired source")
        verified.append(
            {
                "env": env,
                "n": str(stats["n"]),
                "mean_diff": fmt(float(stats["mean_diff"])),
                "ci95": f"[{fmt(float(stats['ci_low']))}, {fmt(float(stats['ci_high']))}]",
                "min_seed_diff": fmt(float(stats["min_diff"])),
                "max_seed_diff": fmt(float(stats["max_diff"])),
            }
        )
    return verified


def verify_antmaze_paired_stats(errors: list[str]) -> list[dict[str, str]]:
    specs = [
        (
            "antmaze-teleport-navigate-v0",
            RESULTS / "antmaze_bc_topology_navigate_fullgoal_50k_ep20_seed012_bodyk16_cpu.csv",
        ),
        (
            "antmaze-teleport-stitch-v0",
            RESULTS / "antmaze_bc_topology_stitch_fullgoal_20k_ep20_seed012_bodyk16_cpu.csv",
        ),
    ]
    verified: list[dict[str, str]] = []
    for env, path in specs:
        stats = paired_stats(path)
        if stats["n"] != 3:
            errors.append(f"{env}: expected 3 paired eval seeds, got {stats['n']}")
        if stats["mean_diff"] < 0.50:
            errors.append(f"{env}: paired AntMaze improvement below 0.50 ({stats['mean_diff']:.3f})")
        if stats["ci_low"] <= 0.0:
            errors.append(f"{env}: AntMaze paired 95% CI does not exclude zero ({stats['ci_low']:.3f})")
        if stats["min_diff"] <= 0.50:
            errors.append(f"{env}: minimum AntMaze seed improvement not above 0.50 ({stats['min_diff']:.3f})")
        if stats["max_full_minus_sto"] > 1e-12:
            errors.append(f"{env}: full Bellman differs from stochastic TRL in AntMaze paired source")
        verified.append(
            {
                "env": env,
                "n": str(stats["n"]),
                "mean_diff": fmt(float(stats["mean_diff"])),
                "ci95": f"[{fmt(float(stats['ci_low']))}, {fmt(float(stats['ci_high']))}]",
                "min_seed_diff": fmt(float(stats["min_diff"])),
                "max_seed_diff": fmt(float(stats["max_diff"])),
            }
        )
    return verified


def verify_support_ablation(errors: list[str]) -> list[dict[str, str]]:
    if not SUPPORT_ABLATION.exists():
        errors.append(f"missing support ablation source {SUPPORT_ABLATION.relative_to(ROOT)}")
        return []
    summary = summarize_source(SUPPORT_ABLATION)
    required = {"bellman_matched", "support_trl_matched", "sto_trl_matched", "bellman_full"}
    if not required.issubset(summary):
        errors.append(f"support ablation missing methods {sorted(required - set(summary))}")
        return []

    matched = summary["bellman_matched"]["success"]
    support = summary["support_trl_matched"]["success"]
    sto = summary["sto_trl_matched"]["success"]
    full = summary["bellman_full"]["success"]
    sweeps = int(summary["support_trl_matched"]["sweeps"])

    if sweeps != 6:
        errors.append(f"support ablation: expected 6 support TRL sweeps, got {sweeps}")
    if not (matched < support < sto):
        errors.append(
            "support ablation: expected matched < support TRL < stochastic TRL "
            f"but got {matched:.3f}, {support:.3f}, {sto:.3f}"
        )
    if sto - support < 0.40:
        errors.append(f"support ablation: stochastic TRL gap over support below 0.40 ({sto - support:.3f})")
    if support > 0.60:
        errors.append(f"support ablation: support TRL unexpectedly high ({support:.3f})")
    if abs(sto - full) > 1e-12:
        errors.append("support ablation: stochastic TRL does not match full Bellman")

    return [
        {
            "env": "pointmaze-teleport-stitch-v0",
            "matched": fmt(matched),
            "support_trl": fmt(support),
            "stochastic_trl": fmt(sto),
            "full": fmt(full),
            "sto_minus_support": fmt(sto - support),
        }
    ]


def verify_antmaze_controller_seed_screen(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for env, path, expected_bc_steps in ANTMAZE_BCSEED1_SPECS:
        if not path.exists():
            errors.append(f"{env}: missing controller-seed screen source {path.relative_to(ROOT)}")
            continue

        rows = read_rows(path)
        raw_bc_seed = distinct_values(rows, "bc_seed")
        raw_bc_steps = distinct_values(rows, "bc_steps")
        raw_episodes = distinct_values(rows, "episodes_per_task")
        if raw_bc_seed != {"1"}:
            errors.append(f"{env}: raw bc_seed metadata {sorted(raw_bc_seed)} does not match 1")
        if raw_bc_steps != {expected_bc_steps}:
            errors.append(
                f"{env}: raw bc_steps metadata {sorted(raw_bc_steps)} does not match {expected_bc_steps}"
            )
        if raw_episodes != {"20"}:
            errors.append(f"{env}: raw episodes_per_task metadata {sorted(raw_episodes)} does not match 20")

        summary = summarize_rows(rows)
        if not required.issubset(summary):
            errors.append(f"{env}: controller-seed screen missing methods {sorted(required - set(summary))}")
            continue

        matched = summary["bellman_matched"]["success"]
        sto = summary["sto_trl_matched"]["success"]
        full = summary["bellman_full"]["success"]
        matched_sweeps = int(summary["bellman_matched"]["sweeps"])
        sto_sweeps = int(summary["sto_trl_matched"]["sweeps"])
        full_sweeps = int(summary["bellman_full"]["sweeps"])
        stats = paired_stats(path)

        if stats["n"] != 3:
            errors.append(f"{env}: expected 3 eval seeds, got {stats['n']}")
        if matched_sweeps != 6 or sto_sweeps != 6:
            errors.append(f"{env}: matched and stochastic TRL sweeps are not 6")
        if full_sweeps != 180:
            errors.append(f"{env}: full Bellman sweeps are not 180 ({full_sweeps})")
        if sto < 0.90:
            errors.append(f"{env}: controller-seed stochastic TRL success below 0.90 ({sto:.3f})")
        if matched > 0.40:
            errors.append(f"{env}: controller-seed matched Bellman success above 0.40 ({matched:.3f})")
        if sto - matched < 0.50:
            errors.append(f"{env}: controller-seed improvement below 0.50 ({sto - matched:.3f})")
        if abs(sto - full) > 1e-12:
            errors.append(f"{env}: controller-seed stochastic TRL does not match full Bellman")

        verified.append(
            {
                "env": env,
                "controller_seed": "1",
                "n_eval_seeds": str(stats["n"]),
                "matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "full": fmt(full),
                "improvement": fmt(sto - matched),
            }
        )
    return verified


def verify_antmaze_stitch_controller_seeds(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for controller_seed, path in ANTMAZE_STITCH_CONTROLLER_SEED_SPECS:
        if not path.exists():
            errors.append(f"stitch controller seed {controller_seed}: missing source {path.relative_to(ROOT)}")
            continue
        rows = read_rows(path)
        raw_bc_steps = distinct_values(rows, "bc_steps")
        raw_episodes = distinct_values(rows, "episodes_per_task")
        raw_bc_seed = distinct_values(rows, "bc_seed")
        if raw_bc_steps != {"20000"}:
            errors.append(
                f"stitch controller seed {controller_seed}: raw bc_steps metadata {sorted(raw_bc_steps)} "
                "does not match 20000"
            )
        if raw_episodes != {"20"}:
            errors.append(
                f"stitch controller seed {controller_seed}: raw episodes_per_task metadata "
                f"{sorted(raw_episodes)} does not match 20"
            )
        if raw_bc_seed and raw_bc_seed != {controller_seed}:
            errors.append(
                f"stitch controller seed {controller_seed}: raw bc_seed metadata {sorted(raw_bc_seed)} "
                f"does not match {controller_seed}"
            )

        summary = summarize_rows(rows)
        if not required.issubset(summary):
            errors.append(
                f"stitch controller seed {controller_seed}: missing methods {sorted(required - set(summary))}"
            )
            continue

        matched = summary["bellman_matched"]["success"]
        sto = summary["sto_trl_matched"]["success"]
        full = summary["bellman_full"]["success"]
        matched_sweeps = int(summary["bellman_matched"]["sweeps"])
        sto_sweeps = int(summary["sto_trl_matched"]["sweeps"])
        full_sweeps = int(summary["bellman_full"]["sweeps"])
        full_minus_sto = full - sto
        stats = paired_stats(path)

        if stats["n"] != 3:
            errors.append(f"stitch controller seed {controller_seed}: expected 3 eval seeds, got {stats['n']}")
        if matched_sweeps != 6 or sto_sweeps != 6:
            errors.append(f"stitch controller seed {controller_seed}: matched and stochastic TRL sweeps are not 6")
        if full_sweeps != 180:
            errors.append(f"stitch controller seed {controller_seed}: full Bellman sweeps are not 180 ({full_sweeps})")
        if sto < 0.90:
            errors.append(f"stitch controller seed {controller_seed}: stochastic TRL success below 0.90 ({sto:.3f})")
        if matched > 0.40:
            errors.append(f"stitch controller seed {controller_seed}: matched Bellman success above 0.40 ({matched:.3f})")
        if sto - matched < 0.50:
            errors.append(f"stitch controller seed {controller_seed}: improvement below 0.50 ({sto - matched:.3f})")
        if abs(full_minus_sto) > 0.02:
            errors.append(
                f"stitch controller seed {controller_seed}: full Bellman gap above 0.02 ({full_minus_sto:.3f})"
            )

        verified.append(
            {
                "controller_seed": controller_seed,
                "n_eval_seeds": str(stats["n"]),
                "matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "full": fmt(full),
                "improvement": fmt(sto - matched),
                "full_minus_sto": fmt(full_minus_sto),
            }
        )
    return verified


def verify_antmaze_navigate_controller_seeds(errors: list[str]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for controller_seed, path in ANTMAZE_NAVIGATE_CONTROLLER_SEED_SPECS:
        if not path.exists():
            errors.append(f"navigate controller seed {controller_seed}: missing source {path.relative_to(ROOT)}")
            continue
        rows = read_rows(path)
        raw_bc_steps = distinct_values(rows, "bc_steps")
        raw_episodes = distinct_values(rows, "episodes_per_task")
        raw_bc_seed = distinct_values(rows, "bc_seed")
        if raw_bc_steps != {"50000"}:
            errors.append(
                f"navigate controller seed {controller_seed}: raw bc_steps metadata {sorted(raw_bc_steps)} "
                "does not match 50000"
            )
        if raw_episodes != {"20"}:
            errors.append(
                f"navigate controller seed {controller_seed}: raw episodes_per_task metadata "
                f"{sorted(raw_episodes)} does not match 20"
            )
        if raw_bc_seed and raw_bc_seed != {controller_seed}:
            errors.append(
                f"navigate controller seed {controller_seed}: raw bc_seed metadata {sorted(raw_bc_seed)} "
                f"does not match {controller_seed}"
            )

        summary = summarize_rows(rows)
        if not required.issubset(summary):
            errors.append(
                f"navigate controller seed {controller_seed}: missing methods {sorted(required - set(summary))}"
            )
            continue

        matched = summary["bellman_matched"]["success"]
        sto = summary["sto_trl_matched"]["success"]
        full = summary["bellman_full"]["success"]
        matched_sweeps = int(summary["bellman_matched"]["sweeps"])
        sto_sweeps = int(summary["sto_trl_matched"]["sweeps"])
        full_sweeps = int(summary["bellman_full"]["sweeps"])
        full_minus_sto = full - sto
        stats = paired_stats(path)

        if stats["n"] != 3:
            errors.append(f"navigate controller seed {controller_seed}: expected 3 eval seeds, got {stats['n']}")
        if matched_sweeps != 6 or sto_sweeps != 6:
            errors.append(f"navigate controller seed {controller_seed}: matched and stochastic TRL sweeps are not 6")
        if full_sweeps != 180:
            errors.append(f"navigate controller seed {controller_seed}: full Bellman sweeps are not 180 ({full_sweeps})")
        if sto < 0.90:
            errors.append(f"navigate controller seed {controller_seed}: stochastic TRL success below 0.90 ({sto:.3f})")
        if matched > 0.40:
            errors.append(f"navigate controller seed {controller_seed}: matched Bellman success above 0.40 ({matched:.3f})")
        if sto - matched < 0.50:
            errors.append(f"navigate controller seed {controller_seed}: improvement below 0.50 ({sto - matched:.3f})")
        if abs(full_minus_sto) > 0.02:
            errors.append(
                f"navigate controller seed {controller_seed}: full Bellman gap above 0.02 ({full_minus_sto:.3f})"
            )

        verified.append(
            {
                "controller_seed": controller_seed,
                "n_eval_seeds": str(stats["n"]),
                "matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "full": fmt(full),
                "improvement": fmt(sto - matched),
                "full_minus_sto": fmt(full_minus_sto),
            }
        )
    return verified


def markdown_table(fields: list[str], rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[name] for name in fields) + " |")
    return "\n".join(lines)


def main() -> int:
    errors: list[str] = []
    main_verified = verify_main_table(errors)
    hard_stress_verified = verify_hard_task_stress_table(errors)
    pointmaze_single_verified = verify_pointmaze_single_task(errors)
    pointmaze_lt_verified = verify_pointmaze_learned_transition(errors)
    pointmaze_bc_verified = verify_pointmaze_bc_controller(errors)
    controller_iso_verified = verify_controller_execution_isolation(errors)
    fast_eval_verified = verify_fast_eval_profile(errors)
    antmaze_support_verified = verify_antmaze_support_ablation(errors)
    pointmaze_tie_verified = verify_pointmaze_tie_policy_head(errors)
    pointmaze_prev_verified = verify_pointmaze_prev_policy_head(errors)
    pointmaze_tie_eval_verified = verify_pointmaze_tie_policy_head_eval_seed(errors)
    pointmaze_tie_transition_verified = verify_pointmaze_tie_policy_head_transition_seed(errors)
    antmaze_tie_verified = verify_antmaze_tie_policy_head(errors)
    antmaze_rawobs_tie_verified = verify_antmaze_rawobs_tie_policy_head(errors)
    antmaze_rawobs_eval_verified = verify_antmaze_rawobs_tie_policy_eval_seed(errors)
    antmaze_rawobs_prev_eval_verified = verify_antmaze_rawobs_prev_policy_eval_seed(errors)
    antmaze_lt_verified = verify_antmaze_learned_transition(errors)
    paired_verified = verify_paired_stats(errors)
    antmaze_paired_verified = verify_antmaze_paired_stats(errors)
    support_verified = verify_support_ablation(errors)
    controller_seed_verified = verify_antmaze_controller_seed_screen(errors)
    stitch_controller_seed_verified = verify_antmaze_stitch_controller_seeds(errors)
    navigate_controller_seed_verified = verify_antmaze_navigate_controller_seeds(errors)

    lines = [
        "# Main Claim Verification",
        "",
        "Generated by `scripts/verify_main_claims.py`.",
        "",
        "## Status",
        "",
        "PASS" if not errors else "FAIL",
        "",
        "## Main Hard-Task Checks",
        "",
        markdown_table(["env", "source", "matched", "stochastic_trl", "full", "improvement"], main_verified),
        "",
        "Checks: matched and stochastic TRL use 6 sweeps, full Bellman uses 180 sweeps, stochastic TRL success is at least 0.90, matched Bellman is at most 0.40, improvement is at least 0.50, stochastic TRL exactly matches full Bellman, and AntMaze raw bc_steps metadata matches the headline controller budget.",
        "",
        "## Hard-Task Stress Checks",
        "",
        markdown_table(
            ["env", "task_scope", "matched", "support_trl", "stochastic_trl", "full", "improvement"],
            hard_stress_verified,
        ),
        "",
        "Checks: generated stress table values match their raw sources, stochastic TRL success is at least 0.90, improvement over matched Bellman is at least 0.40, and stochastic TRL matches full Bellman. The PointMaze stitch stress row also checks matched < support TRL < stochastic TRL.",
        "",
        "## Focused PointMaze Single-Task Check",
        "",
        markdown_table(
            [
                "env",
                "task_id",
                "n_seeds",
                "episodes_per_seed",
                "matched",
                "stochastic_trl",
                "full",
                "improvement",
            ],
            pointmaze_single_verified,
        ),
        "",
        "Checks: PointMaze teleport stitch task 5 uses five evaluation seeds and 100 episodes per seed, matched and stochastic TRL use 6 sweeps, full Bellman uses 180 sweeps, stochastic TRL success is at least 0.90, matched Bellman is at most 0.40, improvement is at least 0.50, and stochastic TRL matches full Bellman.",
        "",
        "## PointMaze Learned-Transition Screen Check",
        "",
        markdown_table(
            [
                "env",
                "transition_model",
                "transition_seeds",
                "matched",
                "stochastic_trl",
                "full",
                "improvement",
                "oracle_l1_range",
                "oracle_top1_range",
            ],
            pointmaze_lt_verified,
        ),
        "",
        "Checks: generated learned-transition rows match raw source CSVs and model-specific transition seeds, use collapsed offline cell-change targets, one evaluation seed, 50 episodes per task, 6 matched sweeps, 180 full sweeps, stochastic TRL success at least 0.90, model-specific matched-Bellman and improvement thresholds, and stochastic TRL matches full Bellman.",
        "",
        "## PointMaze Learned-Controller Screen Check",
        "",
        markdown_table(
            [
                "env",
                "controller",
                "eval_seeds",
                "matched",
                "stochastic_trl",
                "full",
                "improvement",
            ],
            pointmaze_bc_verified,
        ),
        "",
        "Checks: generated learned-controller rows match raw source CSVs, use a saved 5k-step full-goal BC controller with body-nearest k16 waypoint goals, cover eval seeds 0, 1, and 2 with 20 episodes per task, use the JAX policy backend, use 6 matched sweeps and 180 full sweeps, reach stochastic TRL success at least 0.99, keep matched Bellman at most 0.35, improve by at least 0.65, and match full Bellman.",
        "",
        "## Controller Execution Isolation Check",
        "",
        markdown_table(
            [
                "execution_path",
                "env",
                "best_success",
                "final_success",
                "source",
            ],
            controller_iso_verified,
        ),
        "",
        "Checks: direct final-goal actor rows match raw OGBench eval CSVs and remain at or below 0.15 success in the short diagnostic, learned waypoint-executor rows match the generated PointMaze learned-controller table and remain at least 0.99 success, and the navigate waypoint/direct gap is at least 0.85.",
        "",
        "## Fast Evaluation Profile Check",
        "",
        markdown_table(
            [
                "screen",
                "role",
                "matched",
                "stochastic_trl",
                "improvement",
                "sto_eval_seconds",
            ],
            fast_eval_verified,
        ),
        "",
        "Checks: generated fast-profile rows match raw profile CSVs, recommended screens use action repeat 1, recommended AntMaze hard-slice stochastic TRL success remains at least 0.90 with at least 0.35-0.40 improvement over matched Bellman, and the action-repeat-2 ablation remains below the repeat-1 two-episode baseline.",
        "",
        "## AntMaze Deterministic-Support Ablation Check",
        "",
        markdown_table(
            ["env", "matched", "support_trl", "stochastic_trl", "full", "sto_minus_support"],
            antmaze_support_verified,
        ),
        "",
        "Checks: generated AntMaze support-ablation rows match raw CSVs, use the JAX policy backend, cover tasks 4 and 5 with 5 episodes per task, use action repeat 1, use 6 matched/support sweeps and 180 full Bellman sweeps, keep support-only TRL equal to matched Bellman in these diagnostics, and keep stochastic TRL at least the task-specific margin above support-only TRL.",
        "",
        "## PointMaze Tie-Policy Head Check",
        "",
        markdown_table(
            [
                "env",
                "matched",
                "stochastic_trl",
                "full",
                "improvement",
                "transition_top1",
                "value_action_agreement",
            ],
            pointmaze_tie_verified,
        ),
        "",
        "Checks: generated tie-policy head row matches the raw source CSV, uses raw-observation MLP cell-change transitions and a raw-observation tie-policy MLP, one eval seed, all tasks, 20 episodes per task, 6 matched sweeps, 180 full sweeps, stochastic TRL success at least 0.95, matched Bellman at most 0.60, improvement at least 0.40, action agreement at least 0.95, and stochastic TRL matches full Bellman.",
        "",
        "## PointMaze Previous-Action Policy Head Check",
        "",
        markdown_table(
            [
                "env",
                "matched",
                "stochastic_trl",
                "full",
                "improvement",
                "transition_top1",
                "value_action_agreement",
            ],
            pointmaze_prev_verified,
        ),
        "",
        "Checks: generated previous-action policy head rows match raw source CSVs, use raw-observation MLP cell-change transitions and a previous-action policy MLP, one eval seed, all tasks, 20 episodes per task, 1000 value steps, 6 matched sweeps, 180 full sweeps, stochastic TRL success at least 0.95, matched Bellman at most 0.60, improvement at least 0.40, action agreement at least 0.99, and stochastic TRL matches full Bellman.",
        "",
        "## PointMaze Tie-Policy Eval-Seed Check",
        "",
        markdown_table(
            [
                "env",
                "eval_seeds",
                "matched",
                "stochastic_trl",
                "full",
                "improvement",
                "transition_top1",
                "value_action_agreement_min",
            ],
            pointmaze_tie_eval_verified,
        ),
        "",
        "Checks: generated PointMaze tie-policy eval-seed rows match the raw source CSVs across eval seeds 0, 1, and 2; use raw-observation MLP cell-change transitions and a raw-observation tie-policy MLP; cover all tasks with 20 episodes per task; use 6 matched sweeps and 180 full sweeps; reach stochastic TRL success at least 0.90; keep matched Bellman at most 0.45; improve by at least 0.45; action agreement remains at least 0.95; and stochastic TRL matches full Bellman.",
        "",
        "## PointMaze Tie-Policy Transition-Seed Check",
        "",
        markdown_table(
            [
                "env",
                "transition_seeds",
                "matched",
                "stochastic_trl",
                "full",
                "improvement",
                "transition_top1_min",
                "value_action_agreement_min",
            ],
            pointmaze_tie_transition_verified,
        ),
        "",
        "Checks: generated PointMaze tie-policy transition-seed rows match raw source CSVs across transition seeds 0, 1, and 2 with eval seed 0 fixed; use raw-observation MLP cell-change transitions and a raw-observation tie-policy MLP; cover all tasks with 20 episodes per task; use 6 matched sweeps and 180 full sweeps; reach stochastic TRL success at least 0.95; improve by at least 0.40; action agreement remains at least 0.95; and stochastic TRL matches full Bellman.",
        "",
        "## AntMaze Tie-Policy Head Check",
        "",
        markdown_table(
            [
                "env",
                "controller",
                "matched",
                "stochastic_trl",
                "full",
                "improvement",
                "value_action_agreement",
            ],
            antmaze_tie_verified,
        ),
        "",
        "Checks: generated AntMaze tie-policy head rows match raw source CSVs, use saved full-goal BC controllers with body-nearest k16 waypoint goals, use the topology transition model and a raw-observation tie-policy MLP, cover tasks 4 and 5 with one eval seed and 20 episodes per task, use the JAX policy backend, use 6 matched sweeps and 180 full sweeps, reach stochastic TRL success at least 0.90 on navigate and 0.95 on stitch, preserve action agreement at least 0.99, and improve over matched Bellman by the configured task-specific margin.",
        "",
        "## AntMaze Raw-Observation Transition Tie-Policy Head Check",
        "",
        markdown_table(
            [
                "env",
                "controller",
                "transition_seeds",
                "matched",
                "stochastic_trl",
                "full",
                "improvement",
                "transition_oracle_l1_max",
                "transition_oracle_top1_min",
                "value_action_agreement_min",
            ],
            antmaze_rawobs_tie_verified,
        ),
        "",
        "Checks: generated AntMaze raw-observation transition plus tie-policy rows match raw source CSVs across transition seeds 0, 1, and 2; use saved full-goal BC controllers with body-nearest k16 waypoint goals; use raw-observation MLP jump-change transitions and a raw-observation tie-policy MLP; cover tasks 4 and 5 with one eval seed and 10 episodes per task; use the JAX policy backend; use 6 matched sweeps and 180 full sweeps; reach stochastic TRL success at least 0.95; keep matched Bellman at most 0.35; improve by at least 0.60; match full Bellman; and pass transition/action-agreement fit thresholds.",
        "",
        "## AntMaze Raw-Observation Transition Tie-Policy Eval-Seed Check",
        "",
        markdown_table(
            [
                "env",
                "controller",
                "eval_seeds",
                "matched",
                "stochastic_trl",
                "full",
                "improvement",
                "transition_oracle_top1",
                "value_action_agreement_min",
            ],
            antmaze_rawobs_eval_verified,
        ),
        "",
        "Checks: generated AntMaze raw-observation transition plus tie-policy eval-seed rows match raw source CSVs across evaluation seeds 0, 1, and 2 with transition seed 0 fixed; use saved full-goal BC controllers with body-nearest k16 waypoint goals; use raw-observation MLP jump-change transitions and a raw-observation tie-policy MLP; cover tasks 4 and 5 with 10 episodes per task; use the JAX policy backend; use 6 matched sweeps and 180 full sweeps; reach task-specific stochastic TRL success thresholds; keep matched Bellman at most 0.30; improve by at least 0.60; match full Bellman; and pass transition/action-agreement fit thresholds.",
        "",
        "## AntMaze Previous-Action Policy-Head Eval-Seed Check",
        "",
        markdown_table(
            [
                "env",
                "controller",
                "eval_seeds",
                "matched",
                "stochastic_trl",
                "full",
                "improvement",
                "transition_oracle_top1",
                "value_action_agreement_min",
            ],
            antmaze_rawobs_prev_eval_verified,
        ),
        "",
        "Checks: generated AntMaze previous-action policy-head eval-seed rows match raw source CSVs across evaluation seeds 0, 1, and 2 with transition seed 0 fixed; use saved full-goal BC controllers with body-nearest k16 waypoint goals; use raw-observation MLP jump-change transitions and a previous-action-conditioned policy MLP; cover tasks 4 and 5 with 10 episodes per task; use the JAX policy backend; use 6 matched sweeps and 180 full sweeps; reach task-specific stochastic TRL success thresholds; keep matched Bellman at most 0.45; improve by at least 0.45; stay within 0.02 of full Bellman; and pass transition/action-agreement fit thresholds.",
        "",
        "## AntMaze Learned-Transition Screen Check",
        "",
        markdown_table(
            [
                "env",
                "controller",
                "transition_model",
                "transition_seeds",
                "matched",
                "stochastic_trl",
                "full",
                "improvement",
                "oracle_l1_range",
                "oracle_top1_range",
            ],
            antmaze_lt_verified,
        ),
        "",
        "Checks: generated learned-transition rows match raw CSVs and model-specific transition seeds, task scope is tasks 4 and 5, evaluation uses 10 episodes per task with the JAX policy backend, 6 matched sweeps, 180 full sweeps, stochastic TRL success at least 0.95, matched Bellman at most 0.40, improvement at least 0.55, and stochastic TRL matches full Bellman.",
        "",
        "## PointMaze Paired Seed Checks",
        "",
        markdown_table(["env", "n", "mean_diff", "ci95", "min_seed_diff", "max_seed_diff"], paired_verified),
        "",
        "Checks: five paired seeds, positive seed-level improvements, 95% CI excludes zero, and full Bellman exactly matches stochastic TRL.",
        "",
        "## AntMaze Paired Eval-Seed Checks",
        "",
        markdown_table(["env", "n", "mean_diff", "ci95", "min_seed_diff", "max_seed_diff"], antmaze_paired_verified),
        "",
        "Checks: three paired evaluation seeds, every seed improves by more than 0.50 over matched Bellman, 95% CI excludes zero, and full Bellman exactly matches stochastic TRL.",
        "",
        "## Deterministic-Support Ablation Check",
        "",
        markdown_table(["env", "matched", "support_trl", "stochastic_trl", "full", "sto_minus_support"], support_verified),
        "",
        "Checks: support TRL uses 6 sweeps, improves over matched Bellman, remains below 0.60 success, stays at least 0.40 below stochastic TRL, and stochastic TRL matches full Bellman.",
        "",
        "## AntMaze Controller-Seed Robustness Check",
        "",
        markdown_table(
            ["env", "controller_seed", "n_eval_seeds", "matched", "stochastic_trl", "full", "improvement"],
            controller_seed_verified,
        ),
        "",
        "Checks: independent AntMaze BC controller seed 1 for navigate and stitch, raw bc_steps metadata matches the per-task controller budget, 20 episodes per task, three evaluation seeds, 6 matched sweeps, 180 full sweeps, stochastic TRL success at least 0.90, matched Bellman at most 0.40, improvement at least 0.50, and stochastic TRL matches full Bellman.",
        "",
        "## AntMaze Stitch Controller-Seed Aggregate Check",
        "",
        markdown_table(
            [
                "controller_seed",
                "n_eval_seeds",
                "matched",
                "stochastic_trl",
                "full",
                "improvement",
                "full_minus_sto",
            ],
            stitch_controller_seed_verified,
        ),
        "",
        "Checks: stitch controller seeds 0, 1, and 2, raw bc_steps metadata equals 20000, 20 episodes per task, three evaluation seeds, 6 matched sweeps, 180 full sweeps, stochastic TRL success at least 0.90, matched Bellman at most 0.40, improvement at least 0.50, and absolute full-Bellman gap at most 0.02.",
        "",
        "## AntMaze Navigate Controller-Seed Aggregate Check",
        "",
        markdown_table(
            [
                "controller_seed",
                "n_eval_seeds",
                "matched",
                "stochastic_trl",
                "full",
                "improvement",
                "full_minus_sto",
            ],
            navigate_controller_seed_verified,
        ),
        "",
        "Checks: navigate controller seeds 0, 1, and 2, raw bc_steps metadata equals 50000, 20 episodes per task, three evaluation seeds, 6 matched sweeps, 180 full sweeps, stochastic TRL success at least 0.90, matched Bellman at most 0.40, improvement at least 0.50, and absolute full-Bellman gap at most 0.02.",
    ]
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
    REPORT.write_text("\n".join(lines) + "\n")

    print(f"wrote {REPORT.relative_to(ROOT)}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("all main claim checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
