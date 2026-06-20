#!/usr/bin/env python3
"""Verify rejected AntMaze speed-screen knobs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
TRL_DIR = ROOT / "external" / "trl"
if str(TRL_DIR) not in sys.path:
    sys.path.insert(0, str(TRL_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from envs.env_utils import make_env_and_datasets
from run_antmaze_bc_topology_planner import load_bc_policy


RESULTS = ROOT / "results"
REPORT = RESULTS / "eval_speed_rejection_verification.md"
JAX_REF = RESULTS / "antmaze_stitch_prev_policy_hard_task45_ep5_seed0_jax_reference.csv"
JAX_FUSED = RESULTS / "antmaze_stitch_prev_policy_hard_task45_ep5_seed0_jax_fused.csv"
CAP_STO = RESULTS / "antmaze_stitch_prev_policy_hard_task45_ep5_seed0_cap800.csv"
CAP_MATCHED = RESULTS / "antmaze_stitch_prev_policy_hard_task45_ep5_seed0_cap800_matched.csv"
POLICY = RESULTS / "policies" / "antmaze_stitch_fullgoal_bc_20k.pkl"


def read_single(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"{path.relative_to(ROOT)} expected one row, found {len(rows)}")
    return rows[0]


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def require(errors: list[str], cond: bool, msg: str) -> None:
    if not cond:
        errors.append(msg)


def check_common(errors: list[str], row: dict[str, str], source: Path, backend: str, max_steps: str) -> None:
    label = str(source.relative_to(ROOT))
    expected = {
        "env": "antmaze-teleport-stitch-v0",
        "method": row["method"],
        "seed": "0",
        "episodes_per_task": "5",
        "task_ids": "4,5",
        "transition_model": "raw_obs_mlp",
        "transition_target_source": "dataset_jump_changes",
        "transition_steps": "2000",
        "value_model": "raw_obs_prev_policy_mlp",
        "value_steps": "2000",
        "bc_steps": "20000",
        "goal_representation": "full",
        "goal_candidate_mode": "body_nearest",
        "goal_candidates_per_state": "16",
        "waypoint_lookahead": "1",
        "eval_action_repeat": "1",
        "policy_eval_backend": backend,
        "max_episode_steps": max_steps,
    }
    for key, value in expected.items():
        require(errors, row.get(key, "") == value, f"{label}: {key}={row.get(key, '')!r}, expected {value!r}")


def action_diff() -> float:
    _env, train, _val = make_env_and_datasets("antmaze-teleport-stitch-v0", dataset_path=None)
    policy = load_bc_policy(POLICY)
    obs = np.asarray(train["observations"][:128], dtype=np.float32)
    goals = np.asarray(train["observations"][128:256], dtype=np.float32)
    diffs = [
        float(np.max(np.abs(policy.action(o, g, "jax") - policy.action(o, g, "jax_fused"))))
        for o, g in zip(obs, goals)
    ]
    return float(np.max(diffs))


def main() -> None:
    errors: list[str] = []
    jax_ref = read_single(JAX_REF)
    jax_fused = read_single(JAX_FUSED)
    cap_sto = read_single(CAP_STO)
    cap_matched = read_single(CAP_MATCHED)

    check_common(errors, jax_ref, JAX_REF, "jax", "")
    check_common(errors, jax_fused, JAX_FUSED, "jax_fused", "")
    check_common(errors, cap_sto, CAP_STO, "jax", "800")
    check_common(errors, cap_matched, CAP_MATCHED, "jax", "800")
    require(errors, jax_ref["method"] == "sto_trl_matched", "JAX reference must be stochastic TRL")
    require(errors, jax_fused["method"] == "sto_trl_matched", "fused row must be stochastic TRL")
    require(errors, cap_sto["method"] == "sto_trl_matched", "cap800 stochastic row must be stochastic TRL")
    require(errors, cap_matched["method"] == "bellman_matched", "cap800 matched row must be matched Bellman")

    max_action_diff = action_diff()
    require(errors, max_action_diff < 1e-5, f"jax_fused single-step action diff too high: {max_action_diff:.3g}")

    ref_success = as_float(jax_ref, "overall_success")
    fused_success = as_float(jax_fused, "overall_success")
    ref_eval = as_float(jax_ref, "eval_seconds")
    fused_eval = as_float(jax_fused, "eval_seconds")
    ref_action_ms = as_float(jax_ref, "profile_action_ms_per_call")
    fused_action_ms = as_float(jax_fused, "profile_action_ms_per_call")
    cap_success = as_float(cap_sto, "overall_success")
    cap_eval = as_float(cap_sto, "eval_seconds")
    cap_matched_success = as_float(cap_matched, "overall_success")

    require(errors, ref_success >= 0.99, f"JAX reference success unexpectedly low: {ref_success:.3f}")
    require(errors, fused_success <= 0.95, f"jax_fused no longer shows rejected success drop: {fused_success:.3f}")
    require(errors, fused_eval > ref_eval, f"jax_fused was not slower in saved artifact: {fused_eval:.2f} <= {ref_eval:.2f}")
    require(
        errors,
        fused_action_ms > ref_action_ms,
        f"jax_fused action call was not slower in saved artifact: {fused_action_ms:.3f} <= {ref_action_ms:.3f}",
    )
    require(errors, cap_success >= 0.99, f"cap800 stochastic success unexpectedly low: {cap_success:.3f}")
    require(errors, cap_eval >= ref_eval, f"cap800 stochastic row unexpectedly faster: {cap_eval:.2f} < {ref_eval:.2f}")
    require(errors, cap_matched_success <= 0.50, f"cap800 matched row unexpectedly strong: {cap_matched_success:.3f}")

    report = [
        "# Evaluation Speed Rejection Verification",
        "",
        "Status: PASS" if not errors else "Status: FAIL",
        "",
        f"- JAX reference: success {ref_success:.3f}, eval {ref_eval:.2f}s, action {ref_action_ms:.3f} ms/call",
        f"- JAX fused: success {fused_success:.3f}, eval {fused_eval:.2f}s, action {fused_action_ms:.3f} ms/call",
        f"- JAX vs fused single-step max action difference: {max_action_diff:.3g}",
        f"- cap800 stochastic: success {cap_success:.3f}, eval {cap_eval:.2f}s",
        f"- cap800 matched Bellman: success {cap_matched_success:.3f}",
        "",
        "## Sources",
        "",
        f"- `{JAX_REF.relative_to(ROOT)}`",
        f"- `{JAX_FUSED.relative_to(ROOT)}`",
        f"- `{CAP_STO.relative_to(ROOT)}`",
        f"- `{CAP_MATCHED.relative_to(ROOT)}`",
        "",
    ]
    if errors:
        report.extend(["## Errors", "", *[f"- {error}" for error in errors], ""])
        REPORT.write_text("\n".join(report))
        raise SystemExit("FAIL: " + "; ".join(errors))

    REPORT.write_text("\n".join(report))
    print(f"PASS: verified rejected speed knobs; wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
