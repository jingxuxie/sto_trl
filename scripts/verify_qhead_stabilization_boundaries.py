#!/usr/bin/env python3
"""Verify PointMaze q-head self-bootstrap stabilization boundary screens."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = RESULTS / "qhead_stabilization_boundary_verification.md"

SPECS = [
    {
        "label": "raw self-bootstrap",
        "path": RESULTS / "pointmaze_qhead_monotone_navigate_all_exact.csv",
        "method": "qhead_sto_trl",
        "max_success": 0.07,
        "rank_ce_weight": "0.05",
    },
    {
        "label": "monotone self-bootstrap",
        "path": RESULTS / "pointmaze_qhead_monotone_navigate_all_exact.csv",
        "method": "qhead_sto_trl_monotone",
        "max_success": 0.07,
        "rank_ce_weight": "0.05",
    },
    {
        "label": "self-buffered target",
        "path": RESULTS / "pointmaze_qhead_self_buffered_navigate_all_exact.csv",
        "method": "qhead_sto_trl_self_buffered",
        "max_success": 0.01,
        "rank_ce_weight": "0.05",
    },
    {
        "label": "high-rank raw self-bootstrap",
        "path": RESULTS / "pointmaze_qhead_rank1_navigate_all_exact.csv",
        "method": "qhead_sto_trl",
        "max_success": 0.01,
        "rank_ce_weight": "1.0",
    },
    {
        "label": "fresh projected iteration",
        "path": RESULTS / "pointmaze_qhead_fresh_iter_navigate_all_exact.csv",
        "method": "qhead_sto_trl_fresh_iter",
        "max_success": 0.01,
        "rank_ce_weight": "0.05",
    },
    {
        "label": "guided rank projection",
        "path": RESULTS / "pointmaze_qhead_guided_rank_navigate_all_exact.csv",
        "method": "qhead_sto_trl_guided_rank",
        "max_success": 0.01,
        "rank_ce_weight": "1.0",
    },
    {
        "label": "guided rank projection high weight",
        "path": RESULTS / "pointmaze_qhead_guided_rank10_navigate_all_exact.csv",
        "method": "qhead_sto_trl_guided_rank",
        "max_success": 0.13,
        "rank_ce_weight": "10.0",
    },
    {
        "label": "sampled target-network projection",
        "path": RESULTS / "pointmaze_qhead_sampled_target_net_nav_exact_s64_c32_k4.csv",
        "method": "qhead_sto_trl_sampled_target_net",
        "max_success": 0.07,
        "rank_ce_weight": "1.0",
        "warmup_steps": "600",
        "steps_per_iter": "350",
        "sampled_target_update": "target_network",
        "sampled_target_next_samples": "64",
        "sampled_target_waypoint_candidates": "32",
        "sampled_target_waypoints": "4",
    },
    {
        "label": "sampled target-network projection heavy fit",
        "path": RESULTS / "pointmaze_qhead_sampled_target_net_nav_exact_s64_c32_k4_heavy.csv",
        "method": "qhead_sto_trl_sampled_target_net",
        "max_success": 0.01,
        "rank_ce_weight": "1.0",
        "warmup_steps": "1000",
        "steps_per_iter": "1000",
        "sampled_target_update": "target_network",
        "sampled_target_next_samples": "64",
        "sampled_target_waypoint_candidates": "32",
        "sampled_target_waypoints": "4",
    },
    {
        "label": "sampled target buffer without final reset",
        "path": RESULTS / "pointmaze_qhead_sampled_buffered_nav_exact_s64_c32_k4_no_final.csv",
        "method": "qhead_sto_trl_sampled_buffered_reset_final",
        "max_success": 0.01,
        "rank_ce_weight": "1.0",
        "warmup_steps": "600",
        "steps_per_iter": "350",
        "buffered_final_steps": "0",
        "sampled_target_update": "buffer",
        "sampled_target_next_samples": "64",
        "sampled_target_waypoint_candidates": "32",
        "sampled_target_waypoints": "4",
    },
    {
        "label": "sampled target replay reset-each-iteration",
        "path": RESULTS / "pointmaze_qhead_sampled_target_replay_nav_exact_s64_c32_k4.csv",
        "method": "qhead_sto_trl_sampled_target_replay",
        "max_success": 0.01,
        "rank_ce_weight": "1.0",
        "warmup_steps": "600",
        "steps_per_iter": "1000",
        "buffered_final_steps": "0",
        "sampled_target_update": "target_replay",
        "sampled_target_next_samples": "64",
        "sampled_target_waypoint_candidates": "32",
        "sampled_target_waypoints": "4",
    },
]

POSITIVE_SPECS = [
    {
        "label": "buffered reset-final q-head navigate",
        "path": RESULTS / "pointmaze_qhead_buffered_reset_final_navigate_all_exact.csv",
        "method": "qhead_sto_trl_buffered_reset_final",
        "env": "pointmaze-teleport-navigate-v0",
        "min_success": 0.99,
        "min_action_agreement": 0.95,
        "rank_ce_weight": "1.0",
        "buffered_final_steps": "5000",
        "target_buffer_reset_final": "True",
    },
    {
        "label": "buffered reset-final q-head stitch",
        "path": RESULTS / "pointmaze_qhead_buffered_reset_final_stitch_all_exact.csv",
        "method": "qhead_sto_trl_buffered_reset_final",
        "env": "pointmaze-teleport-stitch-v0",
        "min_success": 0.99,
        "min_action_agreement": 0.95,
        "rank_ce_weight": "1.0",
        "buffered_final_steps": "5000",
        "target_buffer_reset_final": "True",
    },
    {
        "label": "generated target q-head",
        "path": RESULTS / "pointmaze_qhead_guided_rank_navigate_all_exact.csv",
        "method": "qhead_sto_trl_target_fit",
        "env": "pointmaze-teleport-navigate-v0",
        "min_success": 0.99,
        "min_action_agreement": 0.95,
    },
    {
        "label": "table stochastic TRL",
        "path": RESULTS / "pointmaze_qhead_guided_rank10_navigate_all_exact.csv",
        "method": "table_sto_trl_matched",
        "env": "pointmaze-teleport-navigate-v0",
        "min_success": 0.99,
        "min_action_agreement": 0.98,
    },
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def find_row(path: Path, method: str) -> dict[str, str]:
    rows = [row for row in read_rows(path) if row.get("method") == method]
    if len(rows) != 1:
        raise ValueError(f"{path.relative_to(ROOT)} method={method}: expected one row, found {len(rows)}")
    return rows[0]


def check_common(errors: list[str], label: str, row: dict[str, str], env: str) -> None:
    expected = {
        "env": env,
        "eval_mode": "model",
        "model_rollout_mode": "exact",
        "feature_mode": "raw_obs_onehot",
        "transition_model": "raw_obs_mlp",
        "transition_target_source": "dataset_cell_changes",
        "transition_steps": "2000",
        "iters": "6",
        "task_ids": "all",
    }
    for key, value in expected.items():
        if row.get(key) != value:
            errors.append(f"{label}: expected {key}={value}, got {row.get(key)}")


def main() -> None:
    errors: list[str] = []
    report = ["# Q-Head Stabilization Boundary Verification", "", "Status: PASS", ""]

    for spec in SPECS:
        path = Path(spec["path"])
        if not path.exists():
            errors.append(f"{spec['label']}: missing {path.relative_to(ROOT)}")
            continue
        row = find_row(path, str(spec["method"]))
        check_common(errors, str(spec["label"]), row, "pointmaze-teleport-navigate-v0")
        success = float(row["overall_success"])
        agreement = float(row["action_agreement_to_full"])
        if success > float(spec["max_success"]):
            errors.append(
                f"{spec['label']}: success {success:.3f} exceeds boundary {spec['max_success']:.3f}"
            )
        if row.get("rank_ce_weight") != str(spec["rank_ce_weight"]):
            errors.append(
                f"{spec['label']}: rank_ce_weight {row.get('rank_ce_weight')} "
                f"does not match {spec['rank_ce_weight']}"
            )
        for key in (
            "warmup_steps",
            "steps_per_iter",
            "buffered_final_steps",
            "sampled_target_update",
            "sampled_target_next_samples",
            "sampled_target_waypoint_candidates",
            "sampled_target_waypoints",
        ):
            if key in spec and row.get(key) != str(spec[key]):
                errors.append(f"{spec['label']}: expected {key}={spec[key]}, got {row.get(key)}")
        report.extend(
            [
                f"- {spec['label']}: `{path.relative_to(ROOT)}` method `{spec['method']}` "
                f"success {success:.3f}, action agreement {agreement:.3f}",
            ]
        )

    for spec in POSITIVE_SPECS:
        path = Path(spec["path"])
        if not path.exists():
            errors.append(f"{spec['label']}: missing {path.relative_to(ROOT)}")
            continue
        row = find_row(path, str(spec["method"]))
        check_common(errors, str(spec["label"]), row, str(spec["env"]))
        success = float(row["overall_success"])
        agreement = float(row["action_agreement_to_full"])
        if success < float(spec["min_success"]):
            errors.append(f"{spec['label']}: success {success:.3f} below {spec['min_success']:.3f}")
        if agreement < float(spec["min_action_agreement"]):
            errors.append(
                f"{spec['label']}: action agreement {agreement:.3f} below "
                f"{spec['min_action_agreement']:.3f}"
            )
        for key in ("rank_ce_weight", "buffered_final_steps", "target_buffer_reset_final"):
            if key in spec and row.get(key) != str(spec[key]):
                errors.append(f"{spec['label']}: expected {key}={spec[key]}, got {row.get(key)}")
        report.extend(
            [
                f"- {spec['label']}: `{path.relative_to(ROOT)}` method `{spec['method']}` "
                f"success {success:.3f}, action agreement {agreement:.3f}",
            ]
        )

    if errors:
        report[2] = "Status: FAIL"
        report.extend(["", "## Errors", "", *[f"- {error}" for error in errors]])
        REPORT.write_text("\n".join(report) + "\n")
        raise SystemExit("FAIL: " + "; ".join(errors))

    REPORT.write_text("\n".join(report) + "\n")
    print(f"PASS: checked {len(SPECS)} negative and {len(POSITIVE_SPECS)} positive q-head rows; wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
