#!/usr/bin/env python3
"""Verify PointMaze sampled-target q-head buffer screens."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = RESULTS / "qhead_sampled_buffered_verification.md"
METHOD = "qhead_sto_trl_sampled_buffered_reset_final"

SPECS = [
    {
        "label": "navigate exact sampled targets",
        "path": RESULTS / "pointmaze_qhead_sampled_buffered_nav_exact_s64_c32_k4.csv",
        "env": "pointmaze-teleport-navigate-v0",
        "eval_mode": "model",
        "model_rollout_mode": "exact",
        "episodes_per_task": "1",
        "min_success": 0.99,
    },
    {
        "label": "stitch exact sampled targets",
        "path": RESULTS / "pointmaze_qhead_sampled_buffered_stitch_exact_s64_c32_k4.csv",
        "env": "pointmaze-teleport-stitch-v0",
        "eval_mode": "model",
        "model_rollout_mode": "exact",
        "episodes_per_task": "1",
        "min_success": 0.99,
    },
    {
        "label": "navigate real env seed0 sampled targets",
        "path": RESULTS / "pointmaze_qhead_sampled_buffered_nav_env_ep10_seed0_s64_c32_k4.csv",
        "env": "pointmaze-teleport-navigate-v0",
        "eval_mode": "env",
        "model_rollout_mode": "",
        "episodes_per_task": "10",
        "min_success": 0.95,
    },
    {
        "label": "stitch real env seed0 sampled targets",
        "path": RESULTS / "pointmaze_qhead_sampled_buffered_stitch_env_ep10_seed0_s64_c32_k4.csv",
        "env": "pointmaze-teleport-stitch-v0",
        "eval_mode": "env",
        "model_rollout_mode": "",
        "episodes_per_task": "10",
        "min_success": 0.95,
    },
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rows_by_method(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    grouped: dict[str, dict[str, str]] = {}
    for row in rows:
        grouped[row["method"]] = row
    return grouped


def check_common(errors: list[str], label: str, row: dict[str, str], spec: dict[str, object]) -> None:
    expected = {
        "env": str(spec["env"]),
        "eval_mode": str(spec["eval_mode"]),
        "model_rollout_mode": str(spec["model_rollout_mode"]),
        "feature_mode": "raw_obs_onehot",
        "transition_model": "raw_obs_mlp",
        "transition_target_source": "dataset_cell_changes",
        "transition_steps": "2000",
        "iters": "6",
        "task_ids": "all",
        "episodes_per_task": str(spec["episodes_per_task"]),
        "rank_ce_weight": "1.0",
        "buffered_final_steps": "5000",
        "sampled_target_next_samples": "64",
        "sampled_target_waypoint_candidates": "32",
        "sampled_target_waypoints": "4",
    }
    for key, value in expected.items():
        if row.get(key, "") != value:
            errors.append(f"{label}: expected {key}={value}, got {row.get(key)}")


def main() -> None:
    errors: list[str] = []
    report = ["# Q-Head Sampled-Target Buffer Verification", "", "Status: PASS", ""]

    for spec in SPECS:
        path = Path(spec["path"])
        label = str(spec["label"])
        if not path.exists():
            errors.append(f"{label}: missing {path.relative_to(ROOT)}")
            continue
        methods = rows_by_method(read_rows(path))
        required = {METHOD, "table_sto_trl_matched", "table_full_bellman"}
        missing = required - set(methods)
        if missing:
            errors.append(f"{label}: missing methods {sorted(missing)}")
            continue

        sampled = methods[METHOD]
        check_common(errors, label, sampled, spec)
        success = float(sampled["overall_success"])
        agreement = float(sampled["action_agreement_to_full"])
        table_sto_success = float(methods["table_sto_trl_matched"]["overall_success"])
        table_full_success = float(methods["table_full_bellman"]["overall_success"])

        if success < float(spec["min_success"]):
            errors.append(f"{label}: sampled q-head success below threshold ({success:.3f})")
        if agreement < 0.90:
            errors.append(f"{label}: sampled q-head action agreement below 0.90 ({agreement:.3f})")
        if abs(success - table_sto_success) > 1e-4:
            errors.append(f"{label}: sampled q-head does not match table stochastic")
        if abs(success - table_full_success) > 1e-4:
            errors.append(f"{label}: sampled q-head does not match full Bellman")

        report.extend(
            [
                f"## {label}",
                "",
                f"- source: `{path.relative_to(ROOT)}`",
                f"- sampled q-head success: {success:.3f}",
                f"- table stochastic success: {table_sto_success:.3f}",
                f"- full Bellman success: {table_full_success:.3f}",
                f"- action agreement to full: {agreement:.3f}",
                "- sampled next states per row: 64",
                "- sampled bridge candidates per state-goal pair: 32",
                "- retained bridge waypoints per backup: 4",
                "",
            ]
        )

    if errors:
        report[2] = "Status: FAIL"
        report.extend(["## Errors", "", *[f"- {error}" for error in errors]])
        REPORT.write_text("\n".join(report) + "\n")
        raise SystemExit("FAIL: " + "; ".join(errors))

    REPORT.write_text("\n".join(report) + "\n")
    print(f"PASS: checked {len(SPECS)} sampled-buffer q-head screens; wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
