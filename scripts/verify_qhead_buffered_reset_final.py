#!/usr/bin/env python3
"""Verify PointMaze q-head target-buffer reset-final screens."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = RESULTS / "qhead_buffered_reset_final_verification.md"

SPECS = [
    {
        "label": "navigate exact",
        "path": RESULTS / "pointmaze_qhead_buffered_reset_final_navigate_all_exact.csv",
        "method": "qhead_sto_trl_buffered_reset_final",
        "env": "pointmaze-teleport-navigate-v0",
        "eval_mode": "model",
        "model_rollout_mode": "exact",
        "episodes_per_task": "1",
        "min_success": 0.99,
        "min_agreement": 0.95,
        "check_old_buffered": True,
    },
    {
        "label": "stitch exact",
        "path": RESULTS / "pointmaze_qhead_buffered_reset_final_stitch_all_exact.csv",
        "method": "qhead_sto_trl_buffered_reset_final",
        "env": "pointmaze-teleport-stitch-v0",
        "eval_mode": "model",
        "model_rollout_mode": "exact",
        "episodes_per_task": "1",
        "min_success": 0.99,
        "min_agreement": 0.95,
        "check_old_buffered": False,
    },
    {
        "label": "navigate topk4 exact",
        "path": RESULTS / "pointmaze_qhead_buffered_topk4_reset_final_navigate_all_exact.csv",
        "method": "qhead_sto_trl_buffered_topk_reset_final",
        "env": "pointmaze-teleport-navigate-v0",
        "eval_mode": "model",
        "model_rollout_mode": "exact",
        "episodes_per_task": "1",
        "min_success": 0.99,
        "min_agreement": 0.95,
        "check_old_buffered": False,
        "check_target_fit": False,
        "target_buffer_waypoint_mode": "topk_bridge",
        "target_buffer_waypoints": "4",
    },
    {
        "label": "stitch topk4 exact",
        "path": RESULTS / "pointmaze_qhead_buffered_topk4_reset_final_stitch_all_exact.csv",
        "method": "qhead_sto_trl_buffered_topk_reset_final",
        "env": "pointmaze-teleport-stitch-v0",
        "eval_mode": "model",
        "model_rollout_mode": "exact",
        "episodes_per_task": "1",
        "min_success": 0.99,
        "min_agreement": 0.95,
        "check_old_buffered": False,
        "check_target_fit": False,
        "target_buffer_waypoint_mode": "topk_bridge",
        "target_buffer_waypoints": "4",
    },
    {
        "label": "navigate topk4 real env seeds012",
        "path": RESULTS / "pointmaze_qhead_buffered_topk4_reset_final_navigate_all_env_ep10_seed012.csv",
        "method": "qhead_sto_trl_buffered_topk_reset_final",
        "env": "pointmaze-teleport-navigate-v0",
        "eval_mode": "env",
        "model_rollout_mode": "",
        "episodes_per_task": "10",
        "min_success": 0.90,
        "min_agreement": 0.95,
        "check_old_buffered": False,
        "seeds": {"0", "1", "2"},
        "target_buffer_waypoint_mode": "topk_bridge",
        "target_buffer_waypoints": "4",
    },
    {
        "label": "stitch topk4 real env seeds012",
        "path": RESULTS / "pointmaze_qhead_buffered_topk4_reset_final_stitch_all_env_ep10_seed012.csv",
        "method": "qhead_sto_trl_buffered_topk_reset_final",
        "env": "pointmaze-teleport-stitch-v0",
        "eval_mode": "env",
        "model_rollout_mode": "",
        "episodes_per_task": "10",
        "min_success": 0.90,
        "min_agreement": 0.95,
        "check_old_buffered": False,
        "seeds": {"0", "1", "2"},
        "target_buffer_waypoint_mode": "topk_bridge",
        "target_buffer_waypoints": "4",
    },
    {
        "label": "navigate real env seed0",
        "path": RESULTS / "pointmaze_qhead_buffered_reset_final_navigate_all_env_ep10_seed0.csv",
        "method": "qhead_sto_trl_buffered_reset_final",
        "env": "pointmaze-teleport-navigate-v0",
        "eval_mode": "env",
        "model_rollout_mode": "",
        "episodes_per_task": "10",
        "min_success": 0.95,
        "min_agreement": 0.95,
        "check_old_buffered": False,
    },
    {
        "label": "navigate real env seeds012",
        "path": RESULTS / "pointmaze_qhead_buffered_reset_final_navigate_all_env_ep10_seed012.csv",
        "method": "qhead_sto_trl_buffered_reset_final",
        "env": "pointmaze-teleport-navigate-v0",
        "eval_mode": "env",
        "model_rollout_mode": "",
        "episodes_per_task": "10",
        "min_success": 0.90,
        "min_agreement": 0.95,
        "check_old_buffered": False,
        "seeds": {"0", "1", "2"},
    },
    {
        "label": "stitch real env seeds012",
        "path": RESULTS / "pointmaze_qhead_buffered_reset_final_stitch_all_env_ep10_seed012.csv",
        "method": "qhead_sto_trl_buffered_reset_final",
        "env": "pointmaze-teleport-stitch-v0",
        "eval_mode": "env",
        "model_rollout_mode": "",
        "episodes_per_task": "10",
        "min_success": 0.90,
        "min_agreement": 0.95,
        "check_old_buffered": False,
        "seeds": {"0", "1", "2"},
    },
    {
        "label": "stitch real env seed0",
        "path": RESULTS / "pointmaze_qhead_buffered_reset_final_stitch_all_env_ep10_seed0.csv",
        "method": "qhead_sto_trl_buffered_reset_final",
        "env": "pointmaze-teleport-stitch-v0",
        "eval_mode": "env",
        "model_rollout_mode": "",
        "episodes_per_task": "10",
        "min_success": 0.95,
        "min_agreement": 0.95,
        "check_old_buffered": False,
    },
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rows_by_method(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    return grouped


def mean(rows: list[dict[str, str]], key: str) -> float:
    return float(sum(float(row[key]) for row in rows) / max(len(rows), 1))


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
    }
    for key, value in expected.items():
        if row.get(key, "") != value:
            errors.append(f"{label}: expected {key}={value}, got {row.get(key)}")


def main() -> None:
    errors: list[str] = []
    report = ["# Q-Head Target-Buffer Reset-Final Verification", "", "Status: PASS", ""]

    for spec in SPECS:
        path = Path(spec["path"])
        label = str(spec["label"])
        if not path.exists():
            errors.append(f"{label}: missing {path.relative_to(ROOT)}")
            continue
        methods = rows_by_method(read_rows(path))
        reset_method = str(spec["method"])
        required = {reset_method, "table_sto_trl_matched", "table_full_bellman"}
        check_target_fit = bool(spec.get("check_target_fit", spec["eval_mode"] == "model"))
        if check_target_fit:
            required.add("qhead_sto_trl_target_fit")
        missing = required - set(methods)
        if missing:
            errors.append(f"{label}: missing methods {sorted(missing)}")
            continue

        reset_rows = methods[reset_method]
        expected_seeds = spec.get("seeds")
        if expected_seeds is not None:
            got_seeds = {row.get("seed", "") for row in reset_rows}
            if got_seeds != expected_seeds:
                errors.append(f"{label}: expected seeds {sorted(expected_seeds)}, got {sorted(got_seeds)}")
        for row in reset_rows:
            check_common(errors, label, row, spec)
            if row.get("target_buffer_reset_final") != "True":
                errors.append(f"{label}: reset-final flag is {row.get('target_buffer_reset_final')}")
            for key in ("target_buffer_waypoint_mode", "target_buffer_waypoints"):
                if key in spec and row.get(key) != str(spec[key]):
                    errors.append(f"{label}: expected {key}={spec[key]}, got {row.get(key)}")
        reset_success = mean(reset_rows, "overall_success")
        reset_agreement = mean(reset_rows, "action_agreement_to_full")
        table_sto_success = mean(methods["table_sto_trl_matched"], "overall_success")
        table_full_success = mean(methods["table_full_bellman"], "overall_success")
        if reset_success < float(spec["min_success"]):
            errors.append(f"{label}: reset-final success below threshold ({reset_success:.3f})")
        if reset_agreement < float(spec["min_agreement"]):
            errors.append(f"{label}: reset-final action agreement below threshold ({reset_agreement:.3f})")
        if abs(reset_success - table_sto_success) > 1e-6:
            errors.append(f"{label}: reset-final does not match table stochastic")
        if abs(reset_success - table_full_success) > 1e-6:
            errors.append(f"{label}: reset-final does not match full Bellman")
        if check_target_fit:
            target_fit_success = mean(methods["qhead_sto_trl_target_fit"], "overall_success")
            if abs(reset_success - target_fit_success) > 1e-6:
                errors.append(f"{label}: reset-final does not match generated-target q-head")
        elif "qhead_sto_trl_target_fit" in methods:
            target_fit_success = mean(methods["qhead_sto_trl_target_fit"], "overall_success")
            if abs(reset_success - target_fit_success) > 1e-6:
                errors.append(f"{label}: reset-final does not match generated-target q-head")
        if spec["check_old_buffered"]:
            old = methods.get("qhead_sto_trl_buffered")
            if old is None:
                errors.append(f"{label}: missing old warm-started buffered comparator")
            elif mean(old, "overall_success") > 0.25:
                errors.append(f"{label}: old buffered comparator unexpectedly high")

        report.extend(
            [
                f"## {label}",
                "",
                f"- source: `{path.relative_to(ROOT)}`",
                f"- reset-final success: {reset_success:.3f}",
                f"- table stochastic success: {table_sto_success:.3f}",
                f"- full Bellman success: {table_full_success:.3f}",
                f"- action agreement to full: {reset_agreement:.3f}",
                "",
            ]
        )

    if errors:
        report[2] = "Status: FAIL"
        report.extend(["## Errors", "", *[f"- {error}" for error in errors]])
        REPORT.write_text("\n".join(report) + "\n")
        raise SystemExit("FAIL: " + "; ".join(errors))

    REPORT.write_text("\n".join(report) + "\n")
    print(f"PASS: checked {len(SPECS)} reset-final q-head screens; wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
