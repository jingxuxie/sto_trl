#!/usr/bin/env python3
"""Verify multi-eval-seed PointMaze q-head real-environment screens."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SPECS = [
    {
        "env": "pointmaze-teleport-navigate-v0",
        "path": RESULTS / "pointmaze_qhead_target_fit_navigate_all_env_ep10_seed012.csv",
        "max_qhead_bellman": 0.05,
    },
    {
        "env": "pointmaze-teleport-stitch-v0",
        "path": RESULTS / "pointmaze_qhead_target_fit_stitch_all_env_ep10_seed012.csv",
        "max_qhead_bellman": 0.05,
    },
]
REPORT = RESULTS / "qhead_multiseed_env_claim_verification.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def distinct(rows: list[dict[str, str]], key: str) -> set[str]:
    return {row.get(key, "") for row in rows}


def mean(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values)


def method_rows(rows: list[dict[str, str]], method: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("method") == method]


def main() -> None:
    errors: list[str] = []
    report = ["# Q-Head Multi-Seed Real-Environment Verification", ""]
    checked = 0

    for spec in SPECS:
        env = str(spec["env"])
        path = Path(spec["path"])
        if not path.exists():
            errors.append(f"{env}: missing source {path.relative_to(ROOT)}")
            continue
        rows = read_rows(path)
        required = {
            "qhead_bellman_td",
            "qhead_sto_trl_target_fit",
            "table_bellman_matched",
            "table_sto_trl_matched",
            "table_full_bellman",
        }
        by_method = {method: method_rows(rows, method) for method in required}
        missing = sorted(method for method, vals in by_method.items() if not vals)
        if missing:
            errors.append(f"{env}: missing methods {missing}")
            continue

        if distinct(rows, "env") != {env}:
            errors.append(f"{env}: row env mismatch {sorted(distinct(rows, 'env'))}")
        if distinct(rows, "seed") != {"0", "1", "2"}:
            errors.append(f"{env}: expected eval seeds 0,1,2, got {sorted(distinct(rows, 'seed'))}")
        if distinct(rows, "episodes_per_task") != {"10"}:
            errors.append(f"{env}: expected 10 episodes/task, got {sorted(distinct(rows, 'episodes_per_task'))}")
        if distinct(rows, "task_ids") != {"all"}:
            errors.append(f"{env}: expected all tasks, got {sorted(distinct(rows, 'task_ids'))}")
        if distinct(rows, "eval_mode") != {"env"}:
            errors.append(f"{env}: expected env eval, got {sorted(distinct(rows, 'eval_mode'))}")
        if distinct(rows, "transition_model") != {"raw_obs_mlp"}:
            errors.append(f"{env}: expected raw_obs_mlp transition, got {sorted(distinct(rows, 'transition_model'))}")
        if distinct(rows, "transition_target_source") != {"dataset_cell_changes"}:
            errors.append(
                f"{env}: expected dataset_cell_changes transition source, "
                f"got {sorted(distinct(rows, 'transition_target_source'))}"
            )
        if distinct(rows, "feature_mode") != {"raw_obs_onehot"}:
            errors.append(f"{env}: expected raw_obs_onehot features, got {sorted(distinct(rows, 'feature_mode'))}")

        qhead_bellman = mean(by_method["qhead_bellman_td"], "overall_success")
        qhead_sto = mean(by_method["qhead_sto_trl_target_fit"], "overall_success")
        table_bellman = mean(by_method["table_bellman_matched"], "overall_success")
        table_sto = mean(by_method["table_sto_trl_matched"], "overall_success")
        table_full = mean(by_method["table_full_bellman"], "overall_success")
        action_agree = mean(by_method["qhead_sto_trl_target_fit"], "action_agreement_to_full")

        if qhead_sto < 0.90:
            errors.append(f"{env}: generated-target qhead below 0.90 ({qhead_sto:.3f})")
        if qhead_bellman > float(spec["max_qhead_bellman"]):
            errors.append(
                f"{env}: qhead Bellman TD above {spec['max_qhead_bellman']:.2f} "
                f"({qhead_bellman:.3f})"
            )
        if table_bellman > 0.45:
            errors.append(f"{env}: table matched Bellman above 0.45 ({table_bellman:.3f})")
        if qhead_sto - table_bellman < 0.50:
            errors.append(f"{env}: qhead-vs-table-Bellman gain below 0.50 ({qhead_sto - table_bellman:.3f})")
        if qhead_sto - qhead_bellman < 0.85:
            errors.append(f"{env}: qhead-vs-qhead-Bellman gain below 0.85 ({qhead_sto - qhead_bellman:.3f})")
        if abs(qhead_sto - table_sto) > 1e-12:
            errors.append(f"{env}: qhead stochastic does not match table stochastic")
        if abs(qhead_sto - table_full) > 1e-12:
            errors.append(f"{env}: qhead stochastic does not match table full Bellman")
        if action_agree < 0.95:
            errors.append(f"{env}: qhead action agreement below 0.95 ({action_agree:.3f})")

        checked += 1
        report.extend(
            [
                f"## {env}",
                "",
                f"- source: `{path.relative_to(ROOT)}`",
                f"- qhead Bellman TD: {qhead_bellman:.3f}",
                f"- generated-target qhead: {qhead_sto:.3f}",
                f"- table matched Bellman: {table_bellman:.3f}",
                f"- table stochastic TRL: {table_sto:.3f}",
                f"- table full Bellman: {table_full:.3f}",
                f"- qhead action agreement to full: {action_agree:.3f}",
                "",
            ]
        )

    if errors:
        report.insert(2, "Status: FAIL")
        report.insert(3, "")
        report.extend(["## Errors", "", *[f"- {error}" for error in errors], ""])
        REPORT.write_text("\n".join(report))
        raise SystemExit("FAIL: " + "; ".join(errors))

    report.insert(2, "Status: PASS")
    report.insert(3, f"Checked environments: {checked}")
    report.insert(4, "")
    REPORT.write_text("\n".join(report))
    print(f"PASS: checked {checked} q-head multi-seed env rows; wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
