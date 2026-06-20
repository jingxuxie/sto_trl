#!/usr/bin/env python3
"""Verify AntMaze raw-observation q-head hard-task diagnostic screens."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = RESULTS / "antmaze_qhead_hard_task_verification.md"

SPECS = [
    {
        "label": "navigate q-head 5k",
        "env": "antmaze-teleport-navigate-v0",
        "path": RESULTS / "antmaze_navigate_qhead_hard_task45_ep3_seed0.csv",
        "bc_steps": "50000",
        "value_steps": "5000",
        "value_model": "raw_obs_qhead_mlp",
        "required": {"bellman_matched", "sto_trl_matched", "bellman_full"},
        "min_sto": 0.99,
        "max_sto": None,
        "max_matched": 0.40,
        "expect_full_match": True,
    },
    {
        "label": "stitch q-head 5k",
        "env": "antmaze-teleport-stitch-v0",
        "path": RESULTS / "antmaze_stitch_qhead_hard_task45_ep3_seed0.csv",
        "bc_steps": "20000",
        "value_steps": "5000",
        "value_model": "raw_obs_qhead_mlp",
        "required": {"bellman_matched", "sto_trl_matched", "bellman_full"},
        "min_sto": 0.60,
        "max_sto": None,
        "max_matched": 0.40,
        "expect_full_match": True,
    },
    {
        "label": "stitch q-head 10k",
        "env": "antmaze-teleport-stitch-v0",
        "path": RESULTS / "antmaze_stitch_qhead_hard_task45_ep3_seed0_10k.csv",
        "bc_steps": "20000",
        "value_steps": "10000",
        "value_model": "raw_obs_qhead_mlp",
        "required": {"sto_trl_matched", "bellman_full"},
        "min_sto": 0.60,
        "max_sto": None,
        "max_matched": None,
        "expect_full_match": True,
    },
    {
        "label": "stitch previous-action q-head 5k",
        "env": "antmaze-teleport-stitch-v0",
        "path": RESULTS / "antmaze_stitch_prev_qhead_hard_task45_ep3_seed0.csv",
        "bc_steps": "20000",
        "value_steps": "5000",
        "value_model": "raw_obs_prev_qhead_mlp",
        "required": {"bellman_matched", "sto_trl_matched", "bellman_full"},
        "min_sto": None,
        "max_sto": 0.01,
        "max_matched": 0.40,
        "expect_full_match": True,
    },
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def by_method(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in rows}


def distinct(rows: list[dict[str, str]], key: str) -> set[str]:
    return {row.get(key, "") for row in rows}


def main() -> None:
    errors: list[str] = []
    report = ["# AntMaze Q-Head Hard-Task Verification", "", "Status: PASS", ""]
    for spec in SPECS:
        path = Path(spec["path"])
        if not path.exists():
            errors.append(f"{spec['label']}: missing {path.relative_to(ROOT)}")
            continue
        rows = read_rows(path)
        methods = by_method(rows)
        missing = set(spec["required"]) - set(methods)
        if missing:
            errors.append(f"{spec['label']}: missing methods {sorted(missing)}")
            continue

        common = {
            "env": str(spec["env"]),
            "seed": "0",
            "episodes_per_task": "3",
            "task_ids": "4,5",
            "bc_steps": str(spec["bc_steps"]),
            "transition_model": "raw_obs_mlp",
            "transition_target_source": "dataset_jump_changes",
            "transition_steps": "2000",
            "value_model": str(spec["value_model"]),
            "value_steps": str(spec["value_steps"]),
            "policy_eval_backend": "jax",
            "goal_candidate_mode": "body_nearest",
            "goal_candidates_per_state": "16",
            "eval_action_repeat": "1",
        }
        for key, expected in common.items():
            vals = distinct(rows, key)
            if vals != {expected}:
                errors.append(f"{spec['label']}: expected {key}={expected}, got {sorted(vals)}")

        sto = float(methods["sto_trl_matched"]["overall_success"])
        full = float(methods["bellman_full"]["overall_success"])
        matched = (
            float(methods["bellman_matched"]["overall_success"])
            if "bellman_matched" in methods
            else None
        )
        if int(float(methods["sto_trl_matched"]["iters"])) != 6:
            errors.append(f"{spec['label']}: stochastic sweeps are not 6")
        if int(float(methods["bellman_full"]["iters"])) != 180:
            errors.append(f"{spec['label']}: full Bellman sweeps are not 180")
        if "bellman_matched" in methods and int(float(methods["bellman_matched"]["iters"])) != 6:
            errors.append(f"{spec['label']}: matched Bellman sweeps are not 6")
        if spec["min_sto"] is not None and sto < float(spec["min_sto"]):
            errors.append(f"{spec['label']}: stochastic q-head below threshold ({sto:.3f})")
        if spec["max_sto"] is not None and sto > float(spec["max_sto"]):
            errors.append(f"{spec['label']}: stochastic q-head above boundary ({sto:.3f})")
        if spec["max_matched"] is not None and matched is not None and matched > float(spec["max_matched"]):
            errors.append(f"{spec['label']}: matched q-head above threshold ({matched:.3f})")
        if bool(spec["expect_full_match"]) and abs(sto - full) > 1e-12:
            errors.append(f"{spec['label']}: stochastic q-head does not match full q-head ({sto:.3f} vs {full:.3f})")

        report.extend(
            [
                f"## {spec['label']}",
                "",
                f"- source: `{path.relative_to(ROOT)}`",
                f"- matched Bellman q-head: {'n/a' if matched is None else f'{matched:.3f}'}",
                f"- stochastic TRL q-head: {sto:.3f}",
                f"- full Bellman q-head: {full:.3f}",
                f"- stochastic task4/task5: "
                f"{float(methods['sto_trl_matched']['task4_success']):.3f}/"
                f"{float(methods['sto_trl_matched']['task5_success']):.3f}",
                f"- stochastic action agreement: "
                f"{float(methods['sto_trl_matched']['value_action_agreement']):.3f}",
                "",
            ]
        )

    if errors:
        report[2] = "Status: FAIL"
        report.extend(["## Errors", "", *[f"- {error}" for error in errors]])
        REPORT.write_text("\n".join(report) + "\n")
        raise SystemExit("FAIL: " + "; ".join(errors))

    REPORT.write_text("\n".join(report) + "\n")
    print(f"PASS: checked {len(SPECS)} AntMaze q-head hard-task screens; wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
