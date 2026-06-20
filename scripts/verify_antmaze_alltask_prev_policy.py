#!/usr/bin/env python3
"""Verify bounded AntMaze all-task previous-action policy-head screens."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SPECS = [
    {
        "env": "antmaze-teleport-navigate-v0",
        "path": RESULTS / "antmaze_navigate_prev_policy_alltasks_ep3_seed0.csv",
        "bc_steps": "50000",
        "episodes": "3",
        "min_sto": 0.90,
        "max_matched": 0.25,
        "min_gain": 0.70,
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "path": RESULTS / "antmaze_stitch_prev_policy_alltasks_ep3_seed0.csv",
        "bc_steps": "20000",
        "episodes": "3",
        "min_sto": 0.90,
        "max_matched": 0.25,
        "min_gain": 0.70,
    },
    {
        "env": "antmaze-teleport-navigate-v0",
        "path": RESULTS / "antmaze_navigate_prev_policy_alltasks_ep5_seed0.csv",
        "bc_steps": "50000",
        "episodes": "5",
        "min_sto": 0.90,
        "max_matched": 0.40,
        "min_gain": 0.50,
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "path": RESULTS / "antmaze_stitch_prev_policy_alltasks_ep5_seed0.csv",
        "bc_steps": "20000",
        "episodes": "5",
        "min_sto": 0.95,
        "max_matched": 0.40,
        "min_gain": 0.50,
    },
]
REPORT = RESULTS / "antmaze_alltask_prev_policy_verification.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def by_method(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in rows}


def distinct(rows: list[dict[str, str]], key: str) -> set[str]:
    return {row.get(key, "") for row in rows}


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    errors: list[str] = []
    report = ["# AntMaze All-Task Previous-Action Policy Verification", ""]
    checked = 0
    for spec in SPECS:
        env = str(spec["env"])
        path = Path(spec["path"])
        if not path.exists():
            errors.append(f"{env}: missing source {path.relative_to(ROOT)}")
            continue
        rows = read_rows(path)
        methods = by_method(rows)
        required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
        if not required.issubset(methods):
            errors.append(f"{env}: missing methods {sorted(required - set(methods))}")
            continue

        if distinct(rows, "env") != {env}:
            errors.append(f"{env}: row env mismatch {sorted(distinct(rows, 'env'))}")
        if distinct(rows, "seed") != {"0"}:
            errors.append(f"{env}: expected eval seed 0, got {sorted(distinct(rows, 'seed'))}")
        if distinct(rows, "episodes_per_task") != {str(spec["episodes"])}:
            errors.append(
                f"{env}: expected {spec['episodes']} episodes/task for {path.name}, "
                f"got {sorted(distinct(rows, 'episodes_per_task'))}"
            )
        if distinct(rows, "task_ids") != {"all"}:
            errors.append(f"{env}: expected all tasks, got {sorted(distinct(rows, 'task_ids'))}")
        if distinct(rows, "bc_steps") != {str(spec["bc_steps"])}:
            errors.append(f"{env}: expected bc_steps={spec['bc_steps']}, got {sorted(distinct(rows, 'bc_steps'))}")
        if distinct(rows, "transition_model") != {"raw_obs_mlp"}:
            errors.append(f"{env}: expected raw_obs_mlp transition, got {sorted(distinct(rows, 'transition_model'))}")
        if distinct(rows, "transition_target_source") != {"dataset_jump_changes"}:
            errors.append(
                f"{env}: expected dataset_jump_changes transition source, "
                f"got {sorted(distinct(rows, 'transition_target_source'))}"
            )
        if distinct(rows, "transition_steps") != {"2000"}:
            errors.append(f"{env}: expected 2000 transition steps, got {sorted(distinct(rows, 'transition_steps'))}")
        if distinct(rows, "value_model") != {"raw_obs_prev_policy_mlp"}:
            errors.append(f"{env}: expected previous-action value model, got {sorted(distinct(rows, 'value_model'))}")
        if distinct(rows, "value_steps") != {"2000"}:
            errors.append(f"{env}: expected 2000 value steps, got {sorted(distinct(rows, 'value_steps'))}")
        if distinct(rows, "policy_eval_backend") != {"jax"}:
            errors.append(f"{env}: expected JAX policy backend, got {sorted(distinct(rows, 'policy_eval_backend'))}")
        if distinct(rows, "goal_candidate_mode") != {"body_nearest"} or distinct(rows, "goal_candidates_per_state") != {"16"}:
            errors.append(
                f"{env}: expected body_nearest k16 goals, got "
                f"{sorted(distinct(rows, 'goal_candidate_mode'))}/{sorted(distinct(rows, 'goal_candidates_per_state'))}"
            )
        if distinct(rows, "eval_action_repeat") != {"1"}:
            errors.append(f"{env}: expected action repeat 1, got {sorted(distinct(rows, 'eval_action_repeat'))}")

        matched = as_float(methods["bellman_matched"], "overall_success")
        sto = as_float(methods["sto_trl_matched"], "overall_success")
        full = as_float(methods["bellman_full"], "overall_success")
        sto_agreement = as_float(methods["sto_trl_matched"], "value_action_agreement")
        transition_top1 = as_float(methods["sto_trl_matched"], "transition_oracle_top1")
        matched_sweeps = int(float(methods["bellman_matched"]["iters"]))
        sto_sweeps = int(float(methods["sto_trl_matched"]["iters"]))
        full_sweeps = int(float(methods["bellman_full"]["iters"]))

        if matched_sweeps != 6 or sto_sweeps != 6:
            errors.append(f"{env}: matched/stochastic sweeps are not 6")
        if full_sweeps != 180:
            errors.append(f"{env}: full Bellman sweeps are not 180")
        if sto < float(spec["min_sto"]):
            errors.append(f"{env}: stochastic TRL below {spec['min_sto']:.2f} ({sto:.3f})")
        if matched > float(spec["max_matched"]):
            errors.append(f"{env}: matched Bellman above {spec['max_matched']:.2f} ({matched:.3f})")
        if sto - matched < float(spec["min_gain"]):
            errors.append(
                f"{env}: stochastic-minus-matched below {spec['min_gain']:.2f} ({sto - matched:.3f})"
            )
        if abs(sto - full) > 1e-12:
            errors.append(f"{env}: stochastic TRL does not match full Bellman ({sto:.3f} vs {full:.3f})")
        if sto_agreement < 0.999:
            errors.append(f"{env}: previous-action policy agreement below 0.999 ({sto_agreement:.3f})")
        if transition_top1 < 0.99:
            errors.append(f"{env}: transition oracle top-1 below 0.99 ({transition_top1:.3f})")

        checked += 1
        report.extend(
            [
                f"## {env} ({spec['episodes']} episodes/task)",
                "",
                f"- source: `{path.relative_to(ROOT)}`",
                f"- matched Bellman: {matched:.3f}",
                f"- stochastic TRL: {sto:.3f}",
                f"- full Bellman: {full:.3f}",
                f"- stochastic-minus-matched: {sto - matched:.3f}",
                f"- value action agreement: {sto_agreement:.3f}",
                f"- transition oracle top-1: {transition_top1:.3f}",
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
    print(f"PASS: checked {checked} AntMaze all-task previous-action rows; wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
