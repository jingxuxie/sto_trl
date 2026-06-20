#!/usr/bin/env python3
"""Verify the PointMaze joint-action neural critic diagnostic."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXACT_SOURCES = {
    "pointmaze-teleport-navigate-v0": RESULTS / "pointmaze_qhead_target_fit_navigate_all_exact.csv",
    "pointmaze-teleport-stitch-v0": RESULTS / "pointmaze_qhead_target_fit_stitch_all_exact.csv",
}
ENV_ALL_SOURCES = {
    "pointmaze-teleport-navigate-v0": RESULTS / "pointmaze_qhead_target_fit_navigate_all_env_ep10_seed0.csv",
    "pointmaze-teleport-stitch-v0": RESULTS / "pointmaze_qhead_target_fit_stitch_all_env_ep10_seed0.csv",
}
ENV_HARD_SOURCES = {
    "pointmaze-teleport-navigate-v0": RESULTS / "pointmaze_qhead_target_fit_navigate_task45_env_ep20_seed0.csv",
    "pointmaze-teleport-stitch-v0": RESULTS / "pointmaze_qhead_target_fit_stitch_task45_env_ep20_seed0.csv",
}
REPORT = RESULTS / "qhead_critic_claim_verification.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def find_method(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    matches = [row for row in rows if row.get("method") == method]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one row for {method}, found {len(matches)}")
    return matches[0]


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def check_row_env(
    errors: list[str],
    env: str,
    row: dict[str, str],
    eval_mode: str,
    task_ids: str,
) -> None:
    if row["env"] != env:
        errors.append(f"{env}: row method={row['method']} has env={row['env']}")
    if row["eval_mode"] != eval_mode:
        errors.append(f"{env}: row method={row['method']} is not {eval_mode} eval")
    if row["task_ids"] != task_ids:
        errors.append(f"{env}: row method={row['method']} task_ids={row['task_ids']} not {task_ids}")


def check_generated_target_rollout(
    *,
    env: str,
    path: Path,
    errors: list[str],
    report_lines: list[str],
    label: str,
    expected_task_ids: str,
    min_sto_success: float,
    max_bellman_success: float,
    max_table_bellman_success: float,
    min_improvement: float,
) -> bool:
    if not path.exists():
        errors.append(f"missing source {path.relative_to(ROOT)}")
        return False
    rows = read_rows(path)
    try:
        sto = find_method(rows, "qhead_sto_trl_target_fit")
        bellman = find_method(rows, "qhead_bellman_td")
        table_bellman = find_method(rows, "table_bellman_matched")
        table_sto = find_method(rows, "table_sto_trl_matched")
        table_full = find_method(rows, "table_full_bellman")
    except AssertionError as exc:
        errors.append(f"{env}: {exc}")
        return False

    for row in rows:
        check_row_env(errors, env, row, "env", expected_task_ids)

    sto_success = as_float(sto, "overall_success")
    bellman_success = as_float(bellman, "overall_success")
    table_bellman_success = as_float(table_bellman, "overall_success")
    table_sto_success = as_float(table_sto, "overall_success")
    table_full_success = as_float(table_full, "overall_success")
    sto_agree = as_float(sto, "action_agreement_to_full")

    if sto_success < min_sto_success:
        errors.append(f"{env}: {label} qhead stochastic success below {min_sto_success:.2f} ({sto_success:.6f})")
    if bellman_success > max_bellman_success:
        errors.append(f"{env}: {label} qhead Bellman TD success above {max_bellman_success:.2f} ({bellman_success:.6f})")
    if table_bellman_success > max_table_bellman_success:
        errors.append(
            f"{env}: {label} matched Bellman table success above "
            f"{max_table_bellman_success:.2f} ({table_bellman_success:.6f})"
        )
    if abs(sto_success - table_sto_success) > 1e-6:
        errors.append(
            f"{env}: {label} qhead stochastic success does not match table stochastic "
            f"({sto_success:.6f} vs {table_sto_success:.6f})"
        )
    if abs(sto_success - table_full_success) > 1e-6:
        errors.append(
            f"{env}: {label} qhead stochastic success does not match full Bellman "
            f"({sto_success:.6f} vs {table_full_success:.6f})"
        )
    if sto_success - bellman_success < min_improvement:
        errors.append(
            f"{env}: {label} qhead improvement below {min_improvement:.2f} "
            f"({sto_success - bellman_success:.6f})"
        )
    if sto_agree < 0.95:
        errors.append(f"{env}: {label} qhead action agreement below 0.95 ({sto_agree:.6f})")

    for task_id in (4, 5):
        task_key = f"task{task_id}_success"
        if task_key in sto and sto[task_key] != "" and as_float(sto, task_key) < 0.95:
            errors.append(
                f"{env}: {label} qhead task{task_id} success below 0.95 "
                f"({as_float(sto, task_key):.6f})"
            )

    report_lines.extend(
        [
            f"## {label}: {env}",
            "",
            f"- qhead Bellman TD success: {bellman_success:.3f}",
            f"- generated-target qhead critic success: {sto_success:.3f}",
            f"- table matched Bellman success: {table_bellman_success:.3f}",
            f"- table stochastic TRL success: {table_sto_success:.3f}",
            f"- table full Bellman success: {table_full_success:.3f}",
            f"- qhead action agreement to full Bellman: {sto_agree:.3f}",
            "",
        ]
    )
    return True


def main() -> None:
    errors: list[str] = []
    report_lines = [
        "# Q-Head Critic Claim Verification",
        "",
    ]
    checked = 0
    for env, path in EXACT_SOURCES.items():
        if not path.exists():
            errors.append(f"missing source {path.relative_to(ROOT)}")
            continue
        rows = read_rows(path)
        try:
            sto = find_method(rows, "qhead_sto_trl_target_fit")
            self_bootstrap = find_method(rows, "qhead_sto_trl")
            bellman = find_method(rows, "qhead_bellman_td")
            table_bellman = find_method(rows, "table_bellman_matched")
            table_sto = find_method(rows, "table_sto_trl_matched")
            table_full = find_method(rows, "table_full_bellman")
        except AssertionError as exc:
            errors.append(f"{env}: {exc}")
            continue

        for row in rows:
            check_row_env(errors, env, row, "model", "all")
            if row["model_rollout_mode"] != "exact":
                errors.append(f"{env}: row method={row['method']} is not exact model eval")

        sto_success = as_float(sto, "overall_success")
        self_bootstrap_success = as_float(self_bootstrap, "overall_success")
        bellman_success = as_float(bellman, "overall_success")
        table_bellman_success = as_float(table_bellman, "overall_success")
        table_sto_success = as_float(table_sto, "overall_success")
        table_full_success = as_float(table_full, "overall_success")
        sto_agree = as_float(sto, "action_agreement_to_full")

        if sto_success < 0.99:
            errors.append(f"{env}: qhead stochastic distill success below 0.99 ({sto_success:.6f})")
        if bellman_success > 0.10:
            errors.append(f"{env}: qhead Bellman TD success above 0.10 ({bellman_success:.6f})")
        if self_bootstrap_success > 0.10:
            errors.append(
                f"{env}: self-bootstrapped qhead success above 0.10 ({self_bootstrap_success:.6f})"
            )
        if table_bellman_success > 0.50:
            errors.append(f"{env}: matched Bellman table success above 0.50 ({table_bellman_success:.6f})")
        if abs(sto_success - table_sto_success) > 1e-4:
            errors.append(
                f"{env}: qhead stochastic success does not match table stochastic "
                f"({sto_success:.6f} vs {table_sto_success:.6f})"
            )
        if abs(sto_success - table_full_success) > 1e-4:
            errors.append(
                f"{env}: qhead stochastic success does not match full Bellman "
                f"({sto_success:.6f} vs {table_full_success:.6f})"
            )
        if sto_success - bellman_success < 0.90:
            errors.append(f"{env}: qhead improvement below 0.90 ({sto_success - bellman_success:.6f})")
        if sto_agree < 0.95:
            errors.append(f"{env}: qhead action agreement below 0.95 ({sto_agree:.6f})")

        checked += 1
        report_lines.extend(
            [
                f"## Exact model all tasks: {env}",
                "",
                f"- qhead Bellman TD success: {bellman_success:.3f}",
                f"- self-bootstrapped qhead success: {self_bootstrap_success:.3f}",
                f"- generated-target qhead critic success: {sto_success:.3f}",
                f"- table matched Bellman success: {table_bellman_success:.3f}",
                f"- table stochastic TRL success: {table_sto_success:.3f}",
                f"- table full Bellman success: {table_full_success:.3f}",
                f"- qhead action agreement to full Bellman: {sto_agree:.3f}",
                "",
            ]
        )

    for env, path in ENV_ALL_SOURCES.items():
        if check_generated_target_rollout(
            env=env,
            path=path,
            errors=errors,
            report_lines=report_lines,
            label="Real env all tasks",
            expected_task_ids="all",
            min_sto_success=0.95,
            max_bellman_success=0.10,
            max_table_bellman_success=0.60,
            min_improvement=0.40,
        ):
            checked += 1

    for env, path in ENV_HARD_SOURCES.items():
        if check_generated_target_rollout(
            env=env,
            path=path,
            errors=errors,
            report_lines=report_lines,
            label="Real env hard tasks 4,5",
            expected_task_ids="4,5",
            min_sto_success=0.95,
            max_bellman_success=0.15,
            max_table_bellman_success=0.60,
            min_improvement=0.80,
        ):
            checked += 1

    if errors:
        report_lines.insert(2, "Status: FAIL")
        report_lines.insert(3, "")
        report_lines.extend(["## Errors", "", *[f"- {error}" for error in errors], ""])
        REPORT.write_text("\n".join(report_lines))
        raise SystemExit("FAIL: " + "; ".join(errors))

    report_lines.insert(2, "Status: PASS")
    report_lines.insert(3, f"Checked rows: {checked}")
    report_lines.insert(4, "")
    REPORT.write_text("\n".join(report_lines))
    print(f"PASS: checked {checked} q-head critic rows; wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
