#!/usr/bin/env python3
"""Verify LaTeX table claims against generated stochastic TRL CSV tables."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "paper_tables"
RESULTS_DIR = ROOT / "results"
TEX_PATH = ROOT / "paper" / "stochastic_trl" / "main.tex"
REPORT_PATH = ROOT / "paper" / "stochastic_trl" / "latex_claim_verification.md"

GRID_BUDGET = TABLE_DIR / "grid_budget_curve.csv"
NEURAL_SHORTCUT = TABLE_DIR / "neural_shortcut_phase.csv"
MAIN_HARD_TASKS = TABLE_DIR / "main_hard_task_results.csv"
SUPPORT_ABLATION = TABLE_DIR / "pointmaze_topology_stitch_support_baseline_5seed.csv"
ANTMAZE_NAVIGATE_CONTROLLERS = TABLE_DIR / "antmaze_navigate_controller_seeds_ep20_seed012.csv"
ANTMAZE_STITCH_CONTROLLERS = TABLE_DIR / "antmaze_stitch_controller_seeds_ep20_seed012.csv"
HARD_TASK_STRESS = TABLE_DIR / "hard_task_stress_seed0.csv"
POINTMAZE_SINGLE_TASK = TABLE_DIR / "pointmaze_stitch_task5_ep100_seed01234.csv"
POINTMAZE_LEARNED_TRANSITION = TABLE_DIR / "pointmaze_learned_transition.csv"
POINTMAZE_LEARNED_CONTROLLER = TABLE_DIR / "pointmaze_learned_controller_ep20_seed012.csv"
CONTROLLER_EXECUTION_ISOLATION = TABLE_DIR / "controller_execution_isolation.csv"
POINTMAZE_TIE_POLICY_HEAD = TABLE_DIR / "pointmaze_tie_policy_head_ep20_seed0.csv"
POINTMAZE_QHEAD_NAVIGATE = RESULTS_DIR / "pointmaze_qhead_target_fit_navigate_all_exact.csv"
POINTMAZE_QHEAD_STITCH = RESULTS_DIR / "pointmaze_qhead_target_fit_stitch_all_exact.csv"
POINTMAZE_PREV_POLICY_HEAD = TABLE_DIR / "pointmaze_rawobs_transition_prev_policy_head_ep20_seed0.csv"
POINTMAZE_TIE_POLICY_HEAD_EVAL_SEED = TABLE_DIR / "pointmaze_tie_policy_head_ep20_evalseed012.csv"
POINTMAZE_TIE_POLICY_HEAD_TRANSITION_SEED = TABLE_DIR / "pointmaze_tie_policy_head_ep20_tseed012.csv"
ANTMAZE_TIE_POLICY_HEAD = TABLE_DIR / "antmaze_tie_policy_head_hard_tasks_ep20_seed0.csv"
ANTMAZE_RAWOBS_TIE_POLICY_HEAD = TABLE_DIR / "antmaze_rawobs_transition_tie_policy_head_ep10_tseed012.csv"
ANTMAZE_RAWOBS_TIE_POLICY_EVAL_SEED = TABLE_DIR / "antmaze_rawobs_transition_tie_policy_head_ep10_evalseed012.csv"
ANTMAZE_RAWOBS_PREV_POLICY_EVAL_SEED = TABLE_DIR / "antmaze_rawobs_transition_prev_policy_head_ep10_evalseed012.csv"
ANTMAZE_LEARNED_TRANSITION = TABLE_DIR / "antmaze_learned_transition_robustness.csv"

MAIN_ENV_LABELS = {
    "pointmaze-teleport-navigate-v0": ("PointMaze navigate", "topology scaffold"),
    "pointmaze-teleport-stitch-v0": ("PointMaze stitch", "topology scaffold"),
    "antmaze-teleport-navigate-v0": ("AntMaze navigate", "BC executor"),
    "antmaze-teleport-stitch-v0": ("AntMaze stitch", "BC executor"),
}

STRESS_LABELS = {
    "pointmaze-teleport-stitch-v0": "PointMaze stitch",
    "antmaze-teleport-navigate-v0": "AntMaze navigate",
    "antmaze-teleport-stitch-v0": "AntMaze stitch",
}

LEARNED_TRANSITION_LABELS = {
    "pointmaze-teleport-navigate-v0": "PointMaze navigate",
    "pointmaze-teleport-stitch-v0": "PointMaze stitch",
    "antmaze-teleport-navigate-v0": "AntMaze navigate",
    "antmaze-teleport-stitch-v0": "AntMaze stitch",
}

POINTMAZE_TRANSITION_MODEL_LABELS = {
    "dataset cell-change softmax": "table softmax",
    "raw-observation MLP cell-change": "raw obs MLP",
}

POINTMAZE_CONTROLLER_LABELS = {
    "pointmaze-teleport-navigate-v0": "PointMaze navigate",
    "pointmaze-teleport-stitch-v0": "PointMaze stitch",
}

ANTMAZE_TRANSITION_MODEL_LABELS = {
    "table softmax, 20 samples/row": "learned transition + BC",
    "raw-observation MLP jump-change": "raw obs MLP + BC",
}

NEURAL_SHORTCUT_LABELS = {
    "neural_bellman_td": "Neural Bellman TD",
    "neural_sto_trl": "Neural Stochastic TRL",
    "neural_support_trl": "Neural Support TRL",
    "table_sto_trl_matched": "Table Stochastic TRL",
    "table_full_bellman": "Table full Bellman",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fmt(value: str | float) -> str:
    return f"{float(value):.3f}"


def latex_row(*cells: object) -> str:
    return " & ".join(str(cell) for cell in cells) + r"\\"


def lookup(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        criteria_text = ", ".join(f"{key}={value!r}" for key, value in criteria.items())
        raise ValueError(f"expected one row for {criteria_text}, found {len(matches)}")
    return matches[0]


def mean_metric(rows: list[dict[str, str]], method: str, metric: str) -> float:
    vals = [float(row[metric]) for row in rows if row.get("method") == method]
    if not vals:
        raise ValueError(f"no rows for method={method!r} metric={metric!r}")
    return sum(vals) / len(vals)


def add_required(
    checks: list[tuple[str, str]],
    label: str,
    snippet: str,
) -> None:
    checks.append((label, snippet))


def build_expected_rows() -> list[tuple[str, str]]:
    checks: list[tuple[str, str]] = []

    grid_rows = read_rows(GRID_BUDGET)
    for method, sweeps in [
        ("Bellman", "8"),
        ("Bellman", "120"),
        ("Bellman", "126"),
        ("Stochastic TRL", "6"),
        ("Stochastic TRL", "7"),
    ]:
        row = lookup(grid_rows, method=method, sweeps=sweeps)
        add_required(
            checks,
            f"grid budget {method} {sweeps}",
            latex_row(
                method,
                sweeps,
                fmt(row["success_rate"]),
                fmt(row["safe_action_rate"]),
                fmt(row["portal_action_rate"]),
            ),
        )

    neural_rows = read_rows(NEURAL_SHORTCUT)
    for method in [
        "neural_bellman_td",
        "neural_sto_trl",
        "neural_support_trl",
        "table_sto_trl_matched",
        "table_full_bellman",
    ]:
        add_required(
            checks,
            f"neural shortcut {method}",
            latex_row(
                NEURAL_SHORTCUT_LABELS[method],
                fmt(mean_metric(neural_rows, method, "success_rate")),
                fmt(mean_metric(neural_rows, method, "decision_correct")),
                fmt(mean_metric(neural_rows, method, "risky_action")),
            ),
        )

    for row in read_rows(MAIN_HARD_TASKS):
        env_label, executor_label = MAIN_ENV_LABELS[row["env"]]
        add_required(
            checks,
            f"main hard task {row['env']}",
            latex_row(
                env_label,
                executor_label,
                fmt(row["bellman_matched"]),
                fmt(row["stochastic_trl"]),
                fmt(row["bellman_full"]),
                fmt(row["sto_minus_matched"]),
            ),
        )

    for row in read_rows(SUPPORT_ABLATION):
        add_required(
            checks,
            f"support ablation {row['method']}",
            latex_row(row["method"], row["sweeps"], fmt(row["mean_success"])),
        )

    for task_label, path in [
        ("Navigate", ANTMAZE_NAVIGATE_CONTROLLERS),
        ("Stitch", ANTMAZE_STITCH_CONTROLLERS),
    ]:
        rows = read_rows(path)
        for controller_seed in ["0", "1", "2"]:
            matched = lookup(rows, controller_seed=controller_seed, method="Bellman matched")
            stochastic = lookup(rows, controller_seed=controller_seed, method="Stochastic TRL")
            full = lookup(rows, controller_seed=controller_seed, method="Bellman full")
            add_required(
                checks,
                f"antmaze {task_label.lower()} controller seed {controller_seed}",
                latex_row(
                    task_label,
                    controller_seed,
                    fmt(matched["mean_success"]),
                    fmt(stochastic["mean_success"]),
                    fmt(full["mean_success"]),
                ),
            )

    for row in read_rows(HARD_TASK_STRESS):
        support = "--" if row["support_trl"] == "" else fmt(row["support_trl"])
        add_required(
            checks,
            f"hard-task stress {row['env']}",
            latex_row(
                STRESS_LABELS[row["env"]],
                row["task_scope"],
                fmt(row["bellman_matched"]),
                support,
                fmt(row["stochastic_trl"]),
                fmt(row["bellman_full"]),
            ),
        )

    for row in read_rows(POINTMAZE_SINGLE_TASK):
        add_required(
            checks,
            f"focused pointmaze task {row['task_id']}",
            latex_row(
                "PointMaze stitch",
                row["task_id"],
                row["eval_setting"],
                fmt(row["bellman_matched"]),
                fmt(row["stochastic_trl"]),
                fmt(row["bellman_full"]),
            ),
        )

    for row in read_rows(POINTMAZE_LEARNED_TRANSITION):
        model_label = POINTMAZE_TRANSITION_MODEL_LABELS[row["transition_model"]]
        add_required(
            checks,
            f"pointmaze learned transition {row['env']} {row['transition_model']}",
            latex_row(
                LEARNED_TRANSITION_LABELS[row["env"]],
                model_label,
                row["eval_setting"],
                fmt(row["bellman_matched"]),
                fmt(row["stochastic_trl"]),
                fmt(row["bellman_full"]),
            ),
        )

    for row in read_rows(POINTMAZE_LEARNED_CONTROLLER):
        add_required(
            checks,
            f"pointmaze learned controller {row['env']}",
            latex_row(
                POINTMAZE_CONTROLLER_LABELS[row["env"]],
                row["controller"],
                row["eval_setting"],
                fmt(row["bellman_matched"]),
                fmt(row["stochastic_trl"]),
                fmt(row["bellman_full"]),
            ),
        )

    for row in read_rows(CONTROLLER_EXECUTION_ISOLATION):
        add_required(
            checks,
            f"controller execution isolation {row['execution_path']} {row['env']}",
            latex_row(
                row["execution_path"],
                POINTMAZE_CONTROLLER_LABELS[row["env"]],
                row["controller_or_actor"],
                row["eval_setting"],
                fmt(row["best_success"]),
                fmt(row["final_success"]),
            ),
        )

    for row in read_rows(POINTMAZE_TIE_POLICY_HEAD):
        add_required(
            checks,
            f"pointmaze tie-policy head {row['env']}",
            latex_row(
                LEARNED_TRANSITION_LABELS[row["env"]],
                row["control_head"],
                row["eval_setting"],
                fmt(row["bellman_matched"]),
                fmt(row["stochastic_trl"]),
                fmt(row["bellman_full"]),
                fmt(row["value_action_agreement"]),
            ),
        )

    for env_label, path in [
        ("PointMaze navigate", POINTMAZE_QHEAD_NAVIGATE),
        ("PointMaze stitch", POINTMAZE_QHEAD_STITCH),
    ]:
        rows = read_rows(path)
        qhead_bellman = lookup(rows, method="qhead_bellman_td")
        qhead_sto = lookup(rows, method="qhead_sto_trl_target_fit")
        table_bellman = lookup(rows, method="table_bellman_matched")
        table_sto = lookup(rows, method="table_sto_trl_matched")
        add_required(
            checks,
            f"pointmaze qhead critic {env_label}",
            latex_row(
                env_label,
                fmt(qhead_bellman["overall_success"]),
                fmt(qhead_sto["overall_success"]),
                fmt(table_bellman["overall_success"]),
                fmt(table_sto["overall_success"]),
                fmt(qhead_sto["action_agreement_to_full"]),
            ),
        )

    for row in read_rows(POINTMAZE_PREV_POLICY_HEAD):
        add_required(
            checks,
            f"pointmaze previous-action policy head {row['env']}",
            latex_row(
                LEARNED_TRANSITION_LABELS[row["env"]],
                row["control_head"],
                row["eval_setting"],
                fmt(row["bellman_matched"]),
                fmt(row["stochastic_trl"]),
                fmt(row["bellman_full"]),
                fmt(row["value_action_agreement"]),
            ),
        )

    for row in read_rows(POINTMAZE_TIE_POLICY_HEAD_EVAL_SEED):
        add_required(
            checks,
            f"pointmaze tie-policy eval-seed {row['env']}",
            latex_row(
                LEARNED_TRANSITION_LABELS[row["env"]],
                row["control_head"],
                row["eval_setting"],
                fmt(row["bellman_matched"]),
                fmt(row["stochastic_trl"]),
                fmt(row["bellman_full"]),
                fmt(row["value_action_agreement_min"]),
            ),
        )

    for row in read_rows(POINTMAZE_TIE_POLICY_HEAD_TRANSITION_SEED):
        add_required(
            checks,
            f"pointmaze tie-policy transition-seed {row['env']}",
            latex_row(
                LEARNED_TRANSITION_LABELS[row["env"]],
                row["control_head"],
                row["eval_setting"],
                fmt(row["bellman_matched"]),
                fmt(row["stochastic_trl"]),
                fmt(row["bellman_full"]),
                fmt(row["value_action_agreement_min"]),
            ),
        )

    for row in read_rows(ANTMAZE_TIE_POLICY_HEAD):
        add_required(
            checks,
            f"antmaze tie-policy head {row['env']}",
            latex_row(
                LEARNED_TRANSITION_LABELS[row["env"]],
                row["controller"],
                row["eval_setting"],
                fmt(row["bellman_matched"]),
                fmt(row["stochastic_trl"]),
                fmt(row["bellman_full"]),
                fmt(row["value_action_agreement"]),
            ),
        )

    for row in read_rows(ANTMAZE_RAWOBS_TIE_POLICY_HEAD):
        add_required(
            checks,
            f"antmaze rawobs transition tie-policy head {row['env']}",
            latex_row(
                LEARNED_TRANSITION_LABELS[row["env"]],
                row["controller"],
                row["eval_setting"],
                fmt(row["bellman_matched"]),
                fmt(row["stochastic_trl"]),
                fmt(row["bellman_full"]),
                fmt(row["transition_oracle_top1_min"]),
                fmt(row["value_action_agreement_min"]),
            ),
        )

    for row in read_rows(ANTMAZE_RAWOBS_TIE_POLICY_EVAL_SEED):
        add_required(
            checks,
            f"antmaze rawobs transition tie-policy eval-seed {row['env']}",
            latex_row(
                LEARNED_TRANSITION_LABELS[row["env"]],
                row["controller"],
                row["eval_setting"],
                fmt(row["bellman_matched"]),
                fmt(row["stochastic_trl"]),
                fmt(row["bellman_full"]),
                fmt(row["transition_oracle_top1"]),
                fmt(row["value_action_agreement_min"]),
            ),
        )

    for row in read_rows(ANTMAZE_RAWOBS_PREV_POLICY_EVAL_SEED):
        add_required(
            checks,
            f"antmaze rawobs transition previous-action eval-seed {row['env']}",
            latex_row(
                LEARNED_TRANSITION_LABELS[row["env"]],
                row["controller"],
                row["eval_setting"],
                fmt(row["bellman_matched"]),
                fmt(row["stochastic_trl"]),
                fmt(row["bellman_full"]),
                fmt(row["transition_oracle_top1"]),
                fmt(row["value_action_agreement_min"]),
            ),
        )

    for row in read_rows(ANTMAZE_LEARNED_TRANSITION):
        n_transition_seeds = len(row["transition_seeds"].split(","))
        seed_label = "transition seed" if n_transition_seeds == 1 else "transition seeds"
        model_label = ANTMAZE_TRANSITION_MODEL_LABELS[row["transition_model"]]
        add_required(
            checks,
            f"antmaze learned transition {row['env']} {row['transition_model']}",
            latex_row(
                LEARNED_TRANSITION_LABELS[row["env"]],
                model_label,
                f"{n_transition_seeds} {seed_label}, 10 episodes/task",
                fmt(row["bellman_matched"]),
                fmt(row["stochastic_trl"]),
                fmt(row["bellman_full"]),
            ),
        )

    return checks


def write_report(status: str, checked: list[tuple[str, str]], errors: list[str]) -> None:
    lines = [
        "# LaTeX Claim Verification",
        "",
        f"Status: {status}",
        f"Checked rows: {len(checked)}",
        "",
    ]
    if errors:
        lines.extend(["## Missing Or Mismatched Rows", ""])
        lines.extend(f"- {error}" for error in errors)
        lines.append("")
    else:
        lines.extend(["## Checked Row Labels", ""])
        lines.extend(f"- {label}" for label, _ in checked)
        lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    tex = TEX_PATH.read_text(encoding="utf-8")
    expected_rows = build_expected_rows()
    errors = [
        f"{label}: missing `{snippet}`"
        for label, snippet in expected_rows
        if snippet not in tex
    ]
    status = "PASS" if not errors else "FAIL"
    write_report(status, expected_rows, errors)

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"Wrote {REPORT_PATH.relative_to(ROOT)}", file=sys.stderr)
        return 1

    print(
        f"PASS: checked {len(expected_rows)} LaTeX table rows; "
        f"wrote {REPORT_PATH.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
