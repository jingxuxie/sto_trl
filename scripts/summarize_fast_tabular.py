from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def method_family(method: str) -> str:
    if method.startswith("sto_trl_"):
        return "sto_trl_matched"
    if method.startswith("bellman_") and method.endswith("_sweeps"):
        return "bellman_matched"
    return method


def finite_mean(rows: list[dict[str, str]], key: str) -> float:
    vals = []
    for row in rows:
        raw = row.get(key, "")
        if raw in {"", "inf", "-inf", "nan"}:
            continue
        val = float(raw)
        if math.isfinite(val):
            vals.append(val)
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def aggregate(rows: list[dict[str, str]], stage: str) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["stage"] == stage:
            groups[method_family(row["method"])].append(row)

    out = []
    for method in sorted(groups):
        method_rows = groups[method]
        safe_opt = [
            row
            for row in method_rows
            if row.get("optimal_start_action", "") != ""
            and int(float(row["optimal_start_action"])) == 0
        ]
        risky_opt = [
            row
            for row in method_rows
            if row.get("optimal_start_action", "") != ""
            and int(float(row["optimal_start_action"])) == 1
        ]
        out.append(
            {
                "method": method,
                "n": str(len(method_rows)),
                "success": fmt(finite_mean(method_rows, "success_rate")),
                "regret": fmt(finite_mean(method_rows, "regret"), 4),
                "long_mse": fmt(finite_mean(method_rows, "long_horizon_mse"), 5),
                "risky_over": fmt(finite_mean(method_rows, "risky_overestimate_ratio")),
                "safe_opt_risky_rate": fmt(finite_mean(safe_opt, "risky_action"))
                if safe_opt
                else "",
                "risky_opt_risky_rate": fmt(finite_mean(risky_opt, "risky_action"))
                if risky_opt
                else "",
            }
        )
    return out


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row.get(col, "") for col in columns) + " |")
    return "\n".join(lines)


def key_cases(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if row["stage"] != "risky" or row["seed"] != "0":
            continue
        if row["p_success"] not in {"0.1", "0.4"}:
            continue
        family = method_family(row["method"])
        if family not in {"trl_raw_realized", "bellman_matched", "sto_trl_matched"}:
            continue
        out.append(
            {
                "method": family,
                "L": row["safe_length"],
                "p": row["p_success"],
                "action": row["start_action"],
                "success": row["success_rate"],
                "regret": fmt(float(row["regret"]), 4),
                "pred_safe": fmt(float(row["pred_start_safe"])),
                "pred_risky": fmt(float(row["pred_start_risky"])),
                "true_risky": fmt(float(row["true_start_risky"])),
            }
        )
    return sorted(out, key=lambda r: (int(r["L"]), float(r["p"]), r["method"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/fast_tabular_compact_report.md"))
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv_path.open()))
    det = aggregate(rows, "deterministic")
    risky = aggregate(rows, "risky")
    cases = key_cases(rows)
    columns = [
        "method",
        "n",
        "success",
        "regret",
        "long_mse",
        "risky_over",
        "safe_opt_risky_rate",
        "risky_opt_risky_rate",
    ]
    case_columns = [
        "method",
        "L",
        "p",
        "action",
        "success",
        "regret",
        "pred_safe",
        "pred_risky",
        "true_risky",
    ]
    text = "\n\n".join(
        [
            "# Fast Tabular Stochastic TRL Report",
            "Source CSV: `" + str(args.csv_path) + "`",
            "## Deterministic Chain",
            markdown_table(det, columns),
            "## Risky Shortcut Aggregate",
            markdown_table(risky, columns),
            "## Safe-Optimal Key Cases",
            markdown_table(cases, case_columns),
        ]
    )
    args.out.write_text(text + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

