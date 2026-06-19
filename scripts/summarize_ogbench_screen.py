from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def load_last_csv_row(path: Path) -> dict[str, str]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def parse_float(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    if raw is None or raw in {"", "nan", "inf", "-inf"}:
        return float("nan")
    try:
        value = float(raw)
    except ValueError:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def method_from_flags(flags_path: Path) -> str:
    flags = json.loads(flags_path.read_text())
    agent = flags.get("agent", {})
    if isinstance(agent, dict):
        return str(agent.get("agent_name", "unknown"))
    return "unknown"


def summarize_run(run_dir: Path) -> dict[str, str | float]:
    train_path = run_dir / "train.csv"
    eval_path = run_dir / "eval.csv"
    flags_path = run_dir / "flags.json"
    train = load_last_csv_row(train_path) if train_path.exists() else {}
    eval_rows = load_csv_rows(eval_path) if eval_path.exists() else []
    eval_row = eval_rows[-1] if eval_rows else {}
    best_eval = max(
        eval_rows,
        key=lambda row: parse_float(row, "evaluation/overall_success"),
        default={},
    )
    method = method_from_flags(flags_path) if flags_path.exists() else "unknown"
    return {
        "method": method,
        "run_dir": str(run_dir),
        "train_step": train.get("step", "") if train else "",
        "eval_step": eval_row.get("step", ""),
        "best_eval_step": best_eval.get("step", ""),
        "train_q_loss": parse_float(train, "training/critic/q_loss") if train else float("nan"),
        "train_actor_loss": parse_float(train, "training/actor/actor_loss") if train else float("nan"),
        "eval_overall_success": parse_float(eval_row, "evaluation/overall_success"),
        "best_eval_overall_success": parse_float(best_eval, "evaluation/overall_success"),
        "eval_episode_success": parse_float(eval_row, "evaluation/overall_episode.success"),
        "eval_return": parse_float(eval_row, "evaluation/overall_episode.return"),
        "eval_length": parse_float(eval_row, "evaluation/overall_episode.length"),
    }


def fmt(value: str | float) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if math.isnan(value):
        return ""
    return f"{value:.4f}"


def markdown_table(rows: list[dict[str, str | float]]) -> str:
    columns = [
        "method",
        "train_step",
        "eval_step",
        "eval_overall_success",
        "best_eval_step",
        "best_eval_overall_success",
        "eval_episode_success",
        "eval_return",
        "eval_length",
        "train_q_loss",
        "train_actor_loss",
        "run_dir",
    ]
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("results/ogbench_screen_exp"))
    parser.add_argument("--out", type=Path, default=Path("results/ogbench_screen_report.md"))
    args = parser.parse_args()

    run_dirs = [
        path.parent
        for path in sorted(args.root.rglob("flags.json"))
        if (path.parent / "train.csv").exists() and (path.parent / "eval.csv").exists()
    ]
    rows = [summarize_run(path) for path in run_dirs]
    text = "# OGBench PointMaze Screen\n\n" + markdown_table(rows) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
