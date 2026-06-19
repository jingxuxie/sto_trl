from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from itertools import product
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
TRL_DIR = ROOT / "external" / "trl"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRL_DIR) not in sys.path:
    sys.path.insert(0, str(TRL_DIR))

from envs.env_utils import make_env_and_datasets
from scripts.run_pointmaze_graph_planner import (
    GraphPlannerAgent,
    build_graph,
    evaluate_agent,
    solve_reachability,
    task_points,
)


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="pointmaze-teleport-navigate-v0")
    parser.add_argument("--cell-sizes", type=float, nargs="+", default=[1.0])
    parser.add_argument("--action-bin-levels", type=int, nargs="+", default=[3])
    parser.add_argument("--action-thresholds", type=float, nargs="+", default=[0.25])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.995])
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--polish-iters", type=int, nargs="+", default=[40])
    parser.add_argument("--full-iters", type=int, default=220)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--task-ids", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--action-modes", nargs="+", default=["mean"])
    parser.add_argument("--action-scales", type=float, nargs="+", default=[0.2])
    parser.add_argument("--action-gains", type=float, nargs="+", default=[1.0])
    parser.add_argument("--action-smoothings", type=float, nargs="+", default=[0.0])
    parser.add_argument("--transition-k", type=int, default=5)
    parser.add_argument("--transition-candidates", type=int, default=256)
    parser.add_argument("--proximity-weight", type=float, default=0.02)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bellman_polished", "sto_trl_polished", "bellman_full"],
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "pointmaze_graph_sweep.csv")
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=ROOT / "results" / "pointmaze_graph_sweep_summary.json",
    )
    args = parser.parse_args()

    env, train, _val = make_env_and_datasets(args.env_name, dataset_path=None)
    points = task_points(env)
    rows: list[dict[str, float | str]] = []

    for cell_size, bin_levels, action_threshold in product(
        args.cell_sizes, args.action_bin_levels, args.action_thresholds
    ):
        graph = build_graph(train, points, cell_size, action_threshold, bin_levels)
        matched_iters = args.iters
        if matched_iters is None:
            matched_iters = max(1, int(math.ceil(math.log2(max(graph.n_states, 2)))) + 1)
        print(
            f"[graph] cell={cell_size} bins={bin_levels} threshold={action_threshold} "
            f"states={graph.n_states} edges={graph.n_edges} matched_iters={matched_iters}",
            flush=True,
        )

        for gamma, polish_iters in product(args.gammas, args.polish_iters):
            q_cache = {}
            specs = {
                "bellman_polished": (matched_iters, False, polish_iters),
                "sto_trl_polished": (matched_iters, True, polish_iters),
                "bellman_matched": (matched_iters, False, 0),
                "sto_trl_matched": (matched_iters, True, 0),
                "bellman_full": (args.full_iters, False, 0),
            }
            for method in args.methods:
                iters, use_transitive, method_polish_iters = specs[method]
                q_key = (iters, use_transitive, method_polish_iters)
                if q_key not in q_cache:
                    print(
                        f"[solve] gamma={gamma} method={method} iters={iters} "
                        f"transitive={use_transitive} polish={method_polish_iters}",
                        flush=True,
                    )
                    q_cache[q_key] = solve_reachability(
                        graph,
                        gamma,
                        iters,
                        use_transitive,
                        args.chunk_size,
                        polish_iters=method_polish_iters,
                    )
                q = q_cache[q_key]
                for action_mode, action_scale, action_gain, action_smoothing, seed in product(
                    args.action_modes,
                    args.action_scales,
                    args.action_gains,
                    args.action_smoothings,
                    args.seeds,
                ):
                    print(
                        f"[eval] method={method} mode={action_mode} gain={action_gain} "
                        f"scale={action_scale} smoothing={action_smoothing} seed={seed}",
                        flush=True,
                    )
                    agent = GraphPlannerAgent(
                        graph,
                        q,
                        action_mode,
                        action_scale=action_scale,
                        action_gain=action_gain,
                        action_smoothing=action_smoothing,
                        transition_k=args.transition_k,
                        transition_candidates=args.transition_candidates,
                        proximity_weight=args.proximity_weight,
                    )
                    metrics = evaluate_agent(agent, env, args.episodes, seed, task_ids=args.task_ids)
                    row = {
                        "method": method,
                        "env": args.env_name,
                        "task_ids": "all" if args.task_ids is None else ",".join(map(str, args.task_ids)),
                        "episodes_per_task": args.episodes,
                        "cell_size": cell_size,
                        "action_threshold": action_threshold,
                        "action_bin_levels": bin_levels,
                        "gamma": gamma,
                        "iters": iters,
                        "polish_iters": method_polish_iters,
                        "n_states": graph.n_states,
                        "n_edges": graph.n_edges,
                        "action_mode": action_mode,
                        "action_scale": action_scale,
                        "action_gain": action_gain,
                        "action_smoothing": action_smoothing,
                        "transition_k": args.transition_k,
                        "transition_candidates": args.transition_candidates,
                        "proximity_weight": args.proximity_weight,
                        "seed": seed,
                        **metrics,
                    }
                    rows.append(row)
                    write_csv(rows, args.out)
                    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
                    args.summary_out.write_text(json.dumps(rows, indent=2, sort_keys=True))

    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
