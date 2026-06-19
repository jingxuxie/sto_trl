from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_grid_shortcut import parse_grid, safe_action, start_goal_metrics
from sto_trl.data import coverage_summary, estimate_transitions, generate_grid_shortcut_data
from sto_trl.dp import optimal_reachability
from sto_trl.envs import stochastic_grid_shortcut
from sto_trl.learners import train_model_value
from sto_trl.metrics import value_metrics


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    groups: dict[tuple[str, int], list[dict[str, float | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), int(row["sweeps"])), []).append(row)

    out: list[dict[str, float | str]] = []
    for (method, sweeps), vals in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        def avg(key: str) -> float:
            return float(np.mean([float(v[key]) for v in vals]))

        out.append(
            {
                "method": method,
                "sweeps": sweeps,
                "n": len(vals),
                "success_rate": avg("success_rate"),
                "safe_action_rate": avg("safe_action_rate"),
                "portal_action_rate": avg("portal_action_rate"),
                "regret": avg("regret"),
                "pred_start_safe": avg("pred_start_safe"),
                "pred_start_portal": avg("pred_start_portal"),
                "long_horizon_mse": avg("long_horizon_mse"),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/grid_budget_curve.csv"))
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("results/grid_budget_curve_summary.json"),
    )
    parser.add_argument("--grid", type=parse_grid, default=(16, 8))
    parser.add_argument("--p-success", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--trajectories", type=int, default=1_200)
    parser.add_argument("--behavior-portal-prob", type=float, default=0.5)
    parser.add_argument("--action-noise", type=float, default=0.01)
    parser.add_argument("--rollout-episodes", type=int, default=500)
    parser.add_argument("--triangle-samples", type=int, default=0)
    parser.add_argument("--bellman-budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 96, 128])
    parser.add_argument("--sto-budgets", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()

    width, height = args.grid
    rows: list[dict[str, float | str]] = []
    for seed in args.seeds:
        mdp = stochastic_grid_shortcut(width=width, height=height, p_success=args.p_success)
        path_length = mdp.goal_state - mdp.start_state
        safe = safe_action(mdp)
        portal = mdp.n_actions - 1
        print(
            f"[budget] seed={seed} grid={width}x{height} p={args.p_success} path={path_length}",
            flush=True,
        )
        q_star = optimal_reachability(mdp.transitions, args.gamma)
        data = generate_grid_shortcut_data(
            mdp,
            n_trajectories=args.trajectories,
            p_choose_portal=args.behavior_portal_prob,
            action_noise=args.action_noise,
            seed=seed,
        )
        p_hat = estimate_transitions(data, mdp.n_states, mdp.n_actions)
        cov = coverage_summary(data, mdp.n_states, mdp.n_actions, mdp.goal_state)

        specs = [("bellman", budget, False) for budget in args.bellman_budgets]
        specs.extend(("sto_trl", budget, True) for budget in args.sto_budgets)
        for method, sweeps, use_transitive in specs:
            print(f"[solve] seed={seed} method={method} sweeps={sweeps}", flush=True)
            q = train_model_value(p_hat, args.gamma, sweeps, use_transitive=use_transitive)
            metrics = value_metrics(
                q,
                q_star,
                args.gamma,
                triangle_samples=args.triangle_samples,
            )
            metrics.update(
                start_goal_metrics(
                    q,
                    q_star,
                    mdp.transitions,
                    args.gamma,
                    mdp.start_state,
                    mdp.goal_state,
                    safe,
                    portal,
                    mdp.max_steps,
                    rollout_seed=20_000 + seed,
                    rollout_episodes=args.rollout_episodes,
                )
            )
            row = {
                "stage": "grid_budget_curve",
                "method": method,
                "sweeps": sweeps,
                "env": mdp.name,
                "seed": seed,
                "gamma": args.gamma,
                "width": width,
                "height": height,
                "path_length": path_length,
                "p_success": args.p_success,
                "safe_action": safe,
                "portal_action": portal,
                **cov,
                **metrics,
            }
            rows.append(row)
            if args.incremental:
                write_csv(rows, args.out)
                args.summary_out.parent.mkdir(parents=True, exist_ok=True)
                args.summary_out.write_text(json.dumps(summarize(rows), indent=2, sort_keys=True))

    write_csv(rows, args.out)
    summary = summarize(rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {len(rows)} rows to {args.out}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
