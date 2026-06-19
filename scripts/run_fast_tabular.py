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

from sto_trl.data import (
    coverage_summary,
    estimate_transitions,
    generate_chain_start_goal_data,
    generate_risky_data,
)
from sto_trl.dp import optimal_reachability
from sto_trl.envs import deterministic_chain, risky_shortcut
from sto_trl.learners import (
    recommended_transitive_iters,
    train_mc,
    train_model_value,
    train_trl_realized,
)
from sto_trl.metrics import start_goal_metrics, value_metrics


DEFAULT_METHODS = [
    "mc_positive",
    "mc_all_goals",
    "trl_raw_realized",
    "trl_log_realized",
    "bellman_matched",
    "sto_trl_matched",
    "bellman_full",
]


def _wants(args, method: str) -> bool:
    return method in set(args.methods)


def _row(method: str, env_name: str, seed: int, q, q_star, mdp, gamma: float, args, extra):
    metrics = {}
    metrics.update(value_metrics(q, q_star, gamma, triangle_samples=args.triangle_samples))
    metrics.update(
        start_goal_metrics(
            q,
            q_star,
            mdp.transitions,
            gamma,
            mdp.start_state,
            mdp.goal_state,
            mdp.max_steps,
            rollout_seed=10_000 + seed,
            rollout_episodes=args.rollout_episodes,
        )
    )
    metrics.update(extra)
    metrics.update({"method": method, "env": env_name, "seed": seed, "gamma": gamma})
    return metrics


def run_deterministic(args) -> list[dict[str, float | str]]:
    gamma = args.gamma
    rows = []
    for seed in args.seeds:
        print(f"[deterministic] seed={seed}", flush=True)
        mdp = deterministic_chain(args.chain_n)
        q_star = optimal_reachability(mdp.transitions, gamma)
        data = generate_chain_start_goal_data(mdp, repeats=64, noise=0.0, seed=seed)
        p_hat = estimate_transitions(data, mdp.n_states, mdp.n_actions)
        cov = coverage_summary(data, mdp.n_states, mdp.n_actions, mdp.goal_state)
        tr_iters = recommended_transitive_iters(args.chain_n)
        learners = {}
        if _wants(args, "mc_positive"):
            learners["mc_positive"] = train_mc(data, mdp.n_states, mdp.n_actions, gamma, False)
        if _wants(args, "mc_all_goals"):
            learners["mc_all_goals"] = train_mc(data, mdp.n_states, mdp.n_actions, gamma, True)
        if _wants(args, "trl_raw_realized"):
            learners["trl_raw_realized"] = train_trl_realized(
                data, mdp.n_states, mdp.n_actions, gamma, "raw", seed=seed
            )
        if _wants(args, "trl_log_realized"):
            learners["trl_log_realized"] = train_trl_realized(
                data, mdp.n_states, mdp.n_actions, gamma, "log", seed=seed
            )
        if _wants(args, "bellman_matched"):
            learners[f"bellman_{tr_iters}_sweeps"] = train_model_value(
                p_hat, gamma, tr_iters, use_transitive=False
            )
        if _wants(args, "sto_trl_matched"):
            learners[f"sto_trl_{tr_iters}_sweeps"] = train_model_value(
                p_hat, gamma, tr_iters, use_transitive=True
            )
        for method, q in learners.items():
            rows.append(
                _row(
                    method,
                    mdp.name,
                    seed,
                    q,
                    q_star,
                    mdp,
                    gamma,
                    args,
                    {"stage": "deterministic", "p_success": "", "safe_length": args.chain_n - 1, **cov},
                )
            )
        if args.incremental:
            write_csv(rows, args.out)
            args.summary_out.parent.mkdir(parents=True, exist_ok=True)
            args.summary_out.write_text(json.dumps(summarize(rows), indent=2, sort_keys=True))
    return rows


def run_risky(args) -> list[dict[str, float | str]]:
    gamma = args.gamma
    rows = []
    for seed in args.seeds:
        for safe_length in args.safe_lengths:
            for p_success in args.p_success:
                print(
                    f"[risky] seed={seed} safe_length={safe_length} p_success={p_success}",
                    flush=True,
                )
                mdp = risky_shortcut(safe_length=safe_length, p_success=p_success)
                q_star = optimal_reachability(mdp.transitions, gamma)
                data = generate_risky_data(
                    mdp,
                    n_trajectories=args.risky_trajectories,
                    p_choose_risky=args.behavior_risky_prob,
                    action_noise=args.action_noise,
                    seed=seed,
                )
                p_hat = estimate_transitions(data, mdp.n_states, mdp.n_actions)
                cov = coverage_summary(data, mdp.n_states, mdp.n_actions, mdp.goal_state)
                tr_iters = recommended_transitive_iters(safe_length)
                learners = {}
                if _wants(args, "mc_positive"):
                    learners["mc_positive"] = train_mc(data, mdp.n_states, mdp.n_actions, gamma, False)
                if _wants(args, "mc_all_goals"):
                    learners["mc_all_goals"] = train_mc(data, mdp.n_states, mdp.n_actions, gamma, True)
                if _wants(args, "trl_raw_realized"):
                    learners["trl_raw_realized"] = train_trl_realized(
                        data, mdp.n_states, mdp.n_actions, gamma, "raw", seed=seed
                    )
                if _wants(args, "trl_log_realized"):
                    learners["trl_log_realized"] = train_trl_realized(
                        data, mdp.n_states, mdp.n_actions, gamma, "log", seed=seed
                    )
                if _wants(args, "bellman_matched"):
                    learners[f"bellman_{tr_iters}_sweeps"] = train_model_value(
                        p_hat, gamma, tr_iters, use_transitive=False
                    )
                if _wants(args, "sto_trl_matched"):
                    learners[f"sto_trl_{tr_iters}_sweeps"] = train_model_value(
                        p_hat, gamma, tr_iters, use_transitive=True
                    )
                if _wants(args, "bellman_full") and not args.no_bellman_full:
                    learners["bellman_full"] = train_model_value(
                        p_hat, gamma, max(4 * safe_length, 50), use_transitive=False
                    )
                optimal_start_action = int(np.argmax(q_star[mdp.start_state, :, mdp.goal_state]))
                for method, q in learners.items():
                    rows.append(
                        _row(
                            method,
                            mdp.name,
                            seed,
                            q,
                            q_star,
                            mdp,
                            gamma,
                            args,
                            {
                                "stage": "risky",
                                "p_success": p_success,
                                "safe_length": safe_length,
                                "optimal_start_action": optimal_start_action,
                                **cov,
                            },
                        )
                    )
                if args.incremental:
                    write_csv(rows, args.out)
                    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
                    args.summary_out.write_text(json.dumps(summarize(rows), indent=2, sort_keys=True))
    return rows


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict[str, float | str]]) -> dict[str, object]:
    risky = [r for r in rows if r["stage"] == "risky"]
    summary: dict[str, object] = {}
    for method in sorted({str(r["method"]) for r in risky}):
        method_rows = [r for r in risky if r["method"] == method]
        safe_opt = [r for r in method_rows if int(float(r["optimal_start_action"])) == 0]
        risky_opt = [r for r in method_rows if int(float(r["optimal_start_action"])) == 1]
        summary[method] = {
            "mean_success": float(np.mean([float(r["success_rate"]) for r in method_rows])),
            "mean_regret": float(np.mean([float(r["regret"]) for r in method_rows])),
            "safe_opt_risky_action_rate": float(np.mean([float(r["risky_action"]) for r in safe_opt]))
            if safe_opt
            else None,
            "risky_opt_risky_action_rate": float(np.mean([float(r["risky_action"]) for r in risky_opt]))
            if risky_opt
            else None,
            "mean_risky_overestimate_ratio": float(
                np.mean([float(r["risky_overestimate_ratio"]) for r in method_rows])
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/fast_tabular.csv"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/fast_tabular_summary.json"))
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--chain-n", type=int, default=32)
    parser.add_argument("--safe-lengths", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--p-success", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--risky-trajectories", type=int, default=600)
    parser.add_argument("--behavior-risky-prob", type=float, default=0.5)
    parser.add_argument("--action-noise", type=float, default=0.02)
    parser.add_argument("--rollout-episodes", type=int, default=500)
    parser.add_argument("--triangle-samples", type=int, default=20_000)
    parser.add_argument("--no-bellman-full", action="store_true")
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--skip-deterministic", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, float | str]] = []
    if not args.skip_deterministic:
        rows.extend(run_deterministic(args))
    rows.extend(run_risky(args))
    write_csv(rows, args.out)
    summary = summarize(rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {len(rows)} rows to {args.out}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
