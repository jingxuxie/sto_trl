from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sto_trl.data import coverage_summary, estimate_transitions, generate_grid_shortcut_data
from sto_trl.dp import greedy_policy, optimal_reachability, policy_reachability, rollout_policy_success
from sto_trl.envs import stochastic_grid_shortcut
from sto_trl.learners import (
    recommended_transitive_iters,
    train_mc,
    train_model_value,
    train_trl_realized,
)
from sto_trl.metrics import value_metrics


DEFAULT_METHODS = [
    "mc_positive",
    "mc_all_goals",
    "trl_raw_realized",
    "trl_log_realized",
    "bellman_matched",
    "sto_trl_matched",
    "bellman_full",
]


def parse_grid(value: str) -> tuple[int, int]:
    if "x" not in value:
        raise argparse.ArgumentTypeError("grid size must look like WIDTHxHEIGHT")
    width_s, height_s = value.lower().split("x", 1)
    return int(width_s), int(height_s)


def safe_action(mdp) -> int:
    start = mdp.start_state
    for action in range(mdp.n_actions):
        probs = mdp.transitions[start, action]
        if int(np.argmax(probs)) == start + 1 and probs[start + 1] >= 1.0 - 1e-12:
            return action
    raise ValueError("could not identify deterministic safe start action")


def start_goal_metrics(
    q: np.ndarray,
    q_star: np.ndarray,
    transitions: np.ndarray,
    gamma: float,
    start: int,
    goal: int,
    safe: int,
    portal: int,
    max_steps: int,
    rollout_seed: int,
    rollout_episodes: int,
) -> dict[str, float]:
    policy = greedy_policy(q, goal)
    v_pi = policy_reachability(transitions, policy, gamma, goal)
    v_star = q_star.max(axis=1)[:, goal]
    success, median_steps = rollout_policy_success(
        transitions,
        policy,
        start,
        goal,
        max_steps=max_steps,
        episodes=rollout_episodes,
        seed=rollout_seed,
    )
    safe_q = float(q[start, safe, goal])
    portal_q = float(q[start, portal, goal])
    safe_star = float(q_star[start, safe, goal])
    portal_star = float(q_star[start, portal, goal])
    return {
        "pred_start_safe": safe_q,
        "pred_start_portal": portal_q,
        "true_start_safe": safe_star,
        "true_start_portal": portal_star,
        "start_value": float(v_pi[start]),
        "optimal_start_value": float(v_star[start]),
        "regret": float(v_star[start] - v_pi[start]),
        "success_rate": float(success),
        "median_steps": float(median_steps),
        "start_action": float(policy[start]),
        "safe_action_rate": float(policy[start] == safe),
        "portal_action_rate": float(policy[start] == portal),
        "portal_overestimate_ratio": float(portal_q / max(portal_star, 1e-12)),
    }


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | str]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        vals = [row for row in rows if row["method"] == method]
        summary[method] = {
            "mean_success": float(np.mean([float(row["success_rate"]) for row in vals])),
            "mean_regret": float(np.mean([float(row["regret"]) for row in vals])),
            "safe_action_rate": float(np.mean([float(row["safe_action_rate"]) for row in vals])),
            "portal_action_rate": float(np.mean([float(row["portal_action_rate"]) for row in vals])),
            "mean_portal_overestimate_ratio": float(
                np.mean([float(row["portal_overestimate_ratio"]) for row in vals])
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/grid_shortcut.csv"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/grid_shortcut_summary.json"))
    parser.add_argument("--grids", type=parse_grid, nargs="+", default=[(8, 4), (16, 4), (16, 8)])
    parser.add_argument("--p-success", type=float, nargs="+", default=[0.02, 0.05])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--trajectories", type=int, default=1_200)
    parser.add_argument("--behavior-portal-prob", type=float, default=0.5)
    parser.add_argument("--action-noise", type=float, default=0.01)
    parser.add_argument("--rollout-episodes", type=int, default=500)
    parser.add_argument("--triangle-samples", type=int, default=10_000)
    parser.add_argument("--full-sweep-factor", type=int, default=4)
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    args = parser.parse_args()

    wanted = set(args.methods)
    rows: list[dict[str, float | str]] = []
    for seed in args.seeds:
        for width, height in args.grids:
            for p_success in args.p_success:
                mdp = stochastic_grid_shortcut(width=width, height=height, p_success=p_success)
                path_length = mdp.goal_state - mdp.start_state
                matched_iters = recommended_transitive_iters(path_length)
                safe = safe_action(mdp)
                portal = mdp.n_actions - 1
                print(
                    f"[grid] seed={seed} grid={width}x{height} p={p_success} "
                    f"path={path_length} sweeps={matched_iters}",
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
                learners = {}
                if "mc_positive" in wanted:
                    learners["mc_positive"] = train_mc(data, mdp.n_states, mdp.n_actions, args.gamma, False)
                if "mc_all_goals" in wanted:
                    learners["mc_all_goals"] = train_mc(data, mdp.n_states, mdp.n_actions, args.gamma, True)
                if "trl_raw_realized" in wanted:
                    learners["trl_raw_realized"] = train_trl_realized(
                        data, mdp.n_states, mdp.n_actions, args.gamma, "raw", seed=seed
                    )
                if "trl_log_realized" in wanted:
                    learners["trl_log_realized"] = train_trl_realized(
                        data, mdp.n_states, mdp.n_actions, args.gamma, "log", seed=seed
                    )
                if "bellman_matched" in wanted:
                    learners[f"bellman_{matched_iters}_sweeps"] = train_model_value(
                        p_hat, args.gamma, matched_iters, use_transitive=False
                    )
                if "sto_trl_matched" in wanted:
                    learners[f"sto_trl_{matched_iters}_sweeps"] = train_model_value(
                        p_hat, args.gamma, matched_iters, use_transitive=True
                    )
                if "bellman_full" in wanted:
                    learners["bellman_full"] = train_model_value(
                        p_hat,
                        args.gamma,
                        max(args.full_sweep_factor * path_length, 50),
                        use_transitive=False,
                    )

                optimal_action = int(np.argmax(q_star[mdp.start_state, :, mdp.goal_state]))
                for method, q in learners.items():
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
                            rollout_seed=10_000 + seed,
                            rollout_episodes=args.rollout_episodes,
                        )
                    )
                    row = {
                        "stage": "grid_shortcut",
                        "method": method,
                        "env": mdp.name,
                        "seed": seed,
                        "gamma": args.gamma,
                        "width": width,
                        "height": height,
                        "path_length": path_length,
                        "p_success": p_success,
                        "matched_sweeps": matched_iters,
                        "safe_action": safe,
                        "portal_action": portal,
                        "optimal_start_action": optimal_action,
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
