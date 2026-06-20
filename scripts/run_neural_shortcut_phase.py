#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sto-trl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.sto_trl_neural_toy import NeuralToyConfig, train_neural_toy_critic
from sto_trl.data import coverage_summary, estimate_transitions, generate_risky_data
from sto_trl.dp import greedy_policy, optimal_reachability, policy_reachability
from sto_trl.envs import risky_shortcut
from sto_trl.learners import recommended_transitive_iters, train_model_value, train_support_transitive_value
from sto_trl.metrics import value_metrics


NEURAL_METHODS = ("neural_bellman_td", "neural_sto_trl", "neural_support_trl")
TABLE_METHODS = ("table_bellman_matched", "table_sto_trl_matched", "table_support_trl", "table_full_bellman")


def exact_policy_success(
    transitions: np.ndarray,
    policy: np.ndarray,
    start: int,
    goal: int,
    max_steps: int,
) -> tuple[float, float]:
    n_states = transitions.shape[0]
    dist = np.zeros(n_states, dtype=np.float64)
    dist[start] = 1.0
    success = 0.0
    expected_length = 0.0
    for _step in range(max_steps):
        survival = float(np.sum(dist))
        if survival <= 1e-15:
            break
        expected_length += survival
        next_dist = np.zeros_like(dist)
        active = np.nonzero(dist > 0.0)[0]
        for state in active:
            mass = dist[state]
            if state == goal:
                success += mass
                continue
            action = int(policy[state])
            probs = transitions[state, action]
            success += mass * probs[goal]
            non_goal = np.arange(n_states) != goal
            next_dist[non_goal] += mass * probs[non_goal]
        dist = next_dist
    return float(np.clip(success, 0.0, 1.0)), float(expected_length)


def evaluate_q(
    q: np.ndarray,
    q_star: np.ndarray,
    transitions: np.ndarray,
    gamma: float,
    start: int,
    goal: int,
    max_steps: int,
    triangle_samples: int,
) -> dict[str, float]:
    policy = greedy_policy(q, goal)
    value = policy_reachability(transitions, policy, gamma, goal)
    optimal_value = q_star.max(axis=1)[:, goal]
    success, expected_length = exact_policy_success(transitions, policy, start, goal, max_steps)
    metrics = value_metrics(q, q_star, gamma, triangle_samples=triangle_samples)
    metrics.update(
        {
            "pred_start_safe": float(q[start, 0, goal]),
            "pred_start_risky": float(q[start, 1, goal]),
            "true_start_safe": float(q_star[start, 0, goal]),
            "true_start_risky": float(q_star[start, 1, goal]),
            "start_action": float(policy[start]),
            "risky_action": float(policy[start] == 1),
            "success_rate": success,
            "expected_length": expected_length,
            "start_value": float(value[start]),
            "optimal_start_value": float(optimal_value[start]),
            "regret": float(optimal_value[start] - value[start]),
            "decision_correct": float(policy[start] == int(np.argmax(q_star[start, :, goal]))),
            "risky_overestimate_ratio": float(q[start, 1, goal] / max(q_star[start, 1, goal], 1e-12)),
        }
    )
    return metrics


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        vals = [row for row in rows if row["method"] == method]
        safe_opt = [row for row in vals if int(float(row["optimal_start_action"])) == 0]
        summary[method] = {
            "n": float(len(vals)),
            "success_mean": float(np.mean([float(row["success_rate"]) for row in vals])),
            "decision_correct_rate": float(np.mean([float(row["decision_correct"]) for row in vals])),
            "risky_action_rate": float(np.mean([float(row["risky_action"]) for row in vals])),
            "safe_opt_success_mean": float(np.mean([float(row["success_rate"]) for row in safe_opt]))
            if safe_opt
            else float("nan"),
            "safe_opt_risky_action_rate": float(np.mean([float(row["risky_action"]) for row in safe_opt]))
            if safe_opt
            else float("nan"),
        }
    return summary


def plot_phase(rows: list[dict[str, float | int | str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    focus = ["neural_bellman_td", "neural_sto_trl", "neural_support_trl", "table_full_bellman"]
    lengths = sorted({int(row["safe_length"]) for row in rows})
    fig, axes = plt.subplots(1, len(lengths), figsize=(4.2 * len(lengths), 3.2), sharey=True)
    if len(lengths) == 1:
        axes = [axes]
    colors = {
        "neural_bellman_td": "#4C78A8",
        "neural_sto_trl": "#2F9E44",
        "neural_support_trl": "#C92A2A",
        "table_full_bellman": "#222222",
    }
    labels = {
        "neural_bellman_td": "Neural Bellman",
        "neural_sto_trl": "Neural Sto-TRL",
        "neural_support_trl": "Neural Support-TRL",
        "table_full_bellman": "Table Full Bellman",
    }
    for ax, safe_length in zip(axes, lengths):
        for method in focus:
            vals = [
                row
                for row in rows
                if int(row["safe_length"]) == safe_length and str(row["method"]) == method
            ]
            vals.sort(key=lambda row: float(row["p_success"]))
            if not vals:
                continue
            ax.plot(
                [float(row["p_success"]) for row in vals],
                [float(row["decision_correct"]) for row in vals],
                marker="o",
                linewidth=1.8,
                color=colors[method],
                label=labels[method],
            )
        ax.set_title(f"L={safe_length}")
        ax.set_xlabel("shortcut success probability")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25, linewidth=0.8)
    axes[0].set_ylabel("correct start action")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "paper_tables" / "neural_shortcut_phase.csv")
    parser.add_argument("--summary-out", type=Path, default=ROOT / "results" / "neural_shortcut_phase_summary.json")
    parser.add_argument("--figure-out", type=Path, default=ROOT / "results" / "figures" / "neural_shortcut_phase.pdf")
    parser.add_argument("--safe-lengths", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--p-success", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--trajectories", type=int, default=2_000)
    parser.add_argument("--behavior-risky-prob", type=float, default=0.5)
    parser.add_argument("--action-noise", type=float, default=0.02)
    parser.add_argument("--transition-source", choices=["estimated", "exact"], default="estimated")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--steps-per-iter", type=int, default=120)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--diag-weight", type=float, default=4.0)
    parser.add_argument("--rank-ce-weight", type=float, default=0.05)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--triangle-samples", type=int, default=2_000)
    parser.add_argument("--methods", nargs="+", default=[*NEURAL_METHODS, *TABLE_METHODS])
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    config = NeuralToyConfig(
        hidden_dims=tuple(args.hidden_dims),
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        steps_per_iter=args.steps_per_iter,
        positive_weight=args.positive_weight,
        diag_weight=args.diag_weight,
        rank_ce_weight=args.rank_ce_weight,
        log_interval=args.log_interval,
    )
    wanted = set(args.methods)

    for seed in args.seeds:
        for safe_length in args.safe_lengths:
            for p_success in args.p_success:
                mdp = risky_shortcut(safe_length=safe_length, p_success=p_success)
                matched_iters = recommended_transitive_iters(safe_length)
                q_star = optimal_reachability(mdp.transitions, args.gamma)
                data = generate_risky_data(
                    mdp,
                    n_trajectories=args.trajectories,
                    p_choose_risky=args.behavior_risky_prob,
                    action_noise=args.action_noise,
                    seed=seed,
                )
                transitions = (
                    estimate_transitions(data, mdp.n_states, mdp.n_actions)
                    if args.transition_source == "estimated"
                    else mdp.transitions
                )
                cov = coverage_summary(data, mdp.n_states, mdp.n_actions, mdp.goal_state)
                optimal_action = int(np.argmax(q_star[mdp.start_state, :, mdp.goal_state]))
                print(
                    f"[phase] seed={seed} L={safe_length} p={p_success} "
                    f"sweeps={matched_iters} optimal_action={optimal_action}",
                    flush=True,
                )

                table_qs = {
                    "table_bellman_matched": train_model_value(
                        transitions,
                        args.gamma,
                        matched_iters,
                        use_transitive=False,
                    ),
                    "table_sto_trl_matched": train_model_value(
                        transitions,
                        args.gamma,
                        matched_iters,
                        use_transitive=True,
                    ),
                    "table_support_trl": train_support_transitive_value(
                        transitions,
                        args.gamma,
                        matched_iters,
                    ),
                    "table_full_bellman": train_model_value(
                        transitions,
                        args.gamma,
                        max(4 * safe_length, 50),
                        use_transitive=False,
                    ),
                }
                for method, q in table_qs.items():
                    if method not in wanted:
                        continue
                    metrics = evaluate_q(
                        q,
                        q_star,
                        mdp.transitions,
                        args.gamma,
                        mdp.start_state,
                        mdp.goal_state,
                        mdp.max_steps,
                        args.triangle_samples,
                    )
                    rows.append(
                        {
                            "method": method,
                            "family": "table",
                            "env": mdp.name,
                            "seed": seed,
                            "safe_length": safe_length,
                            "p_success": p_success,
                            "gamma": args.gamma,
                            "matched_sweeps": matched_iters,
                            "operator_iters": "",
                            "train_seconds": 0.0,
                            "final_loss": 0.0,
                            "transition_source": args.transition_source,
                            "optimal_start_action": optimal_action,
                            **cov,
                            **metrics,
                        }
                    )

                for method in NEURAL_METHODS:
                    if method not in wanted:
                        continue
                    q, stats = train_neural_toy_critic(
                        transitions,
                        args.gamma,
                        method,
                        matched_iters,
                        seed=seed,
                        config=config,
                    )
                    metrics = evaluate_q(
                        q,
                        q_star,
                        mdp.transitions,
                        args.gamma,
                        mdp.start_state,
                        mdp.goal_state,
                        mdp.max_steps,
                        args.triangle_samples,
                    )
                    rows.append(
                        {
                            "method": method,
                            "family": "neural",
                            "env": mdp.name,
                            "seed": seed,
                            "safe_length": safe_length,
                            "p_success": p_success,
                            "gamma": args.gamma,
                            "matched_sweeps": matched_iters,
                            "operator_iters": stats.operator_iters,
                            "train_seconds": stats.train_seconds,
                            "final_loss": stats.final_loss,
                            "transition_source": args.transition_source,
                            "optimal_start_action": optimal_action,
                            **cov,
                            **metrics,
                        }
                    )
                if args.incremental:
                    write_csv(rows, args.out)
                    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
                    args.summary_out.write_text(json.dumps(summarize(rows), indent=2, sort_keys=True))

    write_csv(rows, args.out)
    summary = summarize(rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    plot_phase(rows, args.figure_out)
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"wrote figure to {args.figure_out}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
