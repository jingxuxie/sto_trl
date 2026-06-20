#!/usr/bin/env python3
"""PointMaze cell-feature neural critic screen for stochastic TRL."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
TRL_DIR = ROOT / "external" / "trl"
if str(TRL_DIR) not in sys.path:
    sys.path.insert(0, str(TRL_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agents.sto_trl_neural_toy import (
    NeuralToyConfig,
    build_state_goal_onehot_features,
    build_onehot_features,
    fit_action_head_neural_critic_to_target,
    fit_neural_critic_to_target,
    train_action_head_buffered_neural_critic_from_features,
    train_action_head_fresh_iter_neural_critic_from_features,
    train_action_head_generated_target_neural_critic_from_features,
    train_action_head_guided_rank_neural_critic_from_features,
    train_action_head_neural_critic_from_features,
    train_action_head_sampled_buffered_neural_critic_from_features,
    train_action_head_sampled_target_network_neural_critic_from_features,
    train_action_head_sampled_target_replay_neural_critic_from_features,
    train_action_head_self_buffered_neural_critic_from_features,
    train_neural_critic_from_features,
)
from envs.env_utils import make_env_and_datasets
from run_pointmaze_learned_transition_planner import (
    build_raw_obs_value_features,
    build_raw_obs_policy_features,
    build_transition_targets,
    fit_raw_obs_transition_mlp,
    fit_transition_softmax,
    transition_metrics,
)
from run_pointmaze_topology_planner import (
    Topology,
    build_dataset_topology,
    build_topology,
    evaluate,
    evaluate_model,
    write_csv,
)
from sto_trl.learners import train_model_value, train_support_transitive_value


NEURAL_METHODS = (
    "neural_bellman_td",
    "neural_sto_trl",
    "neural_support_trl",
    "neural_bellman_distill",
    "neural_sto_trl_distill",
    "neural_full_bellman_distill",
    "qhead_bellman_td",
    "qhead_sto_trl",
    "qhead_sto_trl_monotone",
    "qhead_sto_trl_self_buffered",
    "qhead_sto_trl_fresh_iter",
    "qhead_sto_trl_guided_rank",
    "qhead_sto_trl_buffered",
    "qhead_sto_trl_buffered_reset_final",
    "qhead_sto_trl_buffered_topk_reset_final",
    "qhead_sto_trl_sampled_buffered_reset_final",
    "qhead_sto_trl_sampled_target_net",
    "qhead_sto_trl_sampled_target_replay",
    "qhead_sto_trl_target_iter",
    "qhead_sto_trl_target_fit",
    "qhead_support_trl",
    "qhead_bellman_distill",
    "qhead_sto_trl_distill",
    "qhead_full_bellman_distill",
)
BOOTSTRAP_NEURAL_METHODS = ("neural_bellman_td", "neural_sto_trl", "neural_support_trl")
QHEAD_BOOTSTRAP_METHODS = {
    "qhead_bellman_td": "neural_bellman_td",
    "qhead_sto_trl": "neural_sto_trl",
    "qhead_sto_trl_monotone": "neural_sto_trl_monotone",
    "qhead_support_trl": "neural_support_trl",
}
QHEAD_BUFFERED_METHODS = {
    "qhead_sto_trl_buffered": ("neural_sto_trl", True, False, "all"),
    "qhead_sto_trl_buffered_reset_final": ("neural_sto_trl", True, True, "all"),
    "qhead_sto_trl_buffered_topk_reset_final": ("neural_sto_trl", True, True, "topk_bridge"),
    "qhead_sto_trl_target_iter": ("neural_sto_trl", False, False, "all"),
}
QHEAD_SAMPLED_BUFFERED_METHODS = {
    "qhead_sto_trl_sampled_buffered_reset_final": "neural_sto_trl",
}
QHEAD_SAMPLED_TARGET_NET_METHODS = {
    "qhead_sto_trl_sampled_target_net": "neural_sto_trl",
}
QHEAD_SAMPLED_TARGET_REPLAY_METHODS = {
    "qhead_sto_trl_sampled_target_replay": "neural_sto_trl",
}
SAMPLED_QHEAD_METHODS = {
    *QHEAD_SAMPLED_BUFFERED_METHODS,
    *QHEAD_SAMPLED_TARGET_NET_METHODS,
    *QHEAD_SAMPLED_TARGET_REPLAY_METHODS,
}
QHEAD_SELF_BUFFERED_METHODS = {
    "qhead_sto_trl_self_buffered": "neural_sto_trl",
}
QHEAD_FRESH_ITER_METHODS = {
    "qhead_sto_trl_fresh_iter": "neural_sto_trl",
}
QHEAD_GUIDED_RANK_METHODS = {
    "qhead_sto_trl_guided_rank": "neural_sto_trl",
}
QHEAD_GENERATED_TARGET_METHODS = {
    "qhead_sto_trl_target_fit": "neural_sto_trl",
}
DISTILL_NEURAL_TARGETS = {
    "neural_bellman_distill": "table_bellman_matched",
    "neural_sto_trl_distill": "table_sto_trl_matched",
    "neural_full_bellman_distill": "table_full_bellman",
}
QHEAD_DISTILL_TARGETS = {
    "qhead_bellman_distill": "table_bellman_matched",
    "qhead_sto_trl_distill": "table_sto_trl_matched",
    "qhead_full_bellman_distill": "table_full_bellman",
}
TABLE_METHODS = ("table_bellman_matched", "table_sto_trl_matched", "table_support_trl", "table_full_bellman")


def build_critic_features(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
    mode: str,
) -> np.ndarray:
    raw = build_raw_obs_value_features(env, train, topology).reshape(
        topology.n_states * topology.n_actions * topology.n_states,
        -1,
    )
    onehot = build_onehot_features(topology.n_states, topology.n_actions)
    if mode == "raw_obs":
        return raw
    if mode == "onehot":
        return onehot
    if mode == "raw_obs_onehot":
        return np.concatenate([raw, onehot], axis=-1).astype(np.float32)
    raise ValueError(f"unknown feature mode {mode}")


def build_qhead_features(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
    mode: str,
) -> np.ndarray:
    raw = build_raw_obs_policy_features(env, train, topology).reshape(
        topology.n_states * topology.n_states,
        -1,
    )
    onehot = build_state_goal_onehot_features(topology.n_states)
    if mode == "raw_obs":
        return raw
    if mode == "onehot":
        return onehot
    if mode == "raw_obs_onehot":
        return np.concatenate([raw, onehot], axis=-1).astype(np.float32)
    raise ValueError(f"unknown feature mode {mode}")


def action_agreement(q: np.ndarray, target: np.ndarray) -> float:
    pred_actions = np.argmax(q, axis=1)
    target_actions = np.argmax(target, axis=1)
    mask = ~np.eye(q.shape[0], dtype=bool)
    return float(np.mean((pred_actions == target_actions)[mask])) if np.any(mask) else 1.0


def q_metrics(q: np.ndarray, target_q: np.ndarray, full_q: np.ndarray) -> dict[str, float]:
    return {
        "value_mse_to_target": float(np.mean((q - target_q) ** 2)),
        "value_max_abs_to_target": float(np.max(np.abs(q - target_q))),
        "action_agreement_to_target": action_agreement(q, target_q),
        "value_mse_to_full": float(np.mean((q - full_q) ** 2)),
        "value_max_abs_to_full": float(np.max(np.abs(q - full_q))),
        "action_agreement_to_full": action_agreement(q, full_q),
    }


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    out = []
    for method in sorted({str(row["method"]) for row in rows}):
        vals = [row for row in rows if row["method"] == method]
        summary: dict[str, float | int | str] = {
            "method": method,
            "n": len(vals),
            "success_mean": float(np.mean([float(row["overall_success"]) for row in vals])),
            "success_std": float(np.std([float(row["overall_success"]) for row in vals])),
            "action_agreement_to_full_mean": float(
                np.mean([float(row["action_agreement_to_full"]) for row in vals])
            ),
        }
        for task_id in range(1, 6):
            key = f"task{task_id}_success"
            task_vals = [float(row[key]) for row in vals if key in row]
            if task_vals:
                summary[f"task{task_id}_mean"] = float(np.mean(task_vals))
        out.append(summary)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="pointmaze-teleport-stitch-v0")
    parser.add_argument("--topology-source", choices=["env", "dataset"], default="dataset")
    parser.add_argument(
        "--transition-model",
        choices=["topology", "table_softmax", "raw_obs_mlp"],
        default="raw_obs_mlp",
    )
    parser.add_argument(
        "--transition-target-source",
        choices=["topology", "topology_samples", "dataset_counts", "dataset_cell_changes"],
        default="dataset_cell_changes",
    )
    parser.add_argument("--samples-per-row", type=int, default=50)
    parser.add_argument("--dataset-prior-weight", type=float, default=0.0)
    parser.add_argument("--transition-steps", type=int, default=2_000)
    parser.add_argument("--transition-lr", type=float, default=0.3)
    parser.add_argument("--transition-mlp-lr", type=float, default=3e-3)
    parser.add_argument("--transition-hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--transition-seed", type=int, default=0)
    parser.add_argument("--transition-log-interval", type=int, default=0)
    parser.add_argument("--feature-mode", choices=["raw_obs", "onehot", "raw_obs_onehot"], default="raw_obs_onehot")
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--full-iters", type=int, default=180)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--steps-per-iter", type=int, default=250)
    parser.add_argument("--value-hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--value-lr", type=float, default=3e-3)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--diag-weight", type=float, default=4.0)
    parser.add_argument("--rank-ce-weight", type=float, default=0.05)
    parser.add_argument("--value-seed", type=int, default=0)
    parser.add_argument("--value-log-interval", type=int, default=0)
    parser.add_argument("--distill-steps", type=int, default=None)
    parser.add_argument("--buffered-final-steps", type=int, default=0)
    parser.add_argument("--target-buffer-waypoints", type=int, default=8)
    parser.add_argument("--sampled-target-next-samples", type=int, default=32)
    parser.add_argument("--sampled-target-waypoint-candidates", type=int, default=16)
    parser.add_argument("--sampled-target-waypoints", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--task-ids", type=int, nargs="+", default=None)
    parser.add_argument("--action-scale", type=float, default=0.2)
    parser.add_argument("--eval-mode", choices=["model", "env"], default="model")
    parser.add_argument("--model-rollout-mode", choices=["exact", "mc"], default="exact")
    parser.add_argument("--model-max-steps", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--profile-eval", action="store_true")
    parser.add_argument("--methods", nargs="+", default=[*NEURAL_METHODS, *TABLE_METHODS])
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "pointmaze_neural_critic.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=ROOT / "results" / "pointmaze_neural_critic_summary.json",
    )
    args = parser.parse_args()

    env, train, _val = make_env_and_datasets(args.env_name, dataset_path=None)
    topology = build_topology(env) if args.topology_source == "env" else build_dataset_topology(env, train)
    matched_iters = args.iters
    if matched_iters is None:
        matched_iters = max(1, int(math.ceil(math.log2(max(topology.n_states, 2)))))
    print(
        f"[topology] states={topology.n_states} actions={topology.n_actions} "
        f"matched_iters={matched_iters}",
        flush=True,
    )

    target, fitted_mask = build_transition_targets(
        env,
        train,
        topology,
        args.transition_target_source,
        args.dataset_prior_weight,
        args.samples_per_row,
        args.transition_seed,
    )
    if args.transition_model == "topology":
        learned_transitions = target.copy()
        fit_stats = transition_metrics(target, target, fitted_mask, args.transition_target_source, 0.0)
    elif args.transition_model == "table_softmax":
        learned_transitions, fit_stats = fit_transition_softmax(
            target,
            fitted_mask,
            args.transition_steps,
            args.transition_lr,
            args.transition_seed,
            args.transition_log_interval,
        )
    elif args.transition_model == "raw_obs_mlp":
        learned_transitions, fit_stats = fit_raw_obs_transition_mlp(
            env,
            train,
            topology,
            target,
            fitted_mask,
            args.transition_steps,
            args.transition_mlp_lr,
            args.transition_seed,
            tuple(args.transition_hidden_dims),
            args.transition_log_interval,
        )
    else:
        raise ValueError(f"unknown transition model {args.transition_model}")
    row_sums = learned_transitions.sum(axis=-1, keepdims=True)
    learned_transitions = np.divide(
        learned_transitions,
        row_sums,
        out=np.zeros_like(learned_transitions),
        where=row_sums > 0.0,
    )
    oracle_stats = transition_metrics(
        learned_transitions,
        topology.transitions,
        np.ones(topology.transitions.shape[:2], dtype=bool),
        "topology_oracle",
        fit_stats.train_seconds,
    )
    learned_topology = Topology(
        states=topology.states,
        state_to_idx=topology.state_to_idx,
        transitions=learned_transitions,
        intended_next=topology.intended_next,
        teleport_in=topology.teleport_in,
        teleport_out=topology.teleport_out,
    )
    print(
        f"[transition] ce={fit_stats.ce:.6f} l1={fit_stats.mean_l1:.6f} "
        f"top1={fit_stats.top1_agreement:.3f} oracle_top1={oracle_stats.top1_agreement:.3f}",
        flush=True,
    )

    table_q = {
        "table_bellman_matched": train_model_value(
            learned_transitions,
            args.gamma,
            matched_iters,
            use_transitive=False,
        ),
        "table_sto_trl_matched": train_model_value(
            learned_transitions,
            args.gamma,
            matched_iters,
            use_transitive=True,
        ),
        "table_support_trl": train_support_transitive_value(
            learned_transitions,
            args.gamma,
            matched_iters,
        ),
        "table_full_bellman": train_model_value(
            learned_transitions,
            args.gamma,
            args.full_iters,
            use_transitive=False,
        ),
    }
    target_for_method = {
        "neural_bellman_td": table_q["table_bellman_matched"],
        "neural_sto_trl": table_q["table_sto_trl_matched"],
        "neural_support_trl": table_q["table_support_trl"],
        "neural_bellman_distill": table_q["table_bellman_matched"],
        "neural_sto_trl_distill": table_q["table_sto_trl_matched"],
        "neural_full_bellman_distill": table_q["table_full_bellman"],
        "qhead_bellman_td": table_q["table_bellman_matched"],
        "qhead_sto_trl": table_q["table_sto_trl_matched"],
        "qhead_sto_trl_monotone": table_q["table_sto_trl_matched"],
        "qhead_sto_trl_self_buffered": table_q["table_sto_trl_matched"],
        "qhead_sto_trl_fresh_iter": table_q["table_sto_trl_matched"],
        "qhead_sto_trl_guided_rank": table_q["table_sto_trl_matched"],
        "qhead_sto_trl_buffered": table_q["table_sto_trl_matched"],
        "qhead_sto_trl_buffered_reset_final": table_q["table_sto_trl_matched"],
        "qhead_sto_trl_buffered_topk_reset_final": table_q["table_sto_trl_matched"],
        "qhead_sto_trl_sampled_buffered_reset_final": table_q["table_sto_trl_matched"],
        "qhead_sto_trl_sampled_target_net": table_q["table_sto_trl_matched"],
        "qhead_sto_trl_sampled_target_replay": table_q["table_sto_trl_matched"],
        "qhead_sto_trl_target_iter": table_q["table_sto_trl_matched"],
        "qhead_sto_trl_target_fit": table_q["table_sto_trl_matched"],
        "qhead_support_trl": table_q["table_support_trl"],
        "qhead_bellman_distill": table_q["table_bellman_matched"],
        "qhead_sto_trl_distill": table_q["table_sto_trl_matched"],
        "qhead_full_bellman_distill": table_q["table_full_bellman"],
        **table_q,
    }

    features = build_critic_features(env, train, learned_topology, args.feature_mode)
    qhead_features = build_qhead_features(env, train, learned_topology, args.feature_mode)
    config = NeuralToyConfig(
        hidden_dims=tuple(args.value_hidden_dims),
        lr=args.value_lr,
        warmup_steps=args.warmup_steps,
        steps_per_iter=args.steps_per_iter,
        positive_weight=args.positive_weight,
        diag_weight=args.diag_weight,
        rank_ce_weight=args.rank_ce_weight,
        log_interval=args.value_log_interval,
    )

    distill_steps = args.distill_steps
    if distill_steps is None:
        distill_steps = args.warmup_steps + matched_iters * args.steps_per_iter

    q_by_method: dict[str, np.ndarray] = {}
    train_stats: dict[str, dict[str, float | int]] = {}
    wanted = set(args.methods)
    for method, q in table_q.items():
        if method in wanted:
            q_by_method[method] = q
            train_stats[method] = {"train_seconds": 0.0, "final_loss": 0.0, "operator_iters": 0}
    for method in BOOTSTRAP_NEURAL_METHODS:
        if method not in wanted:
            continue
        print(f"[value] method={method}", flush=True)
        q, stats = train_neural_critic_from_features(
            learned_transitions,
            args.gamma,
            method,
            matched_iters,
            args.value_seed,
            config,
            features,
        )
        q_by_method[method] = q
        train_stats[method] = {
            "train_seconds": stats.train_seconds,
            "final_loss": stats.final_loss,
            "operator_iters": stats.operator_iters,
        }
        print(
            f"[value] method={method} seconds={stats.train_seconds:.2f} "
            f"loss={stats.final_loss:.6f}",
            flush=True,
        )
    for method, base_method in QHEAD_GENERATED_TARGET_METHODS.items():
        if method not in wanted:
            continue
        print(f"[value] method={method} base={base_method} fit_steps={distill_steps}", flush=True)
        q, stats = train_action_head_generated_target_neural_critic_from_features(
            learned_transitions,
            args.gamma,
            base_method,
            matched_iters,
            args.value_seed,
            config,
            qhead_features,
            distill_steps,
        )
        q_by_method[method] = q
        train_stats[method] = {
            "train_seconds": stats.train_seconds,
            "final_loss": stats.final_loss,
            "operator_iters": stats.operator_iters,
        }
        print(
            f"[value] method={method} seconds={stats.train_seconds:.2f} "
            f"loss={stats.final_loss:.6f}",
            flush=True,
        )
    for method, (base_method, monotone_buffer, reset_final, waypoint_mode) in QHEAD_BUFFERED_METHODS.items():
        if method not in wanted:
            continue
        waypoint_k = args.target_buffer_waypoints if waypoint_mode != "all" else 0
        print(
            f"[value] method={method} base={base_method} "
            f"monotone={monotone_buffer} reset_final={reset_final} "
            f"waypoint_mode={waypoint_mode} waypoint_k={waypoint_k}",
            flush=True,
        )
        q, stats = train_action_head_buffered_neural_critic_from_features(
            learned_transitions,
            args.gamma,
            base_method,
            matched_iters,
            args.value_seed,
            config,
            qhead_features,
            args.buffered_final_steps,
            monotone_buffer,
            reset_final,
            waypoint_mode,
            waypoint_k,
        )
        q_by_method[method] = q
        train_stats[method] = {
            "train_seconds": stats.train_seconds,
            "final_loss": stats.final_loss,
            "operator_iters": stats.operator_iters,
        }
        print(
            f"[value] method={method} seconds={stats.train_seconds:.2f} "
            f"loss={stats.final_loss:.6f}",
            flush=True,
        )
    for method, base_method in QHEAD_SAMPLED_BUFFERED_METHODS.items():
        if method not in wanted:
            continue
        print(
            f"[value] method={method} base={base_method} "
            f"next_samples={args.sampled_target_next_samples} "
            f"waypoint_candidates={args.sampled_target_waypoint_candidates} "
            f"waypoint_k={args.sampled_target_waypoints}",
            flush=True,
        )
        q, stats = train_action_head_sampled_buffered_neural_critic_from_features(
            target,
            args.gamma,
            base_method,
            matched_iters,
            args.value_seed,
            config,
            qhead_features,
            args.buffered_final_steps,
            True,
            args.sampled_target_next_samples,
            args.sampled_target_waypoint_candidates,
            args.sampled_target_waypoints,
        )
        q_by_method[method] = q
        train_stats[method] = {
            "train_seconds": stats.train_seconds,
            "final_loss": stats.final_loss,
            "operator_iters": stats.operator_iters,
        }
        print(
            f"[value] method={method} seconds={stats.train_seconds:.2f} "
            f"loss={stats.final_loss:.6f}",
            flush=True,
        )
    for method, base_method in QHEAD_SAMPLED_TARGET_NET_METHODS.items():
        if method not in wanted:
            continue
        print(
            f"[value] method={method} base={base_method} "
            f"next_samples={args.sampled_target_next_samples} "
            f"waypoint_candidates={args.sampled_target_waypoint_candidates} "
            f"waypoint_k={args.sampled_target_waypoints}",
            flush=True,
        )
        q, stats = train_action_head_sampled_target_network_neural_critic_from_features(
            target,
            args.gamma,
            base_method,
            matched_iters,
            args.value_seed,
            config,
            qhead_features,
            args.sampled_target_next_samples,
            args.sampled_target_waypoint_candidates,
            args.sampled_target_waypoints,
            True,
        )
        q_by_method[method] = q
        train_stats[method] = {
            "train_seconds": stats.train_seconds,
            "final_loss": stats.final_loss,
            "operator_iters": stats.operator_iters,
        }
        print(
            f"[value] method={method} seconds={stats.train_seconds:.2f} "
            f"loss={stats.final_loss:.6f}",
            flush=True,
        )
    for method, base_method in QHEAD_SAMPLED_TARGET_REPLAY_METHODS.items():
        if method not in wanted:
            continue
        print(
            f"[value] method={method} base={base_method} "
            f"next_samples={args.sampled_target_next_samples} "
            f"waypoint_candidates={args.sampled_target_waypoint_candidates} "
            f"waypoint_k={args.sampled_target_waypoints}",
            flush=True,
        )
        q, stats = train_action_head_sampled_target_replay_neural_critic_from_features(
            target,
            args.gamma,
            base_method,
            matched_iters,
            args.value_seed,
            config,
            qhead_features,
            args.sampled_target_next_samples,
            args.sampled_target_waypoint_candidates,
            args.sampled_target_waypoints,
        )
        q_by_method[method] = q
        train_stats[method] = {
            "train_seconds": stats.train_seconds,
            "final_loss": stats.final_loss,
            "operator_iters": stats.operator_iters,
        }
        print(
            f"[value] method={method} seconds={stats.train_seconds:.2f} "
            f"loss={stats.final_loss:.6f}",
            flush=True,
        )
    for method, base_method in QHEAD_SELF_BUFFERED_METHODS.items():
        if method not in wanted:
            continue
        print(f"[value] method={method} base={base_method}", flush=True)
        q, stats = train_action_head_self_buffered_neural_critic_from_features(
            learned_transitions,
            args.gamma,
            base_method,
            matched_iters,
            args.value_seed,
            config,
            qhead_features,
            args.buffered_final_steps,
        )
        q_by_method[method] = q
        train_stats[method] = {
            "train_seconds": stats.train_seconds,
            "final_loss": stats.final_loss,
            "operator_iters": stats.operator_iters,
        }
        print(
            f"[value] method={method} seconds={stats.train_seconds:.2f} "
            f"loss={stats.final_loss:.6f}",
            flush=True,
        )
    for method, base_method in QHEAD_FRESH_ITER_METHODS.items():
        if method not in wanted:
            continue
        print(f"[value] method={method} base={base_method}", flush=True)
        q, stats = train_action_head_fresh_iter_neural_critic_from_features(
            learned_transitions,
            args.gamma,
            base_method,
            matched_iters,
            args.value_seed,
            config,
            qhead_features,
        )
        q_by_method[method] = q
        train_stats[method] = {
            "train_seconds": stats.train_seconds,
            "final_loss": stats.final_loss,
            "operator_iters": stats.operator_iters,
        }
        print(
            f"[value] method={method} seconds={stats.train_seconds:.2f} "
            f"loss={stats.final_loss:.6f}",
            flush=True,
        )
    for method, base_method in QHEAD_GUIDED_RANK_METHODS.items():
        if method not in wanted:
            continue
        print(f"[value] method={method} base={base_method}", flush=True)
        q, stats = train_action_head_guided_rank_neural_critic_from_features(
            learned_transitions,
            args.gamma,
            base_method,
            matched_iters,
            args.value_seed,
            config,
            qhead_features,
        )
        q_by_method[method] = q
        train_stats[method] = {
            "train_seconds": stats.train_seconds,
            "final_loss": stats.final_loss,
            "operator_iters": stats.operator_iters,
        }
        print(
            f"[value] method={method} seconds={stats.train_seconds:.2f} "
            f"loss={stats.final_loss:.6f}",
            flush=True,
        )
    for method, base_method in QHEAD_BOOTSTRAP_METHODS.items():
        if method not in wanted:
            continue
        print(f"[value] method={method} base={base_method}", flush=True)
        q, stats = train_action_head_neural_critic_from_features(
            learned_transitions,
            args.gamma,
            base_method,
            matched_iters,
            args.value_seed,
            config,
            qhead_features,
        )
        q_by_method[method] = q
        train_stats[method] = {
            "train_seconds": stats.train_seconds,
            "final_loss": stats.final_loss,
            "operator_iters": stats.operator_iters,
        }
        print(
            f"[value] method={method} seconds={stats.train_seconds:.2f} "
            f"loss={stats.final_loss:.6f}",
            flush=True,
        )
    for method, target_method in DISTILL_NEURAL_TARGETS.items():
        if method not in wanted:
            continue
        print(f"[value] method={method} target={target_method}", flush=True)
        q, stats = fit_neural_critic_to_target(
            learned_transitions,
            table_q[target_method],
            args.value_seed,
            config,
            features,
            distill_steps,
        )
        q_by_method[method] = q
        train_stats[method] = {
            "train_seconds": stats.train_seconds,
            "final_loss": stats.final_loss,
            "operator_iters": 0,
        }
        print(
            f"[value] method={method} seconds={stats.train_seconds:.2f} "
            f"loss={stats.final_loss:.6f}",
            flush=True,
        )
    for method, target_method in QHEAD_DISTILL_TARGETS.items():
        if method not in wanted:
            continue
        print(f"[value] method={method} target={target_method}", flush=True)
        q, stats = fit_action_head_neural_critic_to_target(
            learned_transitions,
            table_q[target_method],
            args.value_seed,
            config,
            qhead_features,
            distill_steps,
        )
        q_by_method[method] = q
        train_stats[method] = {
            "train_seconds": stats.train_seconds,
            "final_loss": stats.final_loss,
            "operator_iters": 0,
        }
        print(
            f"[value] method={method} seconds={stats.train_seconds:.2f} "
            f"loss={stats.final_loss:.6f}",
            flush=True,
        )

    rows: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        for method in args.methods:
            print(f"[run] seed={seed} method={method}", flush=True)
            if args.eval_mode == "model":
                eval_metrics = evaluate_model(
                    env,
                    learned_topology,
                    q_by_method[method],
                    args.episodes,
                    seed,
                    args.task_ids,
                    args.model_max_steps,
                    args.model_rollout_mode,
                )
            else:
                eval_metrics = evaluate(
                    env,
                    learned_topology,
                    q_by_method[method],
                    args.episodes,
                    seed,
                    args.action_scale,
                    args.task_ids,
                    args.profile_eval,
                    args.max_episode_steps,
                )
            metric_row = q_metrics(
                q_by_method[method],
                target_for_method[method],
                table_q["table_full_bellman"],
            )
            row = {
                "method": method,
                "family": "qhead" if method.startswith("qhead_") else "neural" if method.startswith("neural_") else "table",
                "env": args.env_name,
                "seed": seed,
                "episodes_per_task": args.episodes,
                "task_ids": "all" if args.task_ids is None else ",".join(map(str, args.task_ids)),
                "gamma": args.gamma,
                "iters": matched_iters if method != "table_full_bellman" else args.full_iters,
                "full_iters": args.full_iters,
                "n_states": learned_topology.n_states,
                "n_actions": learned_topology.n_actions,
                "feature_mode": args.feature_mode,
                "value_hidden_dims": ",".join(map(str, args.value_hidden_dims)),
                "warmup_steps": args.warmup_steps,
                "steps_per_iter": args.steps_per_iter,
                "distill_steps": distill_steps,
                "buffered_final_steps": args.buffered_final_steps,
                "target_buffer_monotone": (
                    QHEAD_BUFFERED_METHODS[method][1]
                    if method in QHEAD_BUFFERED_METHODS
                    else ""
                ),
                "target_buffer_reset_final": (
                    QHEAD_BUFFERED_METHODS[method][2]
                    if method in QHEAD_BUFFERED_METHODS
                    else ""
                ),
                "target_buffer_waypoint_mode": (
                    QHEAD_BUFFERED_METHODS[method][3]
                    if method in QHEAD_BUFFERED_METHODS
                    else ""
                ),
                "target_buffer_waypoints": (
                    args.target_buffer_waypoints
                    if method in QHEAD_BUFFERED_METHODS and QHEAD_BUFFERED_METHODS[method][3] != "all"
                    else ""
                ),
                "sampled_target_next_samples": (
                    args.sampled_target_next_samples if method in SAMPLED_QHEAD_METHODS else ""
                ),
                "sampled_target_waypoint_candidates": (
                    args.sampled_target_waypoint_candidates if method in SAMPLED_QHEAD_METHODS else ""
                ),
                "sampled_target_waypoints": (
                    args.sampled_target_waypoints if method in SAMPLED_QHEAD_METHODS else ""
                ),
                "sampled_target_update": (
                    "buffer"
                    if method in QHEAD_SAMPLED_BUFFERED_METHODS
                    else "target_network"
                    if method in QHEAD_SAMPLED_TARGET_NET_METHODS
                    else "target_replay"
                    if method in QHEAD_SAMPLED_TARGET_REPLAY_METHODS
                    else ""
                ),
                "rank_ce_weight": args.rank_ce_weight,
                "value_seed": args.value_seed,
                "transition_model": args.transition_model,
                "transition_target_source": args.transition_target_source,
                "transition_seed": args.transition_seed,
                "transition_steps": args.transition_steps,
                "transition_top1": fit_stats.top1_agreement,
                "transition_l1": fit_stats.mean_l1,
                "transition_oracle_top1": oracle_stats.top1_agreement,
                "transition_oracle_l1": oracle_stats.mean_l1,
                "eval_mode": args.eval_mode,
                "model_rollout_mode": args.model_rollout_mode if args.eval_mode == "model" else "",
                "model_max_steps": "" if args.model_max_steps is None else args.model_max_steps,
                "max_episode_steps": "" if args.max_episode_steps is None else args.max_episode_steps,
                **train_stats[method],
                **metric_row,
                **eval_metrics,
            }
            rows.append(row)
            write_csv(rows, args.out)
            args.summary_out.parent.mkdir(parents=True, exist_ok=True)
            args.summary_out.write_text(json.dumps(summarize(rows), indent=2, sort_keys=True))

    print(json.dumps(summarize(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
