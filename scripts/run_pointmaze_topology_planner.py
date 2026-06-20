#!/usr/bin/env python3
"""Topology-level stochastic TRL planner for OGBench PointMaze teleport.

This is a fast diagnostic that separates high-level stochastic reachability
planning from the weak empirical graph executor. It uses the maze topology as a
low-level scaffold, solves the induced stochastic cell MDP, and executes the
greedy cell policy with a proportional point-mass controller.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
TRL_DIR = ROOT / "external" / "trl"
if str(TRL_DIR) not in sys.path:
    sys.path.insert(0, str(TRL_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.env_utils import make_env_and_datasets
from sto_trl.learners import train_model_value, train_support_transitive_value


@dataclass(frozen=True)
class Topology:
    states: tuple[tuple[int, int], ...]
    state_to_idx: dict[tuple[int, int], int]
    transitions: np.ndarray
    intended_next: np.ndarray
    teleport_in: set[tuple[int, int]]
    teleport_out: tuple[tuple[int, int], ...]

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_actions(self) -> int:
        return int(self.transitions.shape[1])


ACTION_DELTAS = (
    (-1, 0),  # north
    (1, 0),   # south
    (0, -1),  # west
    (0, 1),   # east
)


def build_topology(env) -> Topology:
    unwrapped = env.unwrapped
    maze = np.asarray(unwrapped.maze_map)
    free = [
        (i, j)
        for i in range(maze.shape[0])
        for j in range(maze.shape[1])
        if int(maze[i, j]) == 0
    ]
    state_to_idx = {state: idx for idx, state in enumerate(free)}
    teleport_info = unwrapped._teleport_info
    teleport_in = set(tuple(x) for x in teleport_info["teleport_in_ijs"])
    teleport_out = tuple(tuple(x) for x in teleport_info["teleport_out_ijs"])
    out_ids = [state_to_idx[state] for state in teleport_out]

    transitions = np.zeros((len(free), len(ACTION_DELTAS), len(free)), dtype=np.float64)
    intended_next = np.zeros((len(free), len(ACTION_DELTAS)), dtype=np.int64)
    for idx, state in enumerate(free):
        for action, (di, dj) in enumerate(ACTION_DELTAS):
            if state in teleport_in:
                for out_id in out_ids:
                    transitions[idx, action, out_id] += 1.0 / len(out_ids)
                intended_next[idx, action] = idx
                continue

            candidate = (state[0] + di, state[1] + dj)
            if candidate not in state_to_idx:
                candidate = state
            intended_next[idx, action] = state_to_idx[candidate]

            if candidate in teleport_in:
                for out_id in out_ids:
                    transitions[idx, action, out_id] += 1.0 / len(out_ids)
            else:
                transitions[idx, action, state_to_idx[candidate]] = 1.0

    return Topology(
        states=tuple(free),
        state_to_idx=state_to_idx,
        transitions=transitions,
        intended_next=intended_next,
        teleport_in=teleport_in,
        teleport_out=teleport_out,
    )


def build_dataset_topology(env, train: dict[str, np.ndarray]) -> Topology:
    unwrapped = env.unwrapped
    obs = np.asarray(train["observations"], dtype=np.float32)
    valids = np.asarray(train["valids"], dtype=np.float32) > 0
    valid_idxs = np.nonzero(valids[:-1])[0]

    def obs_to_ij(x: np.ndarray) -> tuple[int, int]:
        return tuple(unwrapped.xy_to_ij(np.asarray(x, dtype=np.float64)))

    free: set[tuple[int, int]] = set()
    jump_counts: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
    for idx in valid_idxs:
        s = obs_to_ij(obs[idx])
        sp = obs_to_ij(obs[idx + 1])
        free.add(s)
        free.add(sp)
        if abs(s[0] - sp[0]) + abs(s[1] - sp[1]) > 1:
            jump_counts.setdefault(s, {})
            jump_counts[s][sp] = jump_counts[s].get(sp, 0.0) + 1.0

    for info in unwrapped.task_infos:
        free.add(tuple(info["init_ij"]))
        free.add(tuple(info["goal_ij"]))

    states = tuple(sorted(free))
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    jump_probs: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for src, counts in jump_counts.items():
        total = sum(counts.values())
        jump_probs[src] = [
            (state_to_idx[dst], count / total)
            for dst, count in sorted(counts.items())
            if dst in state_to_idx
        ]

    transitions = np.zeros((len(states), len(ACTION_DELTAS), len(states)), dtype=np.float64)
    intended_next = np.zeros((len(states), len(ACTION_DELTAS)), dtype=np.int64)
    for idx, state in enumerate(states):
        for action, (di, dj) in enumerate(ACTION_DELTAS):
            if state in jump_probs:
                for out_id, prob in jump_probs[state]:
                    transitions[idx, action, out_id] += prob
                intended_next[idx, action] = idx
                continue

            candidate = (state[0] + di, state[1] + dj)
            if candidate not in state_to_idx:
                candidate = state
            intended_next[idx, action] = state_to_idx[candidate]
            if candidate in jump_probs:
                for out_id, prob in jump_probs[candidate]:
                    transitions[idx, action, out_id] += prob
            else:
                transitions[idx, action, state_to_idx[candidate]] = 1.0

    out_states = sorted({states[out_id] for probs in jump_probs.values() for out_id, _prob in probs})
    return Topology(
        states=states,
        state_to_idx=state_to_idx,
        transitions=transitions,
        intended_next=intended_next,
        teleport_in=set(jump_probs),
        teleport_out=tuple(out_states),
    )


def nearest_state(
    topology: Topology,
    env,
    obs: np.ndarray,
    state_centers: np.ndarray | None = None,
) -> int:
    ij = tuple(env.unwrapped.xy_to_ij(np.asarray(obs, dtype=np.float64)))
    if ij in topology.state_to_idx:
        return topology.state_to_idx[ij]
    xy = np.asarray(obs[:2], dtype=np.float64)
    centers = (
        state_centers
        if state_centers is not None
        else np.asarray([env.unwrapped.ij_to_xy(state) for state in topology.states])
    )
    return int(np.argmin(np.sum((centers - xy[None, :]) ** 2, axis=1)))


def greedy_action(q: np.ndarray, state: int, goal: int, previous_action: int | None) -> int:
    if q.ndim == 4:
        prev_idx = q.shape[1] - 1 if previous_action is None else previous_action
        values = q[state, prev_idx, :, goal].copy()
        if previous_action is not None:
            values[previous_action] += 1e-6
        return int(np.argmax(values))
    values = q[state, :, goal].copy()
    if previous_action is not None:
        values[previous_action] += 1e-6
    return int(np.argmax(values))


def evaluate(
    env,
    topology: Topology,
    q: np.ndarray,
    episodes: int,
    seed: int,
    action_scale: float,
    task_ids: list[int] | None,
    profile: bool = False,
    max_episode_steps: int | None = None,
) -> dict[str, float]:
    task_infos = env.unwrapped.task_infos
    selected = (
        list(enumerate(task_infos, start=1))
        if task_ids is None
        else [(task_id, task_infos[task_id - 1]) for task_id in task_ids]
    )
    rows: dict[str, float] = {}
    task_means = []
    state_centers = np.asarray([env.unwrapped.ij_to_xy(state) for state in topology.states], dtype=np.float64)
    eval_start = time.perf_counter()
    profile_reset_seconds = 0.0
    profile_action_seconds = 0.0
    profile_env_step_seconds = 0.0
    profile_env_steps = 0
    for task_id, _task_info in selected:
        successes = []
        lengths = []
        for episode_idx in range(episodes):
            episode_seed = seed * 1_000_000 + task_id * 10_000 + episode_idx
            np.random.seed(episode_seed)
            reset_start = time.perf_counter()
            obs, info = env.reset(seed=episode_seed, options={"task_id": task_id})
            profile_reset_seconds += time.perf_counter() - reset_start
            goal_state = nearest_state(topology, env, info["goal"], state_centers)
            previous_action: int | None = None
            done = False
            length = 0
            while not done:
                action_start = time.perf_counter()
                state = nearest_state(topology, env, obs, state_centers)
                action_id = greedy_action(q, state, goal_state, previous_action)
                previous_action = action_id
                target_state = int(topology.intended_next[state, action_id])
                target_xy = state_centers[target_state]
                action = np.clip((target_xy - np.asarray(obs[:2])) / action_scale, -1.0, 1.0)
                profile_action_seconds += time.perf_counter() - action_start
                step_start = time.perf_counter()
                obs, _reward, terminated, truncated, info = env.step(action.astype(np.float32))
                profile_env_step_seconds += time.perf_counter() - step_start
                profile_env_steps += 1
                done = bool(terminated or truncated)
                length += 1
                if max_episode_steps is not None and length >= max_episode_steps:
                    done = True
            successes.append(float(info.get("success", info.get("episode", {}).get("success", 0.0))))
            lengths.append(length)
        task_mean = float(np.mean(successes))
        rows[f"task{task_id}_success"] = task_mean
        rows[f"task{task_id}_length"] = float(np.mean(lengths))
        task_means.append(task_mean)
        print(f"[eval] task{task_id} success={task_mean:.3f} length={np.mean(lengths):.1f}", flush=True)
    rows["overall_success"] = float(np.mean(task_means))
    rows["eval_seconds"] = time.perf_counter() - eval_start
    if profile:
        rows["profile_reset_seconds"] = profile_reset_seconds
        rows["profile_action_seconds"] = profile_action_seconds
        rows["profile_env_step_seconds"] = profile_env_step_seconds
        rows["profile_env_steps"] = float(profile_env_steps)
        rows["profile_action_ms_per_step"] = 1000.0 * profile_action_seconds / max(profile_env_steps, 1)
        rows["profile_env_step_ms"] = 1000.0 * profile_env_step_seconds / max(profile_env_steps, 1)
    return rows


def env_max_episode_steps(env, fallback: int = 1000) -> int:
    for attr in ("_max_episode_steps", "max_episode_steps"):
        value = getattr(env, attr, None)
        if value is not None:
            return int(value)
    spec = getattr(env, "spec", None)
    value = getattr(spec, "max_episode_steps", None)
    return fallback if value is None else int(value)


def exact_model_task_success(
    topology: Topology,
    q: np.ndarray,
    init_state: int,
    goal_state: int,
    max_steps: int,
) -> tuple[float, float]:
    if init_state == goal_state:
        return 1.0, 0.0

    n_prev = topology.n_actions + 1
    none_prev = topology.n_actions
    n_aug = topology.n_states * n_prev
    dist = np.zeros(n_aug, dtype=np.float64)
    dist[init_state * n_prev + none_prev] = 1.0
    success = 0.0
    expected_length = 0.0

    nonterminal_transition = np.zeros((n_aug, n_aug), dtype=np.float64)
    hit_prob = np.zeros(n_aug, dtype=np.float64)
    for state in range(topology.n_states):
        for prev in range(n_prev):
            previous_action = None if prev == none_prev else int(prev)
            action_id = greedy_action(q, state, goal_state, previous_action)
            row = state * n_prev + prev
            probs = topology.transitions[state, action_id]
            hit_prob[row] = probs[goal_state]
            next_states = np.nonzero(probs > 0.0)[0]
            for next_state in next_states:
                if int(next_state) == goal_state:
                    continue
                col = int(next_state) * n_prev + action_id
                nonterminal_transition[row, col] += probs[int(next_state)]

    for _step in range(max_steps):
        survival_mass = float(np.sum(dist))
        if survival_mass <= 1e-15:
            break
        expected_length += survival_mass
        success += float(dist @ hit_prob)
        dist = dist @ nonterminal_transition
    return float(np.clip(success, 0.0, 1.0)), float(expected_length)


def evaluate_model_exact(
    env,
    topology: Topology,
    q: np.ndarray,
    task_ids: list[int] | None,
    max_steps: int,
) -> dict[str, float]:
    task_infos = env.unwrapped.task_infos
    selected = (
        list(enumerate(task_infos, start=1))
        if task_ids is None
        else [(task_id, task_infos[task_id - 1]) for task_id in task_ids]
    )
    rows: dict[str, float] = {}
    task_means = []
    eval_start = time.perf_counter()
    for task_id, task_info in selected:
        init_state = topology.state_to_idx[tuple(task_info["init_ij"])]
        goal_state = topology.state_to_idx[tuple(task_info["goal_ij"])]
        success, expected_length = exact_model_task_success(
            topology,
            q,
            init_state,
            goal_state,
            max_steps,
        )
        rows[f"task{task_id}_success"] = success
        rows[f"task{task_id}_length"] = expected_length
        task_means.append(success)
        print(
            f"[model-eval-exact] task{task_id} success={success:.3f} "
            f"expected_length={expected_length:.1f}",
            flush=True,
        )
    rows["overall_success"] = float(np.mean(task_means))
    rows["eval_seconds"] = time.perf_counter() - eval_start
    rows["eval_mode"] = "model"
    rows["model_rollout_mode"] = "exact"
    rows["model_max_steps"] = int(max_steps)
    return rows


def evaluate_model(
    env,
    topology: Topology,
    q: np.ndarray,
    episodes: int,
    seed: int,
    task_ids: list[int] | None,
    max_steps: int | None,
    rollout_mode: str = "exact",
) -> dict[str, float]:
    """Fast topology-level rollout proxy.

    This screens the high-level stochastic MDP without stepping the simulator or
    the proportional controller. Use environment evaluation for final claims.
    """
    task_infos = env.unwrapped.task_infos
    selected = (
        list(enumerate(task_infos, start=1))
        if task_ids is None
        else [(task_id, task_infos[task_id - 1]) for task_id in task_ids]
    )
    if max_steps is None:
        max_steps = env_max_episode_steps(env)
    if rollout_mode == "exact":
        return evaluate_model_exact(env, topology, q, task_ids, max_steps)
    if rollout_mode != "mc":
        raise ValueError(f"unknown model rollout mode {rollout_mode}")
    rows: dict[str, float] = {}
    task_means = []
    eval_start = time.perf_counter()
    for task_id, task_info in selected:
        init_state = topology.state_to_idx[tuple(task_info["init_ij"])]
        goal_state = topology.state_to_idx[tuple(task_info["goal_ij"])]
        successes = []
        lengths = []
        for episode_idx in range(episodes):
            episode_seed = seed * 1_000_000 + task_id * 10_000 + episode_idx
            rng = np.random.default_rng(episode_seed)
            state = init_state
            previous_action: int | None = None
            success = float(state == goal_state)
            length = 0
            for step in range(max_steps):
                if state == goal_state:
                    success = 1.0
                    break
                action_id = greedy_action(q, state, goal_state, previous_action)
                previous_action = action_id
                state = int(rng.choice(topology.n_states, p=topology.transitions[state, action_id]))
                length = step + 1
            if state == goal_state:
                success = 1.0
            successes.append(success)
            lengths.append(length)
        task_mean = float(np.mean(successes))
        rows[f"task{task_id}_success"] = task_mean
        rows[f"task{task_id}_length"] = float(np.mean(lengths))
        task_means.append(task_mean)
        print(f"[model-eval] task{task_id} success={task_mean:.3f} length={np.mean(lengths):.1f}", flush=True)
    rows["overall_success"] = float(np.mean(task_means))
    rows["eval_seconds"] = time.perf_counter() - eval_start
    rows["eval_mode"] = "model"
    rows["model_rollout_mode"] = "mc"
    rows["model_max_steps"] = int(max_steps)
    return rows


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    methods = sorted({str(row["method"]) for row in rows})
    out: list[dict[str, float | int | str]] = []
    for method in methods:
        vals = [row for row in rows if row["method"] == method]
        out_row: dict[str, float | int | str] = {
            "method": method,
            "n": len(vals),
            "success_mean": float(np.mean([float(row["overall_success"]) for row in vals])),
            "success_std": float(np.std([float(row["overall_success"]) for row in vals])),
        }
        for task_id in range(1, 6):
            key = f"task{task_id}_success"
            task_vals = [float(row[key]) for row in vals if key in row]
            if task_vals:
                out_row[f"task{task_id}_mean"] = float(np.mean(task_vals))
        out.append(out_row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="pointmaze-teleport-navigate-v0")
    parser.add_argument("--topology-source", choices=["env", "dataset"], default="env")
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--full-iters", type=int, default=180)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--task-ids", type=int, nargs="+", default=None)
    parser.add_argument("--action-scale", type=float, default=0.2)
    parser.add_argument("--eval-mode", choices=["env", "model"], default="env")
    parser.add_argument("--model-rollout-mode", choices=["exact", "mc"], default="exact")
    parser.add_argument("--model-max-steps", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--profile-eval", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bellman_matched", "sto_trl_matched", "bellman_full"],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "pointmaze_topology_planner.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=ROOT / "results" / "pointmaze_topology_planner_summary.json",
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

    q_by_method = {
        "bellman_matched": train_model_value(
            topology.transitions,
            args.gamma,
            matched_iters,
            use_transitive=False,
        ),
        "sto_trl_matched": train_model_value(
            topology.transitions,
            args.gamma,
            matched_iters,
            use_transitive=True,
        ),
        "bellman_full": train_model_value(
            topology.transitions,
            args.gamma,
            args.full_iters,
            use_transitive=False,
        ),
        "support_trl_matched": train_support_transitive_value(
            topology.transitions,
            args.gamma,
            matched_iters,
        ),
    }
    iters_by_method = {
        "bellman_matched": matched_iters,
        "sto_trl_matched": matched_iters,
        "bellman_full": args.full_iters,
        "support_trl_matched": matched_iters,
    }

    rows: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        for method in args.methods:
            print(f"[run] seed={seed} method={method}", flush=True)
            if args.eval_mode == "model":
                metrics = evaluate_model(
                    env,
                    topology,
                    q_by_method[method],
                    args.episodes,
                    seed,
                    args.task_ids,
                    args.model_max_steps,
                    args.model_rollout_mode,
                )
            else:
                metrics = evaluate(
                    env,
                    topology,
                    q_by_method[method],
                    args.episodes,
                    seed,
                    args.action_scale,
                    args.task_ids,
                    args.profile_eval,
                    args.max_episode_steps,
                )
            row = {
                "method": method,
                "env": args.env_name,
                "seed": seed,
                "episodes_per_task": args.episodes,
                "gamma": args.gamma,
                "iters": iters_by_method[method],
                "n_states": topology.n_states,
                "n_actions": topology.n_actions,
                "action_scale": args.action_scale,
                "topology_source": args.topology_source,
                "eval_mode": args.eval_mode,
                "model_rollout_mode": args.model_rollout_mode if args.eval_mode == "model" else "",
                "model_max_steps": "" if args.model_max_steps is None else args.model_max_steps,
                "max_episode_steps": "" if args.max_episode_steps is None else args.max_episode_steps,
                "task_ids": "all" if args.task_ids is None else ",".join(map(str, args.task_ids)),
                **metrics,
            }
            rows.append(row)
            write_csv(rows, args.out)
            args.summary_out.parent.mkdir(parents=True, exist_ok=True)
            args.summary_out.write_text(json.dumps(summarize(rows), indent=2, sort_keys=True))

    print(json.dumps(summarize(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
