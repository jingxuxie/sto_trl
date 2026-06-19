#!/usr/bin/env python3
"""Dataset-topology stochastic TRL planner for OGBench AntMaze teleport.

This is the AntMaze counterpart to the PointMaze topology diagnostic. The
planner infers a coarse stochastic cell MDP from offline transitions, solves it
with Bellman or stochastic-TRL updates, and executes the high-level policy with
a nearest-neighbor local action selector over Ant dataset transitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
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
from sto_trl.learners import train_model_value


ACTION_DELTAS = (
    (-1, 0),  # north
    (1, 0),   # south
    (0, -1),  # west
    (0, 1),   # east
)


@dataclass(frozen=True)
class Topology:
    states: tuple[tuple[int, int], ...]
    state_to_idx: dict[tuple[int, int], int]
    transitions: np.ndarray
    intended_next: np.ndarray
    jump_sources: set[tuple[int, int]]

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_actions(self) -> int:
        return int(self.transitions.shape[1])


@dataclass
class LocalTransitionIndex:
    observations: np.ndarray
    actions: np.ndarray
    all_actions: np.ndarray
    global_indices: np.ndarray
    valids: np.ndarray
    next_xy: np.ndarray
    future_xy: np.ndarray
    src_ids: np.ndarray
    next_ids: np.ndarray
    by_src: list[np.ndarray]
    by_pair: dict[tuple[int, int], np.ndarray]
    feature_scale: np.ndarray


def obs_xy(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs[:2], dtype=np.float64)


def obs_to_ij(env, obs: np.ndarray) -> tuple[int, int]:
    return tuple(env.unwrapped.xy_to_ij(obs_xy(obs)))


def build_dataset_topology(env, train: dict[str, np.ndarray], min_jump_count: int) -> Topology:
    obs = np.asarray(train["observations"], dtype=np.float32)
    valids = np.asarray(train["valids"], dtype=np.float32) > 0
    valid_idxs = np.nonzero(valids[:-1])[0]

    free: set[tuple[int, int]] = set()
    jump_counts: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
    for idx in valid_idxs:
        s = obs_to_ij(env, obs[idx])
        sp = obs_to_ij(env, obs[idx + 1])
        free.add(s)
        free.add(sp)
        if abs(s[0] - sp[0]) + abs(s[1] - sp[1]) > 1:
            jump_counts.setdefault(s, {})
            jump_counts[s][sp] = jump_counts[s].get(sp, 0.0) + 1.0

    for info in env.unwrapped.task_infos:
        free.add(tuple(info["init_ij"]))
        free.add(tuple(info["goal_ij"]))

    states = tuple(sorted(free))
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    jump_probs: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for src, counts in jump_counts.items():
        counts = {dst: count for dst, count in counts.items() if count >= min_jump_count}
        if not counts:
            continue
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

    return Topology(
        states=states,
        state_to_idx=state_to_idx,
        transitions=transitions,
        intended_next=intended_next,
        jump_sources=set(jump_probs),
    )


def nearest_state(topology: Topology, env, obs: np.ndarray) -> int:
    ij = obs_to_ij(env, obs)
    if ij in topology.state_to_idx:
        return topology.state_to_idx[ij]
    xy = obs_xy(obs)
    centers = np.asarray([env.unwrapped.ij_to_xy(state) for state in topology.states])
    return int(np.argmin(np.sum((centers - xy[None, :]) ** 2, axis=1)))


def build_local_transition_index(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
    future_horizon: int,
) -> LocalTransitionIndex:
    obs = np.asarray(train["observations"], dtype=np.float32)
    actions = np.asarray(train["actions"], dtype=np.float32)
    valids = np.asarray(train["valids"], dtype=np.float32) > 0
    valid_idxs = np.nonzero(valids[:-1])[0]

    transition_obs = obs[valid_idxs]
    transition_actions = actions[valid_idxs]
    next_xy = obs[valid_idxs + 1, :2].astype(np.float32)
    future_global = np.minimum(valid_idxs + max(1, future_horizon), len(obs) - 1)
    if future_horizon > 1:
        valid_prefix = np.concatenate([[0], np.cumsum(valids.astype(np.int32))])
        valid_counts = valid_prefix[future_global] - valid_prefix[valid_idxs]
        future_global = np.where(valid_counts >= future_horizon, future_global, valid_idxs + 1)
    future_xy = obs[future_global, :2].astype(np.float32)
    src_ids = np.asarray([nearest_state(topology, env, ob) for ob in transition_obs], dtype=np.int32)
    next_ids = np.asarray([nearest_state(topology, env, ob) for ob in obs[valid_idxs + 1]], dtype=np.int32)

    by_src_lists: list[list[int]] = [[] for _ in range(topology.n_states)]
    by_pair_lists: dict[tuple[int, int], list[int]] = {}
    for local_idx, (src, nxt) in enumerate(zip(src_ids, next_ids)):
        src_i = int(src)
        nxt_i = int(nxt)
        by_src_lists[src_i].append(local_idx)
        by_pair_lists.setdefault((src_i, nxt_i), []).append(local_idx)

    feature_scale = np.std(transition_obs, axis=0).astype(np.float32)
    feature_scale = np.maximum(feature_scale, 0.1)
    # Do not let absolute xy dominate local body-state matching.
    feature_scale[:2] = np.maximum(feature_scale[:2], 4.0)

    return LocalTransitionIndex(
        observations=transition_obs,
        actions=transition_actions,
        all_actions=actions,
        global_indices=valid_idxs.astype(np.int32),
        valids=valids,
        next_xy=next_xy,
        future_xy=future_xy,
        src_ids=src_ids,
        next_ids=next_ids,
        by_src=[np.asarray(v, dtype=np.int32) for v in by_src_lists],
        by_pair={key: np.asarray(v, dtype=np.int32) for key, v in by_pair_lists.items()},
        feature_scale=feature_scale,
    )


class AntTopologyAgent:
    def __init__(
        self,
        env,
        topology: Topology,
        q: np.ndarray,
        local_index: LocalTransitionIndex,
        mode: str,
        candidates: int,
        top_k: int,
        direction_weight: float,
        target_weight: float,
        state_weight: float,
        temperature: float,
        replay_horizon: int,
    ) -> None:
        self.env = env
        self.topology = topology
        self.q = q
        self.local = local_index
        self.mode = mode
        self.candidates = candidates
        self.top_k = top_k
        self.direction_weight = direction_weight
        self.target_weight = target_weight
        self.state_weight = state_weight
        self.temperature = temperature
        self.replay_horizon = replay_horizon
        self.previous_action_id: int | None = None
        self.replay_global_idx: int | None = None
        self.replay_remaining = 0

    def reset_episode(self) -> None:
        self.previous_action_id = None
        self.replay_global_idx = None
        self.replay_remaining = 0

    def _greedy_action_id(self, state: int, goal: int) -> int:
        values = self.q[state, :, goal].copy()
        if self.previous_action_id is not None:
            values[self.previous_action_id] += 1e-6
        action_id = int(np.argmax(values))
        self.previous_action_id = action_id
        return action_id

    def _candidate_indices(self, state: int, target: int) -> np.ndarray:
        if self.mode == "pair_knn":
            pair = self.local.by_pair.get((state, target))
            if pair is not None and len(pair) >= max(4, self.top_k):
                return pair
        src = self.local.by_src[state]
        if len(src) > 0:
            return src
        # Fallback for rare off-dataset cells: use all transitions.
        return np.arange(len(self.local.observations), dtype=np.int32)

    def _replay_action(self) -> np.ndarray | None:
        if self.replay_global_idx is None or self.replay_remaining <= 0:
            return None
        if self.replay_global_idx >= len(self.local.all_actions) or not self.local.valids[self.replay_global_idx]:
            self.replay_global_idx = None
            self.replay_remaining = 0
            return None
        action = self.local.all_actions[self.replay_global_idx]
        self.replay_global_idx += 1
        self.replay_remaining -= 1
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def sample_action(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        replay = self._replay_action()
        if replay is not None:
            return replay

        state = nearest_state(self.topology, self.env, obs)
        goal_state = nearest_state(self.topology, self.env, goal)
        action_id = self._greedy_action_id(state, goal_state)
        target_state = int(self.topology.intended_next[state, action_id])

        idxs = self._candidate_indices(state, target_state)
        if len(idxs) == 0:
            return np.zeros(self.local.actions.shape[1], dtype=np.float32)

        obs_arr = np.asarray(obs, dtype=np.float32)
        if len(idxs) > self.candidates:
            xy_d2 = np.sum((self.local.observations[idxs, :2] - obs_arr[None, :2]) ** 2, axis=1)
            local = np.argpartition(xy_d2, self.candidates - 1)[: self.candidates]
            idxs = idxs[local]

        target_xy = np.asarray(self.env.unwrapped.ij_to_xy(self.topology.states[target_state]), dtype=np.float32)
        current_xy = obs_arr[:2]
        direction = target_xy - current_xy
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            direction = direction / norm

        primitive_xy = self.local.future_xy[idxs]
        primitive_delta_xy = primitive_xy - self.local.observations[idxs, :2]
        progress = primitive_delta_xy @ direction
        target_distance = np.linalg.norm(primitive_xy - target_xy[None, :], axis=1)
        feature_d2 = np.mean(((self.local.observations[idxs] - obs_arr[None, :]) / self.local.feature_scale) ** 2, axis=1)

        if self.mode == "pair_knn":
            scores = -self.state_weight * feature_d2 - self.target_weight * target_distance
        elif self.mode == "direction_knn":
            scores = (
                self.direction_weight * progress
                - self.target_weight * target_distance
                - self.state_weight * feature_d2
            )
        else:
            raise ValueError(f"unknown action mode {self.mode}")

        k = min(self.top_k, len(idxs))
        top = np.argpartition(-scores, k - 1)[:k]
        top_scores = scores[top]
        if self.temperature <= 0:
            best_local = int(top[int(np.argmax(top_scores))])
            action = self.local.actions[idxs[best_local]]
            if self.replay_horizon > 1:
                self.replay_global_idx = int(self.local.global_indices[idxs[best_local]]) + 1
                self.replay_remaining = self.replay_horizon - 1
            return np.clip(action, -1.0, 1.0).astype(np.float32)
        else:
            logits = top_scores / self.temperature
            logits = logits - np.max(logits)
            weights = np.exp(logits)
            action = np.average(self.local.actions[idxs[top]], axis=0, weights=weights)
            if self.replay_horizon > 1:
                best_local = int(top[int(np.argmax(top_scores))])
                self.replay_global_idx = int(self.local.global_indices[idxs[best_local]]) + 1
                self.replay_remaining = self.replay_horizon - 1
            return np.clip(action, -1.0, 1.0).astype(np.float32)


def evaluate_agent(
    agent: AntTopologyAgent,
    env,
    episodes: int,
    seed: int,
    task_ids: list[int] | None,
) -> dict[str, float]:
    task_infos = env.unwrapped.task_infos
    selected = (
        list(enumerate(task_infos, start=1))
        if task_ids is None
        else [(task_id, task_infos[task_id - 1]) for task_id in task_ids]
    )
    row: dict[str, float] = {}
    task_means = []
    for task_id, _task_info in selected:
        successes = []
        for episode_idx in range(episodes):
            agent.reset_episode()
            episode_seed = seed * 1_000_000 + task_id * 10_000 + episode_idx
            np.random.seed(episode_seed)
            obs, info = env.reset(seed=episode_seed, options={"task_id": task_id})
            goal = info["goal"]
            done = False
            while not done:
                action = agent.sample_action(obs, goal)
                obs, _reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
            successes.append(float(info.get("success", info.get("episode", {}).get("success", 0.0))))
        mean_success = float(np.mean(successes))
        row[f"task{task_id}_success"] = mean_success
        task_means.append(mean_success)
        print(f"[eval] task{task_id} success={mean_success:.3f}", flush=True)
    row["overall_success"] = float(np.mean(task_means))
    return row


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    out = []
    for method in sorted({str(row["method"]) for row in rows}):
        vals = [row for row in rows if row["method"] == method]
        summary: dict[str, float | int | str] = {
            "method": method,
            "n": len(vals),
            "success_mean": float(np.mean([float(row["overall_success"]) for row in vals])),
            "success_std": float(np.std([float(row["overall_success"]) for row in vals])),
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
    parser.add_argument("--env-name", default="antmaze-teleport-navigate-v0")
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--full-iters", type=int, default=180)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--task-ids", type=int, nargs="+", default=None)
    parser.add_argument("--action-mode", choices=["direction_knn", "pair_knn"], default="direction_knn")
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--direction-weight", type=float, default=2.0)
    parser.add_argument("--target-weight", type=float, default=0.15)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--min-jump-count", type=int, default=20)
    parser.add_argument("--replay-horizon", type=int, default=1)
    parser.add_argument("--future-horizon", type=int, default=1)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bellman_matched", "sto_trl_matched", "bellman_full"],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "antmaze_topology_planner.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=ROOT / "results" / "antmaze_topology_planner_summary.json",
    )
    args = parser.parse_args()

    env, train, _val = make_env_and_datasets(args.env_name, dataset_path=None)
    topology = build_dataset_topology(env, train, args.min_jump_count)
    local_index = build_local_transition_index(env, train, topology, args.future_horizon)
    matched_iters = args.iters
    if matched_iters is None:
        matched_iters = max(1, int(math.ceil(math.log2(max(topology.n_states, 2)))))
    print(
        f"[topology] states={topology.n_states} actions={topology.n_actions} "
        f"matched_iters={matched_iters} local_transitions={len(local_index.actions)}",
        flush=True,
    )

    q_by_method = {
        "bellman_matched": train_model_value(topology.transitions, args.gamma, matched_iters, False),
        "sto_trl_matched": train_model_value(topology.transitions, args.gamma, matched_iters, True),
        "bellman_full": train_model_value(topology.transitions, args.gamma, args.full_iters, False),
    }
    iters_by_method = {
        "bellman_matched": matched_iters,
        "sto_trl_matched": matched_iters,
        "bellman_full": args.full_iters,
    }

    rows: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        for method in args.methods:
            print(f"[run] seed={seed} method={method}", flush=True)
            agent = AntTopologyAgent(
                env,
                topology,
                q_by_method[method],
                local_index,
                args.action_mode,
                args.candidates,
                args.top_k,
                args.direction_weight,
                args.target_weight,
                args.state_weight,
                args.temperature,
                args.replay_horizon,
            )
            metrics = evaluate_agent(agent, env, args.episodes, seed, args.task_ids)
            row = {
                "method": method,
                "env": args.env_name,
                "seed": seed,
                "episodes_per_task": args.episodes,
                "task_ids": "all" if args.task_ids is None else ",".join(map(str, args.task_ids)),
                "gamma": args.gamma,
                "iters": iters_by_method[method],
                "full_iters": args.full_iters,
                "n_states": topology.n_states,
                "n_actions": topology.n_actions,
                "action_mode": args.action_mode,
                "candidates": args.candidates,
                "top_k": args.top_k,
                "direction_weight": args.direction_weight,
                "target_weight": args.target_weight,
                "state_weight": args.state_weight,
                "temperature": args.temperature,
                "min_jump_count": args.min_jump_count,
                "replay_horizon": args.replay_horizon,
                "future_horizon": args.future_horizon,
                **metrics,
            }
            rows.append(row)
            write_csv(rows, args.out)
            args.summary_out.parent.mkdir(parents=True, exist_ok=True)
            args.summary_out.write_text(json.dumps(summarize(rows), indent=2, sort_keys=True))
    print(json.dumps(summarize(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
