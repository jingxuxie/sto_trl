#!/usr/bin/env python3
"""TRL-style flow BC local-controller evaluation for AntMaze topology planning."""

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
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
TRL_DIR = ROOT / "external" / "trl"
SCRIPT_DIR = ROOT / "scripts"
if str(TRL_DIR) not in sys.path:
    sys.path.insert(0, str(TRL_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import jax
import jax.numpy as jnp

from agents.gcfbc import GCFBCAgent, get_config as get_gcfbc_config
from envs.env_utils import make_env_and_datasets
from run_antmaze_bc_topology_planner import greedy_path, valid_training_indices
from run_antmaze_topology_planner import Topology, build_dataset_topology, nearest_state
from sto_trl.learners import train_model_value


@dataclass
class GoalObservationIndex:
    state_goal_obs: np.ndarray
    body_std: np.ndarray


@dataclass
class GCFBCPolicy:
    agent: GCFBCAgent
    rng: jax.Array
    sample_temperature: float = 1.0

    def action(self, obs: np.ndarray, goal_obs: np.ndarray) -> np.ndarray:
        self.rng, key = jax.random.split(self.rng)
        action = self.agent.sample_actions(
            observations=jnp.asarray(obs, dtype=jnp.float32),
            goals=jnp.asarray(goal_obs, dtype=jnp.float32),
            seed=key,
            temperature=self.sample_temperature,
        )
        return np.clip(np.asarray(action), -1.0, 1.0).astype(np.float32)


def build_goal_observation_index(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
    candidates_per_state: int,
) -> GoalObservationIndex:
    obs = np.asarray(train["observations"], dtype=np.float32)
    xy = obs[:, :2]
    k = max(1, int(candidates_per_state))
    state_goal_obs = []
    for state in topology.states:
        center = np.asarray(env.unwrapped.ij_to_xy(state), dtype=np.float32)
        dist_sq = np.sum((xy - center[None, :]) ** 2, axis=1)
        if k == 1:
            best = np.asarray([int(np.argmin(dist_sq))], dtype=np.int64)
        else:
            near = np.argpartition(dist_sq, kth=min(k - 1, len(dist_sq) - 1))[:k]
            best = near[np.argsort(dist_sq[near])]
        state_goal_obs.append(obs[best])
    body_std = np.maximum(obs[:, 2:].std(axis=0), 1e-3).astype(np.float32)
    return GoalObservationIndex(
        state_goal_obs=np.asarray(state_goal_obs, dtype=np.float32),
        body_std=body_std,
    )


def sample_gcfbc_batch(
    rng: np.random.Generator,
    obs: np.ndarray,
    actions: np.ndarray,
    valid_idxs: np.ndarray,
    final_locs: np.ndarray,
    batch_size: int,
    min_future: int,
    max_future: int,
) -> dict[str, np.ndarray]:
    rows = rng.integers(0, len(valid_idxs), size=batch_size)
    idxs = valid_idxs[rows]
    finals = final_locs[rows]
    offsets = rng.integers(min_future, max_future + 1, size=batch_size)
    goal_idxs = np.minimum(idxs + offsets, finals)
    return {
        "observations": obs[idxs].astype(np.float32),
        "actions": actions[idxs].astype(np.float32),
        "actor_goals": obs[goal_idxs].astype(np.float32),
    }


def train_gcfbc_policy(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    seed: int,
    steps: int,
    batch_size: int,
    min_future: int,
    max_future: int,
    hidden_dims: tuple[int, ...],
    lr: float,
    flow_steps: int,
    layer_norm: bool,
    log_interval: int,
    sample_temperature: float,
) -> GCFBCPolicy:
    obs = np.asarray(train["observations"], dtype=np.float32)
    actions = np.asarray(train["actions"], dtype=np.float32)
    valid_idxs, final_locs = valid_training_indices(train)
    val_obs = np.asarray(val["observations"], dtype=np.float32)
    val_actions = np.asarray(val["actions"], dtype=np.float32)
    val_valid_idxs, val_final_locs = valid_training_indices(val)

    rng = np.random.default_rng(seed)
    config = get_gcfbc_config()
    config["lr"] = lr
    config["batch_size"] = batch_size
    config["actor_hidden_dims"] = hidden_dims
    config["layer_norm"] = layer_norm
    config["flow_steps"] = flow_steps
    example_batch = sample_gcfbc_batch(
        rng, obs, actions, valid_idxs, final_locs, 1, min_future, max_future
    )
    agent = GCFBCAgent.create(seed, example_batch, config)
    print(
        f"[gcfbc] train_idxs={len(valid_idxs)} val_idxs={len(val_valid_idxs)} "
        f"obs_dim={obs.shape[1]} action_dim={actions.shape[1]} device={jax.devices()[0]}",
        flush=True,
    )

    for step in range(1, steps + 1):
        batch = sample_gcfbc_batch(
            rng, obs, actions, valid_idxs, final_locs, batch_size, min_future, max_future
        )
        agent, update_info = agent.update(batch)
        if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == steps):
            val_batch = sample_gcfbc_batch(
                rng,
                val_obs,
                val_actions,
                val_valid_idxs,
                val_final_locs,
                batch_size,
                min_future,
                max_future,
            )
            _val_loss, val_info = agent.total_loss(val_batch, grad_params=None)
            print(
                f"[gcfbc] step={step} train_actor_loss={float(update_info['actor/actor_loss']):.5f} "
                f"val_actor_loss={float(val_info['actor/actor_loss']):.5f}",
                flush=True,
            )
    return GCFBCPolicy(
        agent=agent,
        rng=jax.random.PRNGKey(seed + 10_000),
        sample_temperature=float(sample_temperature),
    )


class GCFBCTopologyAgent:
    def __init__(
        self,
        env,
        topology: Topology,
        q: np.ndarray,
        policy: GCFBCPolicy,
        goal_index: GoalObservationIndex,
        waypoint_lookahead: int,
        path_mode: str,
        advance_distance: float,
        goal_candidate_mode: str,
    ) -> None:
        self.env = env
        self.topology = topology
        self.q = q
        self.policy = policy
        self.goal_index = goal_index
        self.waypoint_lookahead = waypoint_lookahead
        self.path_mode = path_mode
        self.advance_distance = advance_distance
        self.goal_candidate_mode = goal_candidate_mode
        self.previous_action_id: int | None = None
        self.path: list[int] | None = None
        self.path_goal_state: int | None = None
        self.path_cursor = 0

    def reset_episode(self) -> None:
        self.previous_action_id = None
        self.path = None
        self.path_goal_state = None
        self.path_cursor = 0

    def _target_state(self, state: int, goal_state: int) -> int:
        target = state
        cur = state
        first_action_id: int | None = None
        for _ in range(max(1, self.waypoint_lookahead)):
            values = self.q[cur, :, goal_state].copy()
            if cur == state and self.previous_action_id is not None:
                values[self.previous_action_id] += 1e-6
            action_id = int(np.argmax(values))
            if first_action_id is None:
                first_action_id = action_id
            nxt = int(self.topology.intended_next[cur, action_id])
            target = nxt
            if nxt == cur or nxt == goal_state:
                break
            cur = nxt
        self.previous_action_id = first_action_id
        return target

    def _persistent_target_state(self, obs: np.ndarray, state: int, goal_state: int) -> int:
        if self.path is None or self.path_goal_state != goal_state:
            self.path = greedy_path(self.topology, self.q, state, goal_state)
            self.path_goal_state = goal_state
            self.path_cursor = 0
        assert self.path is not None
        if state in self.path:
            self.path_cursor = max(self.path_cursor, self.path.index(state))

        current_xy = np.asarray(obs[:2], dtype=np.float32)
        while self.path_cursor < len(self.path) - 1:
            next_state = self.path[min(self.path_cursor + 1, len(self.path) - 1)]
            next_xy = np.asarray(
                self.env.unwrapped.ij_to_xy(self.topology.states[next_state]), dtype=np.float32
            )
            if float(np.linalg.norm(current_xy - next_xy)) > self.advance_distance:
                break
            self.path_cursor += 1

        target_idx = min(self.path_cursor + max(1, self.waypoint_lookahead), len(self.path) - 1)
        return self.path[target_idx]

    def _waypoint_goal_obs(self, obs: np.ndarray, target_state: int) -> np.ndarray:
        candidates = np.asarray(self.goal_index.state_goal_obs[target_state], dtype=np.float32)
        if candidates.ndim == 1:
            return candidates
        if len(candidates) == 1 or self.goal_candidate_mode == "nearest_xy":
            return candidates[0]
        if self.goal_candidate_mode == "body_nearest":
            if candidates.shape[1] <= 2:
                return candidates[0]
            scale = self.goal_index.body_std
            deltas = (candidates[:, 2:] - np.asarray(obs[2:], dtype=np.float32)[None, :]) / scale[None, :]
            scores = np.mean(np.square(np.clip(deltas, -10.0, 10.0)), axis=1)
            return candidates[int(np.argmin(scores))]
        raise ValueError(f"unknown goal candidate mode {self.goal_candidate_mode}")

    def sample_action(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        state = nearest_state(self.topology, self.env, obs)
        goal_state = nearest_state(self.topology, self.env, goal)
        if state == goal_state:
            goal_obs = np.asarray(goal, dtype=np.float32)
        else:
            if self.path_mode == "persistent":
                target_state = self._persistent_target_state(obs, state, goal_state)
            elif self.path_mode == "greedy":
                target_state = self._target_state(state, goal_state)
            else:
                raise ValueError(f"unknown path mode {self.path_mode}")
            goal_obs = self._waypoint_goal_obs(obs, target_state)
        return self.policy.action(obs, goal_obs)


def evaluate_agent(
    agent: GCFBCTopologyAgent,
    env,
    episodes: int,
    seed: int,
    task_ids: list[int] | None,
    action_repeat: int,
    max_episode_steps: int | None,
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
        lengths = []
        final_dists = []
        final_xys = []
        for episode_idx in range(episodes):
            agent.reset_episode()
            episode_seed = seed * 1_000_000 + task_id * 10_000 + episode_idx
            np.random.seed(episode_seed)
            obs, info = env.reset(seed=episode_seed, options={"task_id": task_id})
            goal = info["goal"]
            done = False
            length = 0
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            repeat_left = 0
            while not done:
                if repeat_left <= 0:
                    action = agent.sample_action(obs, goal)
                    repeat_left = max(1, action_repeat)
                obs, _reward, terminated, truncated, info = env.step(action)
                repeat_left -= 1
                done = bool(terminated or truncated)
                length += 1
                if max_episode_steps is not None and length >= max_episode_steps:
                    done = True
            successes.append(float(info.get("success", info.get("episode", {}).get("success", 0.0))))
            lengths.append(length)
            final_dists.append(float(np.linalg.norm(np.asarray(obs[:2]) - np.asarray(goal[:2]))))
            final_xys.append(np.asarray(obs[:2], dtype=np.float64))
        mean_success = float(np.mean(successes))
        mean_xy = np.mean(np.stack(final_xys, axis=0), axis=0)
        row[f"task{task_id}_success"] = mean_success
        row[f"task{task_id}_length"] = float(np.mean(lengths))
        row[f"task{task_id}_final_dist"] = float(np.mean(final_dists))
        row[f"task{task_id}_final_x"] = float(mean_xy[0])
        row[f"task{task_id}_final_y"] = float(mean_xy[1])
        task_means.append(mean_success)
        print(
            f"[eval] task{task_id} success={mean_success:.3f} "
            f"length={np.mean(lengths):.1f} final_dist={np.mean(final_dists):.2f} "
            f"final_xy=({mean_xy[0]:.2f},{mean_xy[1]:.2f})",
            flush=True,
        )
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
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--task-ids", type=int, nargs="+", default=None)
    parser.add_argument("--min-jump-count", type=int, default=20)
    parser.add_argument("--gcfbc-steps", type=int, default=5000)
    parser.add_argument("--gcfbc-batch-size", type=int, default=1024)
    parser.add_argument("--gcfbc-min-future", type=int, default=1)
    parser.add_argument("--gcfbc-max-future", type=int, default=30)
    parser.add_argument("--gcfbc-hidden-dims", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--gcfbc-lr", type=float, default=3e-4)
    parser.add_argument("--gcfbc-flow-steps", type=int, default=5)
    parser.add_argument("--gcfbc-sample-temperature", type=float, default=1.0)
    parser.add_argument("--gcfbc-log-interval", type=int, default=500)
    parser.add_argument("--no-gcfbc-layer-norm", action="store_true")
    parser.add_argument("--goal-candidates-per-state", type=int, default=1)
    parser.add_argument("--goal-candidate-mode", choices=["nearest_xy", "body_nearest"], default="nearest_xy")
    parser.add_argument("--waypoint-lookahead", type=int, default=1)
    parser.add_argument("--path-mode", choices=["greedy", "persistent"], default="greedy")
    parser.add_argument("--advance-distance", type=float, default=1.5)
    parser.add_argument("--eval-action-repeat", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--print-paths", action="store_true")
    parser.add_argument("--print-paths-only", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bellman_matched", "sto_trl_matched", "bellman_full"],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "antmaze_gcfbc_topology_planner.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=ROOT / "results" / "antmaze_gcfbc_topology_planner_summary.json",
    )
    args = parser.parse_args()

    env, train, val = make_env_and_datasets(args.env_name, dataset_path=None)
    topology = build_dataset_topology(env, train, args.min_jump_count)
    matched_iters = args.iters
    if matched_iters is None:
        matched_iters = max(1, int(math.ceil(math.log2(max(topology.n_states, 2)))))
    print(
        f"[topology] states={topology.n_states} actions={topology.n_actions} "
        f"matched_iters={matched_iters}",
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

    if args.print_paths:
        task_infos = env.unwrapped.task_infos
        selected = (
            list(enumerate(task_infos, start=1))
            if args.task_ids is None
            else [(task_id, task_infos[task_id - 1]) for task_id in args.task_ids]
        )
        for task_id, info in selected:
            start_state = topology.state_to_idx[tuple(info["init_ij"])]
            goal_state = topology.state_to_idx[tuple(info["goal_ij"])]
            for method in args.methods:
                path = greedy_path(topology, q_by_method[method], start_state, goal_state)
                cells = [topology.states[state] for state in path]
                print(f"[path] task{task_id} method={method} cells={cells}", flush=True)
        if args.print_paths_only:
            return

    goal_index = build_goal_observation_index(
        env, train, topology, args.goal_candidates_per_state
    )
    policy = train_gcfbc_policy(
        train=train,
        val=val,
        seed=0,
        steps=args.gcfbc_steps,
        batch_size=args.gcfbc_batch_size,
        min_future=args.gcfbc_min_future,
        max_future=args.gcfbc_max_future,
        hidden_dims=tuple(args.gcfbc_hidden_dims),
        lr=args.gcfbc_lr,
        flow_steps=args.gcfbc_flow_steps,
        layer_norm=not args.no_gcfbc_layer_norm,
        log_interval=args.gcfbc_log_interval,
        sample_temperature=args.gcfbc_sample_temperature,
    )

    rows: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        for method in args.methods:
            print(f"[run] seed={seed} method={method}", flush=True)
            agent = GCFBCTopologyAgent(
                env,
                topology,
                q_by_method[method],
                policy,
                goal_index,
                args.waypoint_lookahead,
                args.path_mode,
                args.advance_distance,
                args.goal_candidate_mode,
            )
            metrics = evaluate_agent(
                agent,
                env,
                args.episodes,
                seed,
                args.task_ids,
                args.eval_action_repeat,
                args.max_episode_steps,
            )
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
                "min_jump_count": args.min_jump_count,
                "gcfbc_steps": args.gcfbc_steps,
                "gcfbc_batch_size": args.gcfbc_batch_size,
                "gcfbc_min_future": args.gcfbc_min_future,
                "gcfbc_max_future": args.gcfbc_max_future,
                "gcfbc_hidden_dims": ",".join(map(str, args.gcfbc_hidden_dims)),
                "gcfbc_lr": args.gcfbc_lr,
                "gcfbc_flow_steps": args.gcfbc_flow_steps,
                "gcfbc_sample_temperature": args.gcfbc_sample_temperature,
                "gcfbc_layer_norm": int(not args.no_gcfbc_layer_norm),
                "goal_candidates_per_state": args.goal_candidates_per_state,
                "goal_candidate_mode": args.goal_candidate_mode,
                "waypoint_lookahead": args.waypoint_lookahead,
                "path_mode": args.path_mode,
                "advance_distance": args.advance_distance,
                "eval_action_repeat": args.eval_action_repeat,
                "max_episode_steps": "" if args.max_episode_steps is None else args.max_episode_steps,
                **metrics,
            }
            rows.append(row)
            write_csv(rows, args.out)
            args.summary_out.parent.mkdir(parents=True, exist_ok=True)
            args.summary_out.write_text(json.dumps(summarize(rows), indent=2, sort_keys=True))
    print(json.dumps(summarize(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
