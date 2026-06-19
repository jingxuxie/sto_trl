from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
TRL_DIR = ROOT / "external" / "trl"
SCRIPT_DIR = ROOT / "scripts"
if str(TRL_DIR) not in sys.path:
    sys.path.insert(0, str(TRL_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from envs.env_utils import make_env_and_datasets


CACHE_VERSION = 1


@dataclass
class EmpiricalGraph:
    cell_size: float
    keys: np.ndarray
    centers: np.ndarray
    edge_src: np.ndarray
    edge_actions: np.ndarray
    edge_next_centers: np.ndarray
    edge_next_ids: list[np.ndarray]
    edge_next_probs: list[np.ndarray]
    edges_by_src: list[np.ndarray]
    transition_obs: np.ndarray
    transition_actions: np.ndarray
    transition_edge_ids: np.ndarray
    transition_src_ids: np.ndarray
    transition_next_ids: np.ndarray
    transition_indices_by_edge: list[np.ndarray]
    transition_indices_by_src: list[np.ndarray]
    key_to_id: dict[tuple[int, int], int]

    @property
    def n_states(self) -> int:
        return int(len(self.keys))

    @property
    def n_edges(self) -> int:
        return int(len(self.edge_src))

    def obs_to_cell(self, obs: np.ndarray) -> int:
        key = tuple(np.floor(np.asarray(obs) / self.cell_size + 0.5).astype(np.int32))
        cell = self.key_to_id.get(key)
        if cell is not None:
            return cell
        d2 = np.sum((self.centers - np.asarray(obs)[None, :]) ** 2, axis=1)
        return int(np.argmin(d2))


def action_bins(actions: np.ndarray, threshold: float, levels: int) -> np.ndarray:
    if levels <= 3:
        bins = np.zeros_like(actions, dtype=np.int32)
        bins[actions > threshold] = 1
        bins[actions < -threshold] = -1
        return bins
    if levels % 2 != 1:
        raise ValueError("action bin levels must be odd")
    half = levels // 2
    return np.clip(np.rint(actions * half), -half, half).astype(np.int32)


def build_graph(
    train: dict[str, np.ndarray],
    task_points: np.ndarray,
    cell_size: float,
    action_threshold: float,
    action_bin_levels: int,
) -> EmpiricalGraph:
    valid_idxs = np.nonzero(train["valids"] > 0)[0]
    obs = np.asarray(train["observations"], dtype=np.float32)
    actions = np.asarray(train["actions"], dtype=np.float32)

    src_keys_raw = np.floor(obs[valid_idxs] / cell_size + 0.5).astype(np.int32)
    next_keys_raw = np.floor(obs[valid_idxs + 1] / cell_size + 0.5).astype(np.int32)
    task_keys_raw = np.floor(task_points / cell_size + 0.5).astype(np.int32)
    all_keys = np.concatenate([src_keys_raw, next_keys_raw, task_keys_raw], axis=0)
    keys, inverse = np.unique(all_keys, axis=0, return_inverse=True)

    n_src = len(src_keys_raw)
    n_next = len(next_keys_raw)
    src_ids = inverse[:n_src]
    next_ids = inverse[n_src : n_src + n_next]
    bins = action_bins(actions[valid_idxs], action_threshold, action_bin_levels)

    edge_key_cols = np.concatenate([src_ids[:, None], bins], axis=1)
    edge_keys, edge_inverse = np.unique(edge_key_cols, axis=0, return_inverse=True)
    edge_src = edge_keys[:, 0].astype(np.int32)

    edge_counts = np.bincount(edge_inverse, minlength=len(edge_keys)).astype(np.float32)
    edge_action_sums = np.zeros((len(edge_keys), actions.shape[1]), dtype=np.float32)
    np.add.at(edge_action_sums, edge_inverse, actions[valid_idxs])
    edge_actions = edge_action_sums / np.maximum(edge_counts[:, None], 1.0)
    edge_actions = np.clip(edge_actions, -1.0, 1.0)

    pair_keys = np.stack([edge_inverse, next_ids], axis=1)
    unique_pairs, pair_counts = np.unique(pair_keys, axis=0, return_counts=True)
    edge_next_ids: list[list[int]] = [[] for _ in range(len(edge_keys))]
    edge_next_counts: list[list[float]] = [[] for _ in range(len(edge_keys))]
    for (edge_id, next_id), count in zip(unique_pairs, pair_counts):
        edge_next_ids[int(edge_id)].append(int(next_id))
        edge_next_counts[int(edge_id)].append(float(count))

    next_id_arrays = []
    next_prob_arrays = []
    edge_next_centers = np.zeros((len(edge_keys), 2), dtype=np.float32)
    for ids, counts in zip(edge_next_ids, edge_next_counts):
        ids_arr = np.asarray(ids, dtype=np.int32)
        counts_arr = np.asarray(counts, dtype=np.float32)
        probs_arr = counts_arr / counts_arr.sum()
        next_id_arrays.append(ids_arr)
        next_prob_arrays.append(probs_arr)
        edge_next_centers[len(next_id_arrays) - 1] = (
            probs_arr[:, None] * (keys[ids_arr].astype(np.float32) * cell_size)
        ).sum(axis=0)

    edges_by_src: list[list[int]] = [[] for _ in range(len(keys))]
    for edge_id, src in enumerate(edge_src):
        edges_by_src[int(src)].append(edge_id)
    edges_by_src_arrays = [np.asarray(ids, dtype=np.int32) for ids in edges_by_src]

    transition_order = np.argsort(edge_inverse, kind="stable")
    transition_counts = np.bincount(edge_inverse, minlength=len(edge_keys))
    transition_splits = np.cumsum(transition_counts)[:-1]
    transition_indices_by_edge = [
        arr.astype(np.int32)
        for arr in np.split(transition_order.astype(np.int32), transition_splits)
    ]
    src_order = np.argsort(src_ids, kind="stable")
    src_counts = np.bincount(src_ids, minlength=len(keys))
    src_splits = np.cumsum(src_counts)[:-1]
    transition_indices_by_src = [
        arr.astype(np.int32)
        for arr in np.split(src_order.astype(np.int32), src_splits)
    ]

    key_to_id = {tuple(key): i for i, key in enumerate(keys)}
    centers = keys.astype(np.float32) * cell_size
    return EmpiricalGraph(
        cell_size=cell_size,
        keys=keys,
        centers=centers,
        edge_src=edge_src,
        edge_actions=edge_actions,
        edge_next_centers=edge_next_centers,
        edge_next_ids=next_id_arrays,
        edge_next_probs=next_prob_arrays,
        edges_by_src=edges_by_src_arrays,
        transition_obs=obs[valid_idxs],
        transition_actions=actions[valid_idxs],
        transition_edge_ids=edge_inverse.astype(np.int32),
        transition_src_ids=src_ids.astype(np.int32),
        transition_next_ids=next_ids.astype(np.int32),
        transition_indices_by_edge=transition_indices_by_edge,
        transition_indices_by_src=transition_indices_by_src,
        key_to_id=key_to_id,
    )


def bellman_backup(q: np.ndarray, graph: EmpiricalGraph, gamma: float) -> np.ndarray:
    n_states = graph.n_states
    v = np.zeros((n_states, n_states), dtype=np.float32)
    np.maximum.at(v, graph.edge_src, q)
    diag = np.arange(n_states)
    v[diag, diag] = 1.0

    target = np.zeros_like(q)
    for edge_id, (next_ids, probs) in enumerate(zip(graph.edge_next_ids, graph.edge_next_probs)):
        target[edge_id] = gamma * (probs[:, None] * v[next_ids]).sum(axis=0)
    target[np.arange(graph.n_edges), graph.edge_src] = 1.0
    return np.clip(target, 0.0, 1.0)


def transitive_backup(q: np.ndarray, graph: EmpiricalGraph, chunk_size: int) -> np.ndarray:
    n_edges, n_states = q.shape
    v = np.zeros((n_states, n_states), dtype=np.float32)
    np.maximum.at(v, graph.edge_src, q)
    diag = np.arange(n_states)
    v[diag, diag] = 1.0

    target = np.zeros_like(q)
    for start in range(0, n_edges, chunk_size):
        end = min(start + chunk_size, n_edges)
        q_chunk = q[start:end].copy()
        src = graph.edge_src[start:end]
        q_chunk[np.arange(end - start), src] = 0.0
        target[start:end] = np.max(q_chunk[:, :, None] * v[None, :, :], axis=1)
        target[np.arange(start, end), src] = 1.0
    return np.clip(target, 0.0, 1.0)


def solve_reachability(
    graph: EmpiricalGraph,
    gamma: float,
    iters: int,
    use_transitive: bool,
    chunk_size: int,
    polish_iters: int = 0,
) -> np.ndarray:
    q = np.zeros((graph.n_edges, graph.n_states), dtype=np.float32)
    for edge_id, (next_ids, probs) in enumerate(zip(graph.edge_next_ids, graph.edge_next_probs)):
        q[edge_id, next_ids] += gamma * probs
    q[np.arange(graph.n_edges), graph.edge_src] = 1.0

    for _ in range(iters):
        bellman = bellman_backup(q, graph, gamma)
        if use_transitive:
            q = np.maximum(bellman, transitive_backup(q, graph, chunk_size))
        else:
            q = bellman
    for _ in range(polish_iters):
        q = bellman_backup(q, graph, gamma)
    return np.clip(q, 0.0, 1.0)


def safe_cache_token(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()


def cache_float_token(value: float) -> str:
    return safe_cache_token(f"{float(value):.8g}")


def array_digest(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def graph_digest(graph: EmpiricalGraph) -> str:
    digest = hashlib.sha256()
    for array in (
        graph.keys,
        graph.edge_src,
        graph.edge_actions,
        graph.transition_edge_ids,
        graph.transition_src_ids,
        graph.transition_next_ids,
    ):
        digest.update(array_digest(array).encode("utf-8"))
    for ids, probs in zip(graph.edge_next_ids, graph.edge_next_probs):
        digest.update(array_digest(ids).encode("utf-8"))
        digest.update(array_digest(probs).encode("utf-8"))
    return digest.hexdigest()


def cached_solve_reachability(
    graph: EmpiricalGraph,
    env_name: str,
    method: str,
    gamma: float,
    iters: int,
    use_transitive: bool,
    chunk_size: int,
    polish_iters: int,
    cache_dir: Path | None,
    use_cache: bool,
) -> tuple[np.ndarray, float, int]:
    start = time.perf_counter()
    digest = graph_digest(graph)
    meta = {
        "version": CACHE_VERSION,
        "kind": "pointmaze_graph_q",
        "env_name": env_name,
        "method": method,
        "gamma": float(gamma),
        "iters": int(iters),
        "use_transitive": bool(use_transitive),
        "chunk_size": int(chunk_size),
        "polish_iters": int(polish_iters),
        "cell_size": float(graph.cell_size),
        "n_states": int(graph.n_states),
        "n_edges": int(graph.n_edges),
        "graph_digest": digest,
    }
    cache_path = None
    if cache_dir is not None:
        cache_path = (
            cache_dir
            / (
                f"pointmaze_graph_q_v{CACHE_VERSION}_"
                f"{safe_cache_token(env_name)}_{safe_cache_token(method)}_"
                f"g{cache_float_token(gamma)}_i{iters}_p{polish_iters}_"
                f"t{int(use_transitive)}_{digest[:12]}.pkl"
            )
        )
    if use_cache and cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        if payload.get("meta") == meta:
            elapsed = time.perf_counter() - start
            print(f"[cache] q hit path={cache_path} seconds={elapsed:.2f}", flush=True)
            return np.asarray(payload["q"], dtype=np.float32), elapsed, 1
        print(f"[cache] q stale path={cache_path}", flush=True)

    q = solve_reachability(
        graph,
        gamma,
        iters,
        use_transitive,
        chunk_size,
        polish_iters=polish_iters,
    )
    elapsed = time.perf_counter() - start
    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump({"meta": meta, "q": q.astype(np.float32)}, f)
        print(f"[cache] q write path={cache_path} seconds={elapsed:.2f}", flush=True)
    return q, elapsed, 0


def load_bc_policy_for_path(path: Path) -> Any:
    from run_antmaze_bc_topology_planner import load_bc_policy

    return load_bc_policy(path)


class GraphPlannerAgent:
    def __init__(
        self,
        graph: EmpiricalGraph,
        q: np.ndarray,
        action_mode: str,
        action_scale: float,
        action_gain: float,
        action_smoothing: float,
        transition_k: int,
        transition_candidates: int,
        proximity_weight: float,
        bc_policy: Any | None = None,
        bc_eval_backend: str = "numpy",
    ):
        self.graph = graph
        self.q = q
        self.action_mode = action_mode
        self.action_scale = action_scale
        self.action_gain = action_gain
        self.action_smoothing = action_smoothing
        self.transition_k = transition_k
        self.transition_candidates = transition_candidates
        self.proximity_weight = proximity_weight
        self.bc_policy = bc_policy
        self.bc_eval_backend = bc_eval_backend
        self.prev_action: np.ndarray | None = None
        self.waypoint_cell: int | None = None
        self.waypoint_goal_cell: int | None = None
        self.config = {"pe_type": "frs"}
        self.source_cells = np.asarray(
            [i for i, edges in enumerate(graph.edges_by_src) if len(edges) > 0],
            dtype=np.int32,
        )
        self.transition_source_cells = np.asarray(
            [i for i, idxs in enumerate(graph.transition_indices_by_src) if len(idxs) > 0],
            dtype=np.int32,
        )
        self.v = np.zeros((graph.n_states, graph.n_states), dtype=np.float32)
        np.maximum.at(self.v, graph.edge_src, q)
        diag = np.arange(graph.n_states)
        self.v[diag, diag] = 1.0

    def _waypoint_action(self, obs: np.ndarray, target: np.ndarray, use_bc: bool = False) -> np.ndarray:
        if not use_bc:
            return np.clip((target - obs[: target.shape[0]]) / self.action_scale, -1.0, 1.0)
        if self.bc_policy is None:
            raise ValueError("bc_persistent_waypoint requires --bc-policy")
        goal_obs = self._bc_goal_obs(obs, target)
        return self.bc_policy.action(obs, goal_obs, self.bc_eval_backend)

    def _bc_goal_obs(self, obs: np.ndarray, target: np.ndarray) -> np.ndarray:
        obs_dim = int(self.bc_policy.stats.obs_mean.shape[0])
        if self.bc_policy.stats.goal_representation == "xy":
            return np.asarray(target[:2], dtype=np.float32)
        if obs_dim == len(target):
            return np.asarray(target, dtype=np.float32)
        d2 = np.sum((self.graph.transition_obs[:, :2] - target[None, :2]) ** 2, axis=1)
        goal = np.asarray(self.graph.transition_obs[int(np.argmin(d2))], dtype=np.float32).copy()
        if goal.shape[0] >= 2:
            goal[:2] = target[:2]
        elif obs.shape[0] == obs_dim:
            goal = np.asarray(obs, dtype=np.float32).copy()
            goal[:2] = target[:2]
        return goal

    def reset_episode(self) -> None:
        self.prev_action = None
        self.waypoint_cell = None
        self.waypoint_goal_cell = None

    def _cell_with_edges(self, obs: np.ndarray) -> int:
        cell = self.graph.obs_to_cell(obs)
        if len(self.graph.edges_by_src[cell]) > 0:
            return cell
        source_centers = self.graph.centers[self.source_cells]
        d2 = np.sum((source_centers - np.asarray(obs)[None, :]) ** 2, axis=1)
        return int(self.source_cells[int(np.argmin(d2))])

    def _cell_with_transitions(self, obs: np.ndarray) -> int:
        cell = self.graph.obs_to_cell(obs)
        if len(self.graph.transition_indices_by_src[cell]) > 0:
            return cell
        source_centers = self.graph.centers[self.transition_source_cells]
        d2 = np.sum((source_centers - np.asarray(obs)[None, :]) ** 2, axis=1)
        return int(self.transition_source_cells[int(np.argmin(d2))])

    def _transition_value_action(self, obs: np.ndarray, goal_cell: int) -> np.ndarray:
        cell = self._cell_with_transitions(obs)
        idxs = self.graph.transition_indices_by_src[cell]
        if len(idxs) == 0:
            return np.zeros(self.graph.edge_actions.shape[1], dtype=np.float32)
        d2 = np.sum((self.graph.transition_obs[idxs] - obs[None, :]) ** 2, axis=1)
        n_candidates = min(self.transition_candidates, len(idxs))
        if n_candidates < len(idxs):
            local = np.argpartition(d2, n_candidates - 1)[:n_candidates]
        else:
            local = np.arange(len(idxs))
        candidate_idxs = idxs[local]
        next_values = self.v[self.graph.transition_next_ids[candidate_idxs], goal_cell]
        proximity_penalty = self.proximity_weight * np.sqrt(d2[local]) / max(
            self.graph.cell_size, 1e-6
        )
        scores = next_values - proximity_penalty
        if self.action_mode == "transition_value_knn" and self.transition_k > 1:
            k = min(self.transition_k, len(candidate_idxs))
            top = np.argpartition(-scores, k - 1)[:k]
            weights = np.exp(20.0 * (scores[top] - np.max(scores[top])))
            return np.average(
                self.graph.transition_actions[candidate_idxs[top]],
                axis=0,
                weights=weights,
            )
        return self.graph.transition_actions[candidate_idxs[int(np.argmax(scores))]]

    def _q_transition_action(self, obs: np.ndarray, goal_cell: int) -> np.ndarray:
        cell = self._cell_with_transitions(obs)
        idxs = self.graph.transition_indices_by_src[cell]
        if len(idxs) == 0:
            return np.zeros(self.graph.edge_actions.shape[1], dtype=np.float32)
        d2 = np.sum((self.graph.transition_obs[idxs] - obs[None, :]) ** 2, axis=1)
        n_candidates = min(self.transition_candidates, len(idxs))
        if n_candidates < len(idxs):
            local = np.argpartition(d2, n_candidates - 1)[:n_candidates]
        else:
            local = np.arange(len(idxs))
        candidate_idxs = idxs[local]
        edge_ids = self.graph.transition_edge_ids[candidate_idxs]
        expected_values = self.q[edge_ids, goal_cell]
        proximity_penalty = self.proximity_weight * np.sqrt(d2[local]) / max(
            self.graph.cell_size, 1e-6
        )
        scores = expected_values - proximity_penalty
        if self.action_mode == "q_transition_knn" and self.transition_k > 1:
            k = min(self.transition_k, len(candidate_idxs))
            top = np.argpartition(-scores, k - 1)[:k]
            weights = np.exp(20.0 * (scores[top] - np.max(scores[top])))
            return np.average(
                self.graph.transition_actions[candidate_idxs[top]],
                axis=0,
                weights=weights,
            )
        return self.graph.transition_actions[candidate_idxs[int(np.argmax(scores))]]

    def _value_waypoint_action(self, obs: np.ndarray, goal_cell: int) -> np.ndarray:
        cell = self._cell_with_edges(obs)
        edges = self.graph.edges_by_src[cell]
        if len(edges) == 0:
            return np.zeros(self.graph.edge_actions.shape[1], dtype=np.float32)
        edge = int(edges[int(np.argmax(self.q[edges, goal_cell]))])
        next_ids = self.graph.edge_next_ids[edge]
        probs = self.graph.edge_next_probs[edge]
        if len(next_ids) == 0:
            return self.graph.edge_actions[edge]

        next_centers = self.graph.centers[next_ids]
        values = self.v[next_ids, goal_cell]
        best_next = int(next_ids[int(np.argmax(values))])
        target = self.graph.centers[best_next]

        if self.action_mode == "value_waypoint_hybrid":
            src_center = self.graph.centers[cell]
            expected_next = (probs[:, None] * next_centers).sum(axis=0)
            transition_span = float(
                np.max(np.linalg.norm(next_centers - expected_next[None, :], axis=1))
            )
            target_distance = float(np.linalg.norm(target - src_center))
            if (
                len(next_ids) > 1
                and (transition_span > 2.5 * self.graph.cell_size or target_distance > 2.5 * self.graph.cell_size)
            ):
                return self.graph.edge_actions[edge]

        delta_action = (target - obs) / self.action_scale
        return np.clip(delta_action, -1.0, 1.0)

    def _persistent_waypoint_action(
        self,
        obs: np.ndarray,
        goal_cell: int,
        use_bc: bool = False,
    ) -> np.ndarray:
        if self.waypoint_goal_cell != goal_cell:
            self.waypoint_cell = None
            self.waypoint_goal_cell = goal_cell

        if self.waypoint_cell is not None:
            target = self.graph.centers[self.waypoint_cell]
            if np.linalg.norm(target - obs) > 0.5 * self.graph.cell_size:
                return self._waypoint_action(obs, target, use_bc=use_bc)

        cell = self._cell_with_edges(obs)
        edges = self.graph.edges_by_src[cell]
        if len(edges) == 0:
            self.waypoint_cell = None
            return np.zeros(self.graph.edge_actions.shape[1], dtype=np.float32)

        edge = int(edges[int(np.argmax(self.q[edges, goal_cell]))])
        next_ids = self.graph.edge_next_ids[edge]
        probs = self.graph.edge_next_probs[edge]
        if len(next_ids) == 0:
            self.waypoint_cell = None
            return self.graph.edge_actions[edge]

        src_center = self.graph.centers[cell]
        next_centers = self.graph.centers[next_ids]
        local_distances = np.linalg.norm(next_centers - src_center[None, :], axis=1)
        local_mask = local_distances <= 2.5 * self.graph.cell_size
        expected_next = (probs[:, None] * next_centers).sum(axis=0)
        transition_span = float(
            np.max(np.linalg.norm(next_centers - expected_next[None, :], axis=1))
        )
        if not np.any(local_mask) or transition_span > 2.5 * self.graph.cell_size:
            self.waypoint_cell = None
            return self.graph.edge_actions[edge]

        local_ids = next_ids[local_mask]
        local_probs = probs[local_mask]
        local_values = self.v[local_ids, goal_cell]
        local_scores = local_values + 0.02 * np.log(np.maximum(local_probs, 1e-6))
        next_cell = int(local_ids[int(np.argmax(local_scores))])
        if next_cell == cell or np.linalg.norm(self.graph.centers[next_cell] - obs) < 0.25 * self.graph.cell_size:
            self.waypoint_cell = None
            return self.graph.edge_actions[edge]

        self.waypoint_cell = next_cell
        target = self.graph.centers[self.waypoint_cell]
        return self._waypoint_action(obs, target, use_bc=use_bc)

    def _mean_edge_action(self, obs: np.ndarray, goal_cell: int) -> np.ndarray:
        cell = self._cell_with_edges(obs)
        edges = self.graph.edges_by_src[cell]
        if len(edges) == 0:
            return np.zeros(self.graph.edge_actions.shape[1], dtype=np.float32)
        edge = int(edges[int(np.argmax(self.q[edges, goal_cell]))])
        return self.graph.edge_actions[edge]

    def sample_actions(self, observations, goals=None, seed=None, temperature=0.0):
        del seed, temperature
        obs = np.asarray(observations, dtype=np.float32)
        goal = np.asarray(goals, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[None, :]
            goal = goal[None, :]
        out = []
        for ob, go in zip(obs, goal):
            goal_cell = self.graph.obs_to_cell(go)
            if self.action_mode in {"transition_value", "transition_value_knn"}:
                out.append(self._transition_value_action(ob, goal_cell))
                continue
            if self.action_mode in {"q_transition", "q_transition_knn"}:
                out.append(self._q_transition_action(ob, goal_cell))
                continue
            if self.action_mode in {"value_waypoint", "value_waypoint_hybrid"}:
                out.append(self._value_waypoint_action(ob, goal_cell))
                continue
            if self.action_mode == "persistent_waypoint":
                out.append(self._persistent_waypoint_action(ob, goal_cell))
                continue
            if self.action_mode == "bc_persistent_waypoint":
                out.append(self._persistent_waypoint_action(ob, goal_cell, use_bc=True))
                continue
            if self.action_mode.startswith("persistent_waypoint_blend"):
                suffix = self.action_mode.removeprefix("persistent_waypoint_blend")
                waypoint_weight = float(suffix) / 100.0
                waypoint_action = self._persistent_waypoint_action(ob, goal_cell)
                edge_action = self._mean_edge_action(ob, goal_cell)
                blended = waypoint_weight * waypoint_action + (1.0 - waypoint_weight) * edge_action
                out.append(np.clip(blended, -1.0, 1.0))
                continue
            cell = self._cell_with_edges(ob)
            edges = self.graph.edges_by_src[cell]
            if len(edges) == 0:
                out.append(np.zeros(self.graph.edge_actions.shape[1], dtype=np.float32))
                continue
            edge = int(edges[int(np.argmax(self.q[edges, goal_cell]))])
            if self.action_mode == "center_delta":
                delta_action = (self.graph.edge_next_centers[edge] - ob) / self.action_scale
                out.append(np.clip(delta_action, -1.0, 1.0))
            elif self.action_mode == "hybrid":
                delta = self.graph.edge_next_centers[edge] - ob
                if np.linalg.norm(delta) <= 2.5 * self.graph.cell_size:
                    out.append(np.clip(delta / self.action_scale, -1.0, 1.0))
                else:
                    out.append(self.graph.edge_actions[edge])
            elif self.action_mode in {"nearest_transition", "knn_transition"}:
                idxs = self.graph.transition_indices_by_edge[edge]
                if len(idxs) == 0:
                    out.append(self.graph.edge_actions[edge])
                    continue
                d2 = np.sum((self.graph.transition_obs[idxs] - ob[None, :]) ** 2, axis=1)
                if self.action_mode == "nearest_transition" or self.transition_k <= 1:
                    out.append(self.graph.transition_actions[idxs[int(np.argmin(d2))]])
                else:
                    k = min(self.transition_k, len(idxs))
                    local = np.argpartition(d2, k - 1)[:k]
                    weights = 1.0 / np.maximum(d2[local], 1e-4)
                    actions = self.graph.transition_actions[idxs[local]]
                    out.append(np.average(actions, axis=0, weights=weights))
            else:
                out.append(self.graph.edge_actions[edge])
        actions = np.clip(self.action_gain * np.asarray(out, dtype=np.float32), -1.0, 1.0)
        if single and self.action_smoothing > 0.0:
            if self.prev_action is None:
                smoothed = actions[0]
            else:
                smoothed = (
                    self.action_smoothing * self.prev_action
                    + (1.0 - self.action_smoothing) * actions[0]
                )
            self.prev_action = np.clip(smoothed, -1.0, 1.0).astype(np.float32)
            actions[0] = self.prev_action
        return actions[0] if single else actions


def evaluate_agent(
    agent: GraphPlannerAgent,
    env,
    episodes: int,
    seed: int,
    task_ids: list[int] | None = None,
    profile: bool = False,
    max_episode_steps: int | None = None,
) -> dict[str, float]:
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, "task_infos") else env.task_infos
    if task_ids is None:
        selected = list(enumerate(task_infos, start=1))
    else:
        selected = []
        for task_id in task_ids:
            if task_id < 1 or task_id > len(task_infos):
                raise ValueError(f"task id {task_id} outside valid range 1..{len(task_infos)}")
            selected.append((task_id, task_infos[task_id - 1]))

    row: dict[str, float] = {}
    successes = []
    eval_start = time.perf_counter()
    profile_reset_seconds = 0.0
    profile_action_seconds = 0.0
    profile_env_step_seconds = 0.0
    profile_env_steps = 0
    for task_id, task_info in selected:
        task_successes = []
        lengths = []
        for _ in range(episodes):
            agent.reset_episode()
            episode_idx = len(task_successes)
            episode_seed = seed * 1_000_000 + task_id * 10_000 + episode_idx
            np.random.seed(episode_seed)
            reset_start = time.perf_counter()
            obs, info = env.reset(
                seed=episode_seed,
                options={"task_id": task_id},
            )
            profile_reset_seconds += time.perf_counter() - reset_start
            goal = info["goal"]
            done = False
            length = 0
            while not done:
                action_start = time.perf_counter()
                action = np.clip(agent.sample_actions(obs, goals=goal), -1.0, 1.0)
                profile_action_seconds += time.perf_counter() - action_start
                step_start = time.perf_counter()
                obs, _reward, terminated, truncated, info = env.step(action)
                profile_env_step_seconds += time.perf_counter() - step_start
                profile_env_steps += 1
                done = bool(terminated or truncated)
                length += 1
                if max_episode_steps is not None and length >= max_episode_steps:
                    done = True
            success = float(info.get("success", info.get("episode", {}).get("success", 0.0)))
            task_successes.append(success)
            lengths.append(length)
        mean_success = float(np.mean(task_successes))
        row[f"task{task_id}_success"] = mean_success
        row[f"task{task_id}_length"] = float(np.mean(lengths))
        successes.append(mean_success)
        print(
            f"[eval] {task_info['task_name']} success={mean_success:.3f} "
            f"length={np.mean(lengths):.1f}",
            flush=True,
        )
    row["overall_success"] = float(np.mean(successes))
    row["eval_seconds"] = time.perf_counter() - eval_start
    if profile:
        row["profile_reset_seconds"] = profile_reset_seconds
        row["profile_action_seconds"] = profile_action_seconds
        row["profile_env_step_seconds"] = profile_env_step_seconds
        row["profile_env_steps"] = float(profile_env_steps)
        row["profile_action_ms_per_step"] = 1000.0 * profile_action_seconds / max(profile_env_steps, 1)
        row["profile_env_step_ms"] = 1000.0 * profile_env_step_seconds / max(profile_env_steps, 1)
    return row


def task_points(env) -> np.ndarray:
    infos = env.unwrapped.task_infos if hasattr(env.unwrapped, "task_infos") else env.task_infos
    points = []
    for info in infos:
        points.append(info["init_xy"])
        points.append(info["goal_xy"])
    return np.asarray(points, dtype=np.float32)


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
    parser.add_argument("--cell-size", type=float, default=2.0)
    parser.add_argument("--action-threshold", type=float, default=0.25)
    parser.add_argument("--action-bin-levels", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--full-iters", type=int, default=80)
    parser.add_argument("--polish-iters", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--task-ids", type=int, nargs="+", default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--profile-eval", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--action-mode",
        choices=[
            "mean",
            "center_delta",
            "hybrid",
            "nearest_transition",
            "knn_transition",
            "transition_value",
            "transition_value_knn",
            "q_transition",
            "q_transition_knn",
            "value_waypoint",
            "value_waypoint_hybrid",
            "persistent_waypoint",
            "bc_persistent_waypoint",
            "persistent_waypoint_blend25",
            "persistent_waypoint_blend50",
            "persistent_waypoint_blend75",
        ],
        default="mean",
    )
    parser.add_argument("--action-scale", type=float, default=0.2)
    parser.add_argument("--action-gain", type=float, default=1.0)
    parser.add_argument("--action-smoothing", type=float, default=0.0)
    parser.add_argument("--action-modes", nargs="+", default=None)
    parser.add_argument("--transition-k", type=int, default=5)
    parser.add_argument("--transition-candidates", type=int, default=256)
    parser.add_argument("--proximity-weight", type=float, default=0.02)
    parser.add_argument("--bc-policy", type=Path, default=None)
    parser.add_argument(
        "--bc-eval-backend",
        choices=["jax", "numpy"],
        default="numpy",
        help="Policy backend for bc_persistent_waypoint. numpy is faster for screening.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "bellman_matched",
            "sto_trl_matched",
            "bellman_polished",
            "sto_trl_polished",
            "bellman_full",
        ],
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "pointmaze_graph_planner.csv")
    parser.add_argument("--summary-out", type=Path, default=ROOT / "results" / "pointmaze_graph_planner_summary.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results" / "cache")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    env, train, _val = make_env_and_datasets(args.env_name, dataset_path=None)
    graph = build_graph(
        train,
        task_points(env),
        args.cell_size,
        args.action_threshold,
        args.action_bin_levels,
    )
    matched_iters = args.iters
    if matched_iters is None:
        matched_iters = max(1, int(math.ceil(math.log2(max(graph.n_states, 2)))) + 1)
    print(
        f"[graph] states={graph.n_states} edges={graph.n_edges} "
        f"matched_iters={matched_iters} cell_size={args.cell_size} "
        f"action_bin_levels={args.action_bin_levels}",
        flush=True,
    )

    specs = {
        "bellman_matched": (matched_iters, False, 0),
        "sto_trl_matched": (matched_iters, True, 0),
        "bellman_polished": (matched_iters, False, args.polish_iters),
        "sto_trl_polished": (matched_iters, True, args.polish_iters),
        "bellman_full": (args.full_iters, False, 0),
    }
    rows = []
    seeds = args.seeds if args.seeds is not None else [args.seed]
    action_modes = args.action_modes if args.action_modes is not None else [args.action_mode]
    bc_policy = None
    if any(mode == "bc_persistent_waypoint" for mode in action_modes):
        if args.bc_policy is None:
            raise ValueError("bc_persistent_waypoint requires --bc-policy")
        bc_policy = load_bc_policy_for_path(args.bc_policy)
        print(
            f"[bc] loaded policy={args.bc_policy} backend={args.bc_eval_backend} "
            f"goal_representation={bc_policy.stats.goal_representation}",
            flush=True,
        )
    for method in args.methods:
        iters, use_transitive, polish_iters = specs[method]
        print(
            f"[solve] method={method} iters={iters} transitive={use_transitive} "
            f"polish_iters={polish_iters}",
            flush=True,
        )
        q, q_solve_seconds, q_cache_hit = cached_solve_reachability(
            graph,
            args.env_name,
            method,
            args.gamma,
            iters,
            use_transitive,
            args.chunk_size,
            polish_iters,
            args.cache_dir,
            use_cache=not args.no_cache,
        )
        for action_mode in action_modes:
            for seed in seeds:
                print(f"[run] method={method} action_mode={action_mode} seed={seed}", flush=True)
                agent = GraphPlannerAgent(
                    graph,
                    q,
                    action_mode,
                    args.action_scale,
                    args.action_gain,
                    args.action_smoothing,
                    args.transition_k,
                    args.transition_candidates,
                    args.proximity_weight,
                    bc_policy=bc_policy,
                    bc_eval_backend=args.bc_eval_backend,
                )
                metrics = evaluate_agent(
                    agent,
                    env,
                    args.episodes,
                    seed,
                    task_ids=args.task_ids,
                    profile=args.profile_eval,
                    max_episode_steps=args.max_episode_steps,
                )
                row = {
                    "method": method,
                    "env": args.env_name,
                    "task_ids": "all" if args.task_ids is None else ",".join(map(str, args.task_ids)),
                    "episodes_per_task": args.episodes,
                    "max_episode_steps": "" if args.max_episode_steps is None else args.max_episode_steps,
                    "cell_size": args.cell_size,
                    "action_threshold": args.action_threshold,
                    "action_bin_levels": args.action_bin_levels,
                    "gamma": args.gamma,
                    "iters": iters,
                    "polish_iters": polish_iters,
                    "n_states": graph.n_states,
                    "n_edges": graph.n_edges,
                    "q_solve_seconds": q_solve_seconds,
                    "q_cache_hit": q_cache_hit,
                    "action_mode": action_mode,
                    "action_scale": args.action_scale,
                    "action_gain": args.action_gain,
                    "action_smoothing": args.action_smoothing,
                    "transition_k": args.transition_k,
                    "transition_candidates": args.transition_candidates,
                    "proximity_weight": args.proximity_weight,
                    "bc_policy": "" if args.bc_policy is None else str(args.bc_policy),
                    "bc_eval_backend": args.bc_eval_backend,
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
