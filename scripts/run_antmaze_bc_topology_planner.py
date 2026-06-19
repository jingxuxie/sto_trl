#!/usr/bin/env python3
"""Learned local-controller evaluation for stochastic TRL on AntMaze teleport."""

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
from dataclasses import dataclass, field
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training import train_state

from envs.env_utils import make_env_and_datasets
from run_antmaze_topology_planner import ACTION_DELTAS, Topology, build_dataset_topology, nearest_state
from sto_trl.learners import train_model_value, train_support_transitive_value


CACHE_VERSION = 1


class BCActor(nn.Module):
    hidden_dims: tuple[int, ...]
    action_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.gelu(x)
        return nn.Dense(self.action_dim)(x)


class RawObsTransitionMLP(nn.Module):
    hidden_dims: tuple[int, ...]
    output_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        return nn.Dense(self.output_dim)(x)


class RawObsPolicyMLP(nn.Module):
    hidden_dims: tuple[int, ...]
    n_actions: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        return nn.Dense(self.n_actions)(x)


@dataclass
class FeatureStats:
    obs_mean: np.ndarray
    obs_std: np.ndarray
    xy_mean: np.ndarray
    xy_std: np.ndarray
    rel_scale: float
    goal_representation: str


@dataclass
class BCPolicy:
    state: train_state.TrainState
    stats: FeatureStats
    metadata: dict[str, Any] = field(default_factory=dict)
    _apply_action: Any = field(init=False, repr=False)
    _numpy_layers: Any = field(init=False, repr=False)
    _numpy_final: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._apply_action = jax.jit(
            lambda feats: self.state.apply_fn({"params": self.state.params}, feats)
        )
        self._numpy_layers, self._numpy_final = self._extract_numpy_params()

    def _extract_numpy_params(self) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]], tuple[np.ndarray, np.ndarray]]:
        params = serialization.to_state_dict(jax.device_get(self.state.params))
        dense_ids = sorted(int(key.split("_")[1]) for key in params if key.startswith("Dense_"))
        if not dense_ids:
            raise ValueError("BC policy has no Dense layers")
        final_id = dense_ids[-1]
        layers = []
        for idx in dense_ids[:-1]:
            dense = params[f"Dense_{idx}"]
            ln = params.get(f"LayerNorm_{idx}")
            layers.append(
                (
                    np.asarray(dense["kernel"], dtype=np.float32),
                    np.asarray(dense["bias"], dtype=np.float32),
                    None if ln is None else np.asarray(ln["scale"], dtype=np.float32),
                    None if ln is None else np.asarray(ln["bias"], dtype=np.float32),
                )
            )
        final = params[f"Dense_{final_id}"]
        return layers, (
            np.asarray(final["kernel"], dtype=np.float32),
            np.asarray(final["bias"], dtype=np.float32),
        )

    @staticmethod
    def _gelu_np(x: np.ndarray) -> np.ndarray:
        return (0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)))).astype(
            np.float32
        )

    def _apply_action_np(self, feats: np.ndarray) -> np.ndarray:
        x = np.asarray(feats, dtype=np.float32)
        for kernel, bias, ln_scale, ln_bias in self._numpy_layers:
            x = x @ kernel + bias
            if ln_scale is not None and ln_bias is not None:
                mean = np.mean(x, axis=-1, keepdims=True)
                var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
                x = (x - mean) * np.reciprocal(np.sqrt(var + 1e-6))
                x = x * ln_scale + ln_bias
            x = self._gelu_np(x)
        final_kernel, final_bias = self._numpy_final
        return x @ final_kernel + final_bias

    def action(self, obs: np.ndarray, goal: np.ndarray, backend: str) -> np.ndarray:
        obs_batch = np.asarray(obs, dtype=np.float32)[None, :]
        goal_batch = np.asarray(goal, dtype=np.float32)[None, :]
        feats = make_features_np(obs_batch, goal_batch, self.stats)
        if backend == "jax":
            action = self._apply_action(jnp.asarray(feats))
        elif backend == "numpy":
            action = self._apply_action_np(feats)
        else:
            raise ValueError(f"unknown policy evaluation backend {backend}")
        return np.clip(np.asarray(action[0]), -1.0, 1.0).astype(np.float32)


@dataclass
class GoalObservationIndex:
    state_goal_obs: np.ndarray


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


def topology_to_payload(topology: Topology) -> dict[str, Any]:
    return {
        "states": topology.states,
        "state_to_idx": topology.state_to_idx,
        "transitions": topology.transitions,
        "intended_next": topology.intended_next,
        "jump_sources": topology.jump_sources,
    }


def topology_from_payload(payload: dict[str, Any]) -> Topology:
    return Topology(
        states=tuple(tuple(x) for x in payload["states"]),
        state_to_idx={tuple(k): int(v) for k, v in payload["state_to_idx"].items()},
        transitions=np.asarray(payload["transitions"], dtype=np.float64),
        intended_next=np.asarray(payload["intended_next"], dtype=np.int64),
        jump_sources={tuple(x) for x in payload["jump_sources"]},
    )


def cache_matches(payload: dict[str, Any], expected: dict[str, Any]) -> bool:
    meta = payload.get("meta", {})
    return all(meta.get(key) == value for key, value in expected.items())


def cached_dataset_topology(
    env_name: str,
    env,
    train: dict[str, np.ndarray],
    min_jump_count: int,
    cache_dir: Path | None,
    use_cache: bool,
) -> tuple[Topology, float, int]:
    expected = {
        "version": CACHE_VERSION,
        "kind": "antmaze_dataset_topology",
        "env_name": env_name,
        "min_jump_count": int(min_jump_count),
        "obs_shape": tuple(int(x) for x in np.asarray(train["observations"]).shape),
    }
    cache_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"antmaze_topology_v{CACHE_VERSION}_{safe_cache_token(env_name)}_jump{min_jump_count}.pkl"
    start = time.perf_counter()
    if use_cache and cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        if cache_matches(payload, expected):
            topology = topology_from_payload(payload["topology"])
            elapsed = time.perf_counter() - start
            print(f"[cache] topology hit path={cache_path} seconds={elapsed:.2f}", flush=True)
            return topology, elapsed, 1
        print(f"[cache] topology stale path={cache_path}", flush=True)

    topology = build_dataset_topology(env, train, min_jump_count)
    elapsed = time.perf_counter() - start
    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump({"meta": expected, "topology": topology_to_payload(topology)}, f)
        print(f"[cache] topology wrote path={cache_path} seconds={elapsed:.2f}", flush=True)
    else:
        print(f"[cache] topology disabled seconds={elapsed:.2f}", flush=True)
    return topology, elapsed, 0


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
    return GoalObservationIndex(state_goal_obs=np.asarray(state_goal_obs, dtype=np.float32))


def cached_goal_observation_index(
    env_name: str,
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
    candidates_per_state: int,
    cache_dir: Path | None,
    use_cache: bool,
) -> tuple[GoalObservationIndex, float, int]:
    expected = {
        "version": CACHE_VERSION,
        "kind": "antmaze_goal_observation_index",
        "env_name": env_name,
        "candidates_per_state": int(candidates_per_state),
        "obs_shape": tuple(int(x) for x in np.asarray(train["observations"]).shape),
        "states": tuple(tuple(x) for x in topology.states),
    }
    cache_path = None
    if cache_dir is not None:
        cache_path = (
            cache_dir
            / f"antmaze_goal_obs_v{CACHE_VERSION}_{safe_cache_token(env_name)}_k{candidates_per_state}.pkl"
        )
    start = time.perf_counter()
    if use_cache and cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        if cache_matches(payload, expected):
            goal_index = GoalObservationIndex(
                state_goal_obs=np.asarray(payload["state_goal_obs"], dtype=np.float32)
            )
            elapsed = time.perf_counter() - start
            print(f"[cache] goal_index hit path={cache_path} seconds={elapsed:.2f}", flush=True)
            return goal_index, elapsed, 1
        print(f"[cache] goal_index stale path={cache_path}", flush=True)

    goal_index = build_goal_observation_index(env, train, topology, candidates_per_state)
    elapsed = time.perf_counter() - start
    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump({"meta": expected, "state_goal_obs": goal_index.state_goal_obs}, f)
        print(f"[cache] goal_index wrote path={cache_path} seconds={elapsed:.2f}", flush=True)
    else:
        print(f"[cache] goal_index disabled seconds={elapsed:.2f}", flush=True)
    return goal_index, elapsed, 0


def trajectory_bounds(terminals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    terminal_locs = np.nonzero(np.asarray(terminals) > 0)[0]
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])
    return initial_locs.astype(np.int32), terminal_locs.astype(np.int32)


def valid_training_indices(train: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    _initial_locs, terminal_locs = trajectory_bounds(train["terminals"])
    valids = np.asarray(train["valids"]) > 0
    idxs = np.nonzero(valids)[0].astype(np.int32)
    idxs = idxs[idxs < len(valids) - 1]
    final_locs = terminal_locs[np.searchsorted(terminal_locs, idxs)]
    keep = final_locs > idxs
    return idxs[keep].astype(np.int32), final_locs[keep].astype(np.int32)


def make_feature_stats(
    obs: np.ndarray,
    rel_scale: float | None,
    goal_representation: str,
) -> FeatureStats:
    obs_mean = obs.mean(axis=0).astype(np.float32)
    obs_std = np.maximum(obs.std(axis=0), 1e-3).astype(np.float32)
    xy = obs[:, :2]
    xy_mean = xy.mean(axis=0).astype(np.float32)
    xy_std = np.maximum(xy.std(axis=0), 1e-3).astype(np.float32)
    if rel_scale is None:
        rel_scale = float(np.maximum(np.median(np.linalg.norm(xy - xy_mean[None, :], axis=1)), 1.0))
    return FeatureStats(
        obs_mean=obs_mean,
        obs_std=obs_std,
        xy_mean=xy_mean,
        xy_std=xy_std,
        rel_scale=float(rel_scale),
        goal_representation=goal_representation,
    )


def make_features_np(obs: np.ndarray, goal: np.ndarray, stats: FeatureStats) -> np.ndarray:
    obs_norm = np.clip((obs - stats.obs_mean[None, :]) / stats.obs_std[None, :], -10.0, 10.0)
    if stats.goal_representation == "full":
        goal_xy = goal[:, :2]
        goal_norm = np.clip((goal - stats.obs_mean[None, :]) / stats.obs_std[None, :], -10.0, 10.0)
    elif stats.goal_representation == "xy":
        goal_xy = goal
        goal_norm = np.clip((goal_xy - stats.xy_mean[None, :]) / stats.xy_std[None, :], -10.0, 10.0)
    else:
        raise ValueError(f"unknown goal representation {stats.goal_representation}")
    delta = goal_xy - obs[:, :2]
    rel = np.clip(delta / stats.rel_scale, -10.0, 10.0)
    dist = np.linalg.norm(delta, axis=1, keepdims=True)
    unit = delta / np.maximum(dist, 1e-6)
    dist_feat = np.clip(dist / stats.rel_scale, 0.0, 10.0)
    return np.concatenate([obs_norm, goal_norm, rel, unit, dist_feat], axis=1).astype(np.float32)


def sample_bc_batch(
    rng: np.random.Generator,
    obs: np.ndarray,
    actions: np.ndarray,
    valid_idxs: np.ndarray,
    final_locs: np.ndarray,
    batch_size: int,
    min_future: int,
    max_future: int,
    stats: FeatureStats,
) -> tuple[np.ndarray, np.ndarray]:
    rows = rng.integers(0, len(valid_idxs), size=batch_size)
    idxs = valid_idxs[rows]
    finals = final_locs[rows]
    offsets = rng.integers(min_future, max_future + 1, size=batch_size)
    goal_idxs = np.minimum(idxs + offsets, finals)
    goals = obs[goal_idxs] if stats.goal_representation == "full" else obs[goal_idxs, :2]
    feats = make_features_np(obs[idxs], goals, stats)
    return feats, actions[idxs].astype(np.float32)


def create_train_state(
    seed: int,
    feature_dim: int,
    action_dim: int,
    hidden_dims: tuple[int, ...],
    lr: float,
    layer_norm: bool,
) -> train_state.TrainState:
    model = BCActor(hidden_dims=hidden_dims, action_dim=action_dim, layer_norm=layer_norm)
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, feature_dim), dtype=jnp.float32))["params"]
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(lr))


def create_loaded_train_state(
    params: dict[str, Any],
    action_dim: int,
    hidden_dims: tuple[int, ...],
    lr: float,
    layer_norm: bool,
) -> train_state.TrainState:
    model = BCActor(hidden_dims=hidden_dims, action_dim=action_dim, layer_norm=layer_norm)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(lr))


@jax.jit
def train_step(state: train_state.TrainState, feats: jax.Array, actions: jax.Array):
    def loss_fn(params):
        pred = state.apply_fn({"params": params}, feats)
        loss = jnp.mean((pred - actions) ** 2)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@jax.jit
def eval_loss(state: train_state.TrainState, feats: jax.Array, actions: jax.Array) -> jax.Array:
    pred = state.apply_fn({"params": state.params}, feats)
    return jnp.mean((pred - actions) ** 2)


def train_bc_policy(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    seed: int,
    steps: int,
    batch_size: int,
    min_future: int,
    max_future: int,
    hidden_dims: tuple[int, ...],
    lr: float,
    rel_scale: float | None,
    goal_representation: str,
    layer_norm: bool,
    log_interval: int,
    progress_log: Path | None,
) -> BCPolicy:
    obs = np.asarray(train["observations"], dtype=np.float32)
    actions = np.asarray(train["actions"], dtype=np.float32)
    stats = make_feature_stats(obs, rel_scale, goal_representation)
    valid_idxs, final_locs = valid_training_indices(train)
    val_obs = np.asarray(val["observations"], dtype=np.float32)
    val_actions = np.asarray(val["actions"], dtype=np.float32)
    val_valid_idxs, val_final_locs = valid_training_indices(val)

    rng = np.random.default_rng(seed)
    example_goal = obs[:1] if goal_representation == "full" else obs[:1, :2]
    feature_dim = make_features_np(obs[:1], example_goal, stats).shape[1]
    state = create_train_state(seed, feature_dim, actions.shape[1], hidden_dims, lr, layer_norm)
    print(
        f"[bc] train_idxs={len(valid_idxs)} val_idxs={len(val_valid_idxs)} "
        f"feature_dim={feature_dim} action_dim={actions.shape[1]} "
        f"goal={goal_representation} device={jax.devices()[0]}",
        flush=True,
    )

    last_loss = math.nan
    for step in range(1, steps + 1):
        feats, target_actions = sample_bc_batch(
            rng, obs, actions, valid_idxs, final_locs, batch_size, min_future, max_future, stats
        )
        state, loss = train_step(state, jnp.asarray(feats), jnp.asarray(target_actions))
        last_loss = float(loss)
        if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == steps):
            val_feats, val_targets = sample_bc_batch(
                rng,
                val_obs,
                val_actions,
                val_valid_idxs,
                val_final_locs,
                batch_size,
                min_future,
                max_future,
                stats,
            )
            val_mse = float(eval_loss(state, jnp.asarray(val_feats), jnp.asarray(val_targets)))
            print(f"[bc] step={step} train_mse={last_loss:.5f} val_mse={val_mse:.5f}", flush=True)
            append_progress(
                progress_log,
                {
                    "type": "bc_train",
                    "step": step,
                    "train_mse": last_loss,
                    "val_mse": val_mse,
                    "goal_representation": goal_representation,
                },
            )
    return BCPolicy(
        state=state,
        stats=stats,
        metadata={
            "bc_steps": int(steps),
            "bc_batch_size": int(batch_size),
            "bc_min_future": int(min_future),
            "bc_max_future": int(max_future),
            "bc_seed": int(seed),
            "bc_hidden_dims": tuple(int(x) for x in hidden_dims),
            "bc_lr": float(lr),
            "bc_layer_norm": bool(layer_norm),
            "goal_representation": goal_representation,
        },
    )


def save_bc_policy(
    path: Path,
    policy: BCPolicy,
    hidden_dims: tuple[int, ...],
    lr: float,
    layer_norm: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": serialization.to_bytes(policy.state.params),
        "stats": {
            "obs_mean": policy.stats.obs_mean,
            "obs_std": policy.stats.obs_std,
            "xy_mean": policy.stats.xy_mean,
            "xy_std": policy.stats.xy_std,
            "rel_scale": policy.stats.rel_scale,
            "goal_representation": policy.stats.goal_representation,
        },
        "feature_dim": int(policy.stats.obs_mean.shape[0] * 2 + 5)
        if policy.stats.goal_representation == "full"
        else int(policy.stats.obs_mean.shape[0] + 7),
        "action_dim": int(policy.state.params[f"Dense_{len(hidden_dims)}"]["kernel"].shape[1]),
        "hidden_dims": tuple(int(x) for x in hidden_dims),
        "lr": float(lr),
        "layer_norm": bool(layer_norm),
        "metadata": dict(policy.metadata),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def infer_policy_steps_from_path(path: Path) -> int | None:
    for token in reversed(path.stem.split("_")):
        token = token.lower()
        if token.endswith("k") and token[:-1].isdigit():
            return int(token[:-1]) * 1000
        if token.endswith("m") and token[:-1].isdigit():
            return int(token[:-1]) * 1_000_000
        if token.isdigit():
            return int(token)
    return None


def load_bc_policy(path: Path) -> BCPolicy:
    with path.open("rb") as f:
        payload = pickle.load(f)
    stats_payload = payload["stats"]
    stats = FeatureStats(
        obs_mean=np.asarray(stats_payload["obs_mean"], dtype=np.float32),
        obs_std=np.asarray(stats_payload["obs_std"], dtype=np.float32),
        xy_mean=np.asarray(stats_payload["xy_mean"], dtype=np.float32),
        xy_std=np.asarray(stats_payload["xy_std"], dtype=np.float32),
        rel_scale=float(stats_payload["rel_scale"]),
        goal_representation=str(stats_payload["goal_representation"]),
    )
    hidden_dims = tuple(int(x) for x in payload["hidden_dims"])
    action_dim = int(payload["action_dim"])
    lr = float(payload["lr"])
    layer_norm = bool(payload["layer_norm"])
    try:
        params = serialization.msgpack_restore(payload["params"])
        state = create_loaded_train_state(
            params=params,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            lr=lr,
            layer_norm=layer_norm,
        )
    except Exception:
        state = create_train_state(
            seed=0,
            feature_dim=int(payload["feature_dim"]),
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            lr=lr,
            layer_norm=layer_norm,
        )
        params = serialization.from_bytes(state.params, payload["params"])
        state = state.replace(params=params)
    metadata = dict(payload.get("metadata", {}))
    metadata.setdefault("bc_hidden_dims", hidden_dims)
    metadata.setdefault("bc_lr", lr)
    metadata.setdefault("bc_layer_norm", layer_norm)
    metadata.setdefault("goal_representation", stats.goal_representation)
    inferred_steps = infer_policy_steps_from_path(path)
    if inferred_steps is not None:
        metadata.setdefault("bc_steps", int(inferred_steps))
    return BCPolicy(state=state, stats=stats, metadata=metadata)


class BCTopologyAgent:
    def __init__(
        self,
        env,
        topology: Topology,
        q: np.ndarray,
        policy: BCPolicy,
        goal_index: GoalObservationIndex,
        waypoint_lookahead: int,
        path_mode: str,
        advance_distance: float,
        goal_candidate_mode: str,
        policy_eval_backend: str,
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
        self.policy_eval_backend = policy_eval_backend
        self.state_centers = np.asarray(
            [self.env.unwrapped.ij_to_xy(state) for state in self.topology.states],
            dtype=np.float32,
        )
        self.previous_action_id: int | None = None
        self.path: list[int] | None = None
        self.path_goal_state: int | None = None
        self.path_cursor = 0

    def reset_episode(self) -> None:
        self.previous_action_id = None
        self.path = None
        self.path_goal_state = None
        self.path_cursor = 0

    def nearest_state(self, obs: np.ndarray) -> int:
        ij = tuple(self.env.unwrapped.xy_to_ij(np.asarray(obs[:2], dtype=np.float64)))
        if ij in self.topology.state_to_idx:
            return self.topology.state_to_idx[ij]
        xy = np.asarray(obs[:2], dtype=np.float32)
        return int(np.argmin(np.sum((self.state_centers - xy[None, :]) ** 2, axis=1)))

    def _action_values(self, state: int, goal: int, previous_action_id: int | None) -> np.ndarray:
        if self.q.ndim == 4:
            prev_idx = self.topology.n_actions if previous_action_id is None else previous_action_id
            return self.q[state, prev_idx, :, goal].copy()
        values = self.q[state, :, goal].copy()
        if previous_action_id is not None:
            values[previous_action_id] += 1e-6
        return values

    def _greedy_action_id(self, state: int, goal: int) -> int:
        values = self._action_values(state, goal, self.previous_action_id)
        action_id = int(np.argmax(values))
        self.previous_action_id = action_id
        return action_id

    def _target_state(self, state: int, goal_state: int) -> int:
        target = state
        cur = state
        first_action_id: int | None = None
        for _ in range(max(1, self.waypoint_lookahead)):
            prev_id = self.previous_action_id if cur == state else None
            values = self._action_values(cur, goal_state, prev_id)
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
            next_xy = self.state_centers[next_state]
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
            scale = np.maximum(self.policy.stats.obs_std[2:], 1e-3)
            deltas = (candidates[:, 2:] - np.asarray(obs[2:], dtype=np.float32)[None, :]) / scale[None, :]
            scores = np.mean(np.square(np.clip(deltas, -10.0, 10.0)), axis=1)
            return candidates[int(np.argmin(scores))]
        raise ValueError(f"unknown goal candidate mode {self.goal_candidate_mode}")

    def sample_action(self, obs: np.ndarray, goal: np.ndarray, goal_state: int | None = None) -> np.ndarray:
        state = self.nearest_state(obs)
        if goal_state is None:
            goal_state = self.nearest_state(goal)
        if state == goal_state:
            low_goal = (
                np.asarray(goal, dtype=np.float32)
                if self.policy.stats.goal_representation == "full"
                else np.asarray(goal[:2], dtype=np.float32)
            )
        else:
            if self.path_mode == "persistent":
                target_state = self._persistent_target_state(obs, state, goal_state)
            elif self.path_mode == "greedy":
                target_state = self._target_state(state, goal_state)
            else:
                raise ValueError(f"unknown path mode {self.path_mode}")
            low_goal = (
                self._waypoint_goal_obs(obs, target_state)
                if self.policy.stats.goal_representation == "full"
                else np.asarray(self.env.unwrapped.ij_to_xy(self.topology.states[target_state]), dtype=np.float32)
            )
        return self.policy.action(obs, low_goal, self.policy_eval_backend)


def evaluate_agent(
    agent: BCTopologyAgent,
    env,
    episodes: int,
    seed: int,
    task_ids: list[int] | None,
    action_repeat: int,
    max_episode_steps: int | None,
    profile: bool = False,
) -> dict[str, float]:
    task_infos = env.unwrapped.task_infos
    selected = (
        list(enumerate(task_infos, start=1))
        if task_ids is None
        else [(task_id, task_infos[task_id - 1]) for task_id in task_ids]
    )
    row: dict[str, float] = {}
    task_means = []
    profile_reset_seconds = 0.0
    profile_action_seconds = 0.0
    profile_env_step_seconds = 0.0
    profile_env_steps = 0
    profile_policy_calls = 0
    for task_id, _task_info in selected:
        successes = []
        lengths = []
        final_dists = []
        final_xys = []
        for episode_idx in range(episodes):
            agent.reset_episode()
            episode_seed = seed * 1_000_000 + task_id * 10_000 + episode_idx
            np.random.seed(episode_seed)
            reset_start = time.perf_counter()
            obs, info = env.reset(seed=episode_seed, options={"task_id": task_id})
            profile_reset_seconds += time.perf_counter() - reset_start
            goal = info["goal"]
            goal_state = agent.nearest_state(goal)
            done = False
            length = 0
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            repeat_left = 0
            while not done:
                if repeat_left <= 0:
                    action_start = time.perf_counter()
                    action = agent.sample_action(obs, goal, goal_state)
                    profile_action_seconds += time.perf_counter() - action_start
                    profile_policy_calls += 1
                    repeat_left = max(1, action_repeat)
                step_start = time.perf_counter()
                obs, _reward, terminated, truncated, info = env.step(action)
                profile_env_step_seconds += time.perf_counter() - step_start
                profile_env_steps += 1
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
        row[f"task{task_id}_success"] = mean_success
        row[f"task{task_id}_length"] = float(np.mean(lengths))
        row[f"task{task_id}_final_dist"] = float(np.mean(final_dists))
        mean_xy = np.mean(np.stack(final_xys, axis=0), axis=0)
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
    if profile:
        row["profile_reset_seconds"] = profile_reset_seconds
        row["profile_action_seconds"] = profile_action_seconds
        row["profile_env_step_seconds"] = profile_env_step_seconds
        row["profile_env_steps"] = float(profile_env_steps)
        row["profile_policy_calls"] = float(profile_policy_calls)
        row["profile_action_ms_per_call"] = (
            1000.0 * profile_action_seconds / max(profile_policy_calls, 1)
        )
        row["profile_env_step_ms"] = 1000.0 * profile_env_step_seconds / max(profile_env_steps, 1)
    return row


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def append_progress(path: Path | None, event: dict[str, float | int | str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


@dataclass
class TransitionFitStats:
    ce: float
    kl: float
    mean_l1: float
    max_abs: float
    top1_agreement: float
    train_seconds: float
    fitted_rows: int
    fallback_rows: int


@dataclass
class ValueFitStats:
    loss: float
    mse: float
    max_abs: float
    action_agreement: float
    train_seconds: float


def transition_stats_to_payload(stats: TransitionFitStats) -> dict[str, float | int]:
    return {
        "ce": float(stats.ce),
        "kl": float(stats.kl),
        "mean_l1": float(stats.mean_l1),
        "max_abs": float(stats.max_abs),
        "top1_agreement": float(stats.top1_agreement),
        "train_seconds": float(stats.train_seconds),
        "fitted_rows": int(stats.fitted_rows),
        "fallback_rows": int(stats.fallback_rows),
    }


def transition_stats_from_payload(payload: dict[str, float | int]) -> TransitionFitStats:
    return TransitionFitStats(
        ce=float(payload["ce"]),
        kl=float(payload["kl"]),
        mean_l1=float(payload["mean_l1"]),
        max_abs=float(payload["max_abs"]),
        top1_agreement=float(payload["top1_agreement"]),
        train_seconds=float(payload["train_seconds"]),
        fitted_rows=int(payload["fitted_rows"]),
        fallback_rows=int(payload["fallback_rows"]),
    )


def value_stats_to_payload(stats: ValueFitStats) -> dict[str, float]:
    return {
        "loss": float(stats.loss),
        "mse": float(stats.mse),
        "max_abs": float(stats.max_abs),
        "action_agreement": float(stats.action_agreement),
        "train_seconds": float(stats.train_seconds),
    }


def value_stats_from_payload(payload: dict[str, float]) -> ValueFitStats:
    return ValueFitStats(
        loss=float(payload["loss"]),
        mse=float(payload["mse"]),
        max_abs=float(payload["max_abs"]),
        action_agreement=float(payload["action_agreement"]),
        train_seconds=float(payload["train_seconds"]),
    )


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def transition_metrics(
    probs: np.ndarray,
    target: np.ndarray,
    fitted_mask: np.ndarray,
    train_seconds: float,
) -> TransitionFitStats:
    eps = 1e-12
    fitted = fitted_mask[..., None]
    row_count = int(np.sum(fitted_mask))
    fallback_rows = int(fitted_mask.size - row_count)
    denom = max(row_count, 1)
    ce = float(-np.sum(fitted * target * np.log(np.maximum(probs, eps))) / denom)
    entropy = float(-np.sum(fitted * target * np.log(np.maximum(target, eps))) / denom)
    kl = max(0.0, ce - entropy)
    l1_by_row = np.sum(np.abs(probs - target), axis=-1)
    mean_l1 = float(np.sum(fitted_mask * l1_by_row) / denom)
    max_abs = float(np.max(np.abs(probs - target)))
    target_top = np.argmax(target, axis=-1)
    pred_top = np.argmax(probs, axis=-1)
    top1 = float(np.sum(fitted_mask * (target_top == pred_top)) / denom)
    return TransitionFitStats(
        ce=ce,
        kl=kl,
        mean_l1=mean_l1,
        max_abs=max_abs,
        top1_agreement=top1,
        train_seconds=train_seconds,
        fitted_rows=row_count,
        fallback_rows=fallback_rows,
    )


def build_topology_sample_targets(
    topology: Topology,
    samples_per_row: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = np.zeros_like(topology.transitions, dtype=np.float64)
    n_states, n_actions, _ = topology.transitions.shape
    for s in range(n_states):
        for a in range(n_actions):
            probs = topology.transitions[s, a]
            samples = rng.choice(n_states, size=samples_per_row, p=probs)
            counts[s, a] = np.bincount(samples, minlength=n_states)
    row_sums = counts.sum(axis=-1, keepdims=True)
    target = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0.0)
    return target, np.ones(topology.transitions.shape[:2], dtype=bool)


def infer_action_from_xy(env, state: tuple[int, int], movement_xy: np.ndarray) -> int:
    current_xy = np.asarray(env.unwrapped.ij_to_xy(state), dtype=np.float64)
    scores = []
    for di, dj in ACTION_DELTAS:
        target_xy = np.asarray(env.unwrapped.ij_to_xy((state[0] + di, state[1] + dj)), dtype=np.float64)
        direction = target_xy - current_xy
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
        scores.append(float(np.dot(np.asarray(movement_xy, dtype=np.float64), direction)))
    return int(np.argmax(scores))


def build_dataset_cell_change_targets(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros_like(topology.transitions, dtype=np.float64)
    obs = np.asarray(train["observations"], dtype=np.float32)
    valids = np.asarray(train["valids"], dtype=np.float32) > 0
    valid_idxs = np.nonzero(valids[:-1])[0]

    def obs_to_state(x: np.ndarray) -> tuple[int, int]:
        return tuple(env.unwrapped.xy_to_ij(np.asarray(x[:2], dtype=np.float64)))

    delta_to_action = {delta: idx for idx, delta in enumerate(ACTION_DELTAS)}
    for idx in valid_idxs:
        state = obs_to_state(obs[idx])
        next_state = obs_to_state(obs[idx + 1])
        if state == next_state:
            continue
        if state not in topology.state_to_idx or next_state not in topology.state_to_idx:
            continue
        s = topology.state_to_idx[state]
        sp = topology.state_to_idx[next_state]
        delta = (next_state[0] - state[0], next_state[1] - state[1])
        if abs(delta[0]) + abs(delta[1]) > 1:
            counts[s, :, sp] += 1.0
        elif delta in delta_to_action:
            counts[s, delta_to_action[delta], sp] += 1.0
        else:
            movement_xy = np.asarray(obs[idx + 1, :2] - obs[idx, :2], dtype=np.float64)
            action_id = infer_action_from_xy(env, state, movement_xy)
            counts[s, action_id, sp] += 1.0

    fitted_mask = counts.sum(axis=-1) > 0.0
    target = topology.transitions.copy()
    row_sums = counts.sum(axis=-1, keepdims=True)
    np.divide(counts, row_sums, out=target, where=row_sums > 0.0)
    return target, fitted_mask


def build_dataset_jump_change_targets(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros_like(topology.transitions, dtype=np.float64)
    obs = np.asarray(train["observations"], dtype=np.float32)
    valids = np.asarray(train["valids"], dtype=np.float32) > 0
    valid_idxs = np.nonzero(valids[:-1])[0]

    def obs_to_state(x: np.ndarray) -> tuple[int, int]:
        return tuple(env.unwrapped.xy_to_ij(np.asarray(x[:2], dtype=np.float64)))

    def jump_actions(state: tuple[int, int], movement_xy: np.ndarray) -> list[int]:
        if state in topology.jump_sources:
            return list(range(topology.n_actions))
        candidates = [
            action_id
            for action_id, (di, dj) in enumerate(ACTION_DELTAS)
            if (state[0] + di, state[1] + dj) in topology.jump_sources
        ]
        if len(candidates) <= 1:
            return candidates
        scored = []
        current_xy = np.asarray(env.unwrapped.ij_to_xy(state), dtype=np.float64)
        for action_id in candidates:
            di, dj = ACTION_DELTAS[action_id]
            target_xy = np.asarray(env.unwrapped.ij_to_xy((state[0] + di, state[1] + dj)), dtype=np.float64)
            direction = target_xy - current_xy
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            scored.append((float(np.dot(movement_xy, direction)), action_id))
        return [max(scored)[1]]

    for idx in valid_idxs:
        state = obs_to_state(obs[idx])
        next_state = obs_to_state(obs[idx + 1])
        if state not in topology.state_to_idx or next_state not in topology.state_to_idx:
            continue
        delta = (next_state[0] - state[0], next_state[1] - state[1])
        if abs(delta[0]) + abs(delta[1]) <= 1:
            continue
        actions = jump_actions(state, np.asarray(obs[idx + 1, :2] - obs[idx, :2], dtype=np.float64))
        if not actions:
            continue
        s = topology.state_to_idx[state]
        sp = topology.state_to_idx[next_state]
        for action_id in actions:
            counts[s, action_id, sp] += 1.0

    fitted_mask = counts.sum(axis=-1) > 0.0
    target = topology.transitions.copy()
    row_sums = counts.sum(axis=-1, keepdims=True)
    np.divide(counts, row_sums, out=target, where=row_sums > 0.0)
    return target, fitted_mask


def fit_transition_softmax(
    target: np.ndarray,
    fitted_mask: np.ndarray,
    steps: int,
    lr: float,
    seed: int,
    log_interval: int,
) -> tuple[np.ndarray, TransitionFitStats]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 1e-3, size=target.shape).astype(np.float64)
    m = np.zeros_like(logits)
    v = np.zeros_like(logits)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    row_weight = fitted_mask[..., None].astype(np.float64)
    denom = max(float(np.sum(fitted_mask)), 1.0)
    start = time.perf_counter()
    probs = stable_softmax(logits)
    for step in range(1, steps + 1):
        probs = stable_softmax(logits)
        grad = row_weight * (probs - target) / denom
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        logits -= lr * m_hat / (np.sqrt(v_hat) + eps)
        if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == steps):
            metrics = transition_metrics(probs, target, fitted_mask, time.perf_counter() - start)
            print(
                f"[transition] step={step} ce={metrics.ce:.6f} "
                f"kl={metrics.kl:.6f} l1={metrics.mean_l1:.6f} top1={metrics.top1_agreement:.3f}",
                flush=True,
            )
    probs = stable_softmax(logits)
    probs = np.where(fitted_mask[..., None], probs, target)
    stats = transition_metrics(probs, target, fitted_mask, time.perf_counter() - start)
    return probs, stats


def representative_observations(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
) -> np.ndarray:
    obs = np.asarray(train["observations"], dtype=np.float32)
    valids = np.asarray(train["valids"], dtype=np.float32) > 0
    obs_mean = obs.mean(axis=0)
    obs_std = np.maximum(obs.std(axis=0), 1e-3)
    fallback_obs = np.tile(obs_mean[None, :], (topology.n_states, 1)).astype(np.float64)
    fallback_obs[:, :2] = np.asarray(
        [env.unwrapped.ij_to_xy(state) for state in topology.states],
        dtype=np.float64,
    )
    rep_sums = np.zeros((topology.n_states, obs.shape[1]), dtype=np.float64)
    rep_counts = np.zeros(topology.n_states, dtype=np.float64)

    valid_idxs = np.nonzero(valids)[0]
    for idx in valid_idxs:
        state = tuple(env.unwrapped.xy_to_ij(np.asarray(obs[idx, :2], dtype=np.float64)))
        if state not in topology.state_to_idx:
            continue
        state_id = topology.state_to_idx[state]
        rep_sums[state_id] += obs[idx]
        rep_counts[state_id] += 1.0
    rep_obs = fallback_obs
    seen = rep_counts > 0.0
    rep_obs[seen] = rep_sums[seen] / rep_counts[seen, None]
    return rep_obs.astype(np.float32)


def build_raw_obs_row_features(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
) -> np.ndarray:
    obs = np.asarray(train["observations"], dtype=np.float32)
    obs_mean = obs.mean(axis=0)
    obs_std = np.maximum(obs.std(axis=0), 1e-3)
    rep_obs = representative_observations(env, train, topology)
    norm_rep_obs = (rep_obs.astype(np.float32) - obs_mean[None, :]) / obs_std[None, :]
    n_states = topology.n_states
    n_actions = topology.n_actions
    features = np.zeros((n_states, n_actions, obs.shape[1] + n_actions), dtype=np.float32)
    eye = np.eye(n_actions, dtype=np.float32)
    for action_id in range(n_actions):
        features[:, action_id, : obs.shape[1]] = norm_rep_obs
        features[:, action_id, obs.shape[1] :] = eye[action_id]
    return features


def create_transition_train_state(
    seed: int,
    feature_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
    lr: float,
) -> train_state.TrainState:
    model = RawObsTransitionMLP(hidden_dims=hidden_dims, output_dim=output_dim)
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, feature_dim), dtype=jnp.float32))["params"]
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(lr))


@jax.jit
def transition_mlp_train_step(
    state: train_state.TrainState,
    feats: jax.Array,
    targets: jax.Array,
):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, feats)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(jnp.sum(targets * log_probs, axis=-1))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


def transition_mlp_predict(
    state: train_state.TrainState,
    features: np.ndarray,
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    flat = features.reshape(n_states * n_actions, features.shape[-1])
    logits = state.apply_fn({"params": state.params}, jnp.asarray(flat, dtype=jnp.float32))
    probs = np.asarray(jax.nn.softmax(logits, axis=-1), dtype=np.float64)
    return probs.reshape(n_states, n_actions, n_states)


def fit_raw_obs_transition_mlp(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
    target: np.ndarray,
    fitted_mask: np.ndarray,
    steps: int,
    lr: float,
    seed: int,
    hidden_dims: tuple[int, ...],
    log_interval: int,
) -> tuple[np.ndarray, TransitionFitStats]:
    features = build_raw_obs_row_features(env, train, topology)
    train_features = features[fitted_mask]
    train_targets = target[fitted_mask].astype(np.float32)
    if len(train_features) == 0:
        stats = transition_metrics(target, target, fitted_mask, 0.0)
        return target.copy(), stats

    state = create_transition_train_state(
        seed,
        int(train_features.shape[-1]),
        topology.n_states,
        hidden_dims,
        lr,
    )
    feats_jax = jnp.asarray(train_features, dtype=jnp.float32)
    targets_jax = jnp.asarray(train_targets, dtype=jnp.float32)
    start = time.perf_counter()
    for step in range(1, steps + 1):
        state, loss = transition_mlp_train_step(state, feats_jax, targets_jax)
        if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == steps):
            probs = transition_mlp_predict(state, features, topology.n_states, topology.n_actions)
            probs_for_eval = np.where(fitted_mask[..., None], probs, target)
            metrics = transition_metrics(
                probs_for_eval,
                target,
                fitted_mask,
                time.perf_counter() - start,
            )
            print(
                f"[transition-mlp] step={step} ce={metrics.ce:.6f} "
                f"kl={metrics.kl:.6f} l1={metrics.mean_l1:.6f} "
                f"top1={metrics.top1_agreement:.3f} loss={float(loss):.6f}",
                flush=True,
            )

    probs = transition_mlp_predict(state, features, topology.n_states, topology.n_actions)
    probs = np.where(fitted_mask[..., None], probs, target)
    stats = transition_metrics(probs, target, fitted_mask, time.perf_counter() - start)
    return probs, stats


def cached_transition_fit(
    env_name: str,
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
    transition_model: str,
    transition_target_source: str,
    samples_per_row: int,
    steps: int,
    transition_lr: float,
    transition_mlp_lr: float,
    seed: int,
    hidden_dims: tuple[int, ...],
    log_interval: int,
    cache_dir: Path | None,
    use_cache: bool,
) -> tuple[np.ndarray, TransitionFitStats, TransitionFitStats, float, int]:
    lr = transition_lr if transition_model == "learned_softmax" else transition_mlp_lr
    hidden_token = "x".join(map(str, hidden_dims))
    expected = {
        "version": CACHE_VERSION,
        "kind": "antmaze_transition_fit",
        "env_name": env_name,
        "transition_model": transition_model,
        "transition_target_source": transition_target_source,
        "samples_per_row": int(samples_per_row),
        "steps": int(steps),
        "lr": float(lr),
        "seed": int(seed),
        "hidden_dims": tuple(int(x) for x in hidden_dims),
        "obs_shape": tuple(int(x) for x in np.asarray(train["observations"]).shape),
        "states": tuple(tuple(x) for x in topology.states),
    }
    cache_path = None
    if cache_dir is not None:
        cache_path = (
            cache_dir
            / (
                f"antmaze_transition_v{CACHE_VERSION}_{safe_cache_token(env_name)}_"
                f"{safe_cache_token(transition_model)}_{safe_cache_token(transition_target_source)}_"
                f"seed{seed}_steps{steps}_lr{cache_float_token(lr)}_h{hidden_token}.pkl"
            )
        )

    start = time.perf_counter()
    if use_cache and cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        if cache_matches(payload, expected):
            elapsed = time.perf_counter() - start
            print(f"[cache] transition hit path={cache_path} seconds={elapsed:.2f}", flush=True)
            return (
                np.asarray(payload["planning_transitions"], dtype=np.float64),
                transition_stats_from_payload(payload["transition_fit_stats"]),
                transition_stats_from_payload(payload["transition_oracle_stats"]),
                elapsed,
                1,
            )
        print(f"[cache] transition stale path={cache_path}", flush=True)

    if transition_target_source == "topology":
        transition_target = topology.transitions.copy()
        fitted_mask = np.ones(topology.transitions.shape[:2], dtype=bool)
    elif transition_target_source == "topology_samples":
        transition_target, fitted_mask = build_topology_sample_targets(
            topology,
            samples_per_row,
            seed,
        )
    elif transition_target_source == "dataset_cell_changes":
        transition_target, fitted_mask = build_dataset_cell_change_targets(env, train, topology)
    elif transition_target_source == "dataset_jump_changes":
        transition_target, fitted_mask = build_dataset_jump_change_targets(env, train, topology)
    else:
        raise ValueError(f"unknown transition target source {transition_target_source}")

    if transition_model == "learned_softmax":
        planning_transitions, transition_fit_stats = fit_transition_softmax(
            transition_target,
            fitted_mask,
            steps,
            transition_lr,
            seed,
            log_interval,
        )
    elif transition_model == "raw_obs_mlp":
        planning_transitions, transition_fit_stats = fit_raw_obs_transition_mlp(
            env,
            train,
            topology,
            transition_target,
            fitted_mask,
            steps,
            transition_mlp_lr,
            seed,
            hidden_dims,
            log_interval,
        )
    else:
        raise ValueError(f"unknown transition model {transition_model}")

    row_sums = planning_transitions.sum(axis=-1, keepdims=True)
    planning_transitions = np.divide(
        planning_transitions,
        row_sums,
        out=np.zeros_like(planning_transitions),
        where=row_sums > 0.0,
    )
    transition_oracle_stats = transition_metrics(
        planning_transitions,
        topology.transitions,
        np.ones(topology.transitions.shape[:2], dtype=bool),
        transition_fit_stats.train_seconds,
    )
    elapsed = time.perf_counter() - start
    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(
                {
                    "meta": expected,
                    "planning_transitions": planning_transitions,
                    "transition_fit_stats": transition_stats_to_payload(transition_fit_stats),
                    "transition_oracle_stats": transition_stats_to_payload(transition_oracle_stats),
                },
                f,
            )
        print(f"[cache] transition wrote path={cache_path} seconds={elapsed:.2f}", flush=True)
    else:
        print(f"[cache] transition disabled seconds={elapsed:.2f}", flush=True)
    return planning_transitions, transition_fit_stats, transition_oracle_stats, elapsed, 0


def build_raw_obs_policy_features(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
) -> np.ndarray:
    obs = np.asarray(train["observations"], dtype=np.float32)
    obs_mean = obs.mean(axis=0)
    obs_std = np.maximum(obs.std(axis=0), 1e-3)
    rep_obs = representative_observations(env, train, topology)
    norm_rep_obs = (rep_obs - obs_mean[None, :]) / obs_std[None, :]
    features = np.zeros((topology.n_states, topology.n_states, obs.shape[1] * 3), dtype=np.float32)
    features[:, :, : obs.shape[1]] = norm_rep_obs[:, None, :]
    features[:, :, obs.shape[1] : 2 * obs.shape[1]] = norm_rep_obs[None, :, :]
    features[:, :, 2 * obs.shape[1] :] = norm_rep_obs[None, :, :] - norm_rep_obs[:, None, :]
    return features


def create_highlevel_policy_train_state(
    seed: int,
    feature_dim: int,
    n_actions: int,
    hidden_dims: tuple[int, ...],
    lr: float,
) -> train_state.TrainState:
    model = RawObsPolicyMLP(hidden_dims=hidden_dims, n_actions=n_actions)
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, feature_dim), dtype=jnp.float32))["params"]
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(lr))


@jax.jit
def tie_policy_mlp_train_step(
    state: train_state.TrainState,
    feats: jax.Array,
    targets: jax.Array,
    mask: jax.Array,
):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, feats)
        bce = optax.sigmoid_binary_cross_entropy(logits, targets)
        row_loss = jnp.mean(bce, axis=-1)
        return jnp.sum(row_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


def policy_mlp_predict_sigmoid(
    state: train_state.TrainState,
    features: np.ndarray,
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    flat = features.reshape(n_states * n_states, features.shape[-1])
    logits = state.apply_fn({"params": state.params}, jnp.asarray(flat, dtype=jnp.float32))
    probs = np.asarray(jax.nn.sigmoid(logits), dtype=np.float64).reshape(n_states, n_states, n_actions)
    return probs.transpose(0, 2, 1)


def value_metrics(pred: np.ndarray, target: np.ndarray) -> ValueFitStats:
    diff = pred - target
    target_actions = np.argmax(target, axis=1)
    pred_actions = np.argmax(pred, axis=1)
    mask = ~np.eye(target.shape[0], dtype=bool)
    action_agreement = float(np.mean((target_actions == pred_actions)[mask])) if np.any(mask) else 1.0
    return ValueFitStats(
        loss=float(np.mean(diff * diff)),
        mse=float(np.mean(diff * diff)),
        max_abs=float(np.max(np.abs(diff))),
        action_agreement=action_agreement,
        train_seconds=0.0,
    )


def fit_raw_obs_tie_policy_mlp(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
    target_q: np.ndarray,
    steps: int,
    lr: float,
    seed: int,
    hidden_dims: tuple[int, ...],
    log_interval: int,
    method: str,
    tie_tol: float,
) -> tuple[np.ndarray, ValueFitStats]:
    features = build_raw_obs_policy_features(env, train, topology)
    max_q = target_q.max(axis=1, keepdims=True)
    tie_targets = (target_q >= (max_q - tie_tol)).transpose(0, 2, 1).astype(np.float32)
    action_mask = (~np.eye(topology.n_states, dtype=bool)).astype(np.float32)
    state = create_highlevel_policy_train_state(
        seed,
        int(features.shape[-1]),
        topology.n_actions,
        hidden_dims,
        lr,
    )
    feats_jax = jnp.asarray(features.reshape(-1, features.shape[-1]), dtype=jnp.float32)
    targets_jax = jnp.asarray(tie_targets.reshape(-1, topology.n_actions), dtype=jnp.float32)
    mask_jax = jnp.asarray(action_mask.reshape(-1), dtype=jnp.float32)
    start = time.perf_counter()
    pred_probs = policy_mlp_predict_sigmoid(state, features, topology.n_states, topology.n_actions)
    for step in range(1, steps + 1):
        state, loss = tie_policy_mlp_train_step(state, feats_jax, targets_jax, mask_jax)
        if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == steps):
            pred_probs = policy_mlp_predict_sigmoid(state, features, topology.n_states, topology.n_actions)
            pred_mask = (pred_probs >= 0.5).astype(np.float64)
            empty = pred_mask.sum(axis=1, keepdims=True) == 0.0
            fallback = np.zeros_like(pred_mask)
            np.put_along_axis(fallback, np.argmax(pred_probs, axis=1, keepdims=True), 1.0, axis=1)
            pred_q = np.where(empty, fallback, pred_mask)
            metrics = value_metrics(pred_q, target_q)
            target_ties = tie_targets.transpose(0, 2, 1)
            tie_exact = np.all(pred_q == target_ties, axis=1)
            tie_agreement = float(np.mean(tie_exact[action_mask.astype(bool)]))
            print(
                f"[tie-policy-mlp] method={method} step={step} mse={metrics.mse:.6f} "
                f"max_abs={metrics.max_abs:.6f} action_agree={metrics.action_agreement:.3f} "
                f"tie_agree={tie_agreement:.3f} loss={float(loss):.6f}",
                flush=True,
            )
    pred_probs = policy_mlp_predict_sigmoid(state, features, topology.n_states, topology.n_actions)
    pred_mask = (pred_probs >= 0.5).astype(np.float64)
    empty = pred_mask.sum(axis=1, keepdims=True) == 0.0
    fallback = np.zeros_like(pred_mask)
    np.put_along_axis(fallback, np.argmax(pred_probs, axis=1, keepdims=True), 1.0, axis=1)
    pred_q = np.where(empty, fallback, pred_mask)
    for goal_state in range(topology.n_states):
        pred_q[goal_state, :, goal_state] = 1.0
    stats = value_metrics(pred_q, target_q)
    stats.train_seconds = time.perf_counter() - start
    stats.loss = float(stats.mse)
    return pred_q, stats


def build_raw_obs_prev_policy_features(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
) -> np.ndarray:
    base = build_raw_obs_policy_features(env, train, topology)
    n_prev = topology.n_actions + 1
    prev_eye = np.eye(n_prev, dtype=np.float32)
    features = np.zeros(
        (topology.n_states, n_prev, topology.n_states, base.shape[-1] + n_prev),
        dtype=np.float32,
    )
    features[..., : base.shape[-1]] = base[:, None, :, :]
    features[..., base.shape[-1] :] = prev_eye[None, :, None, :]
    return features


def previous_action_policy_labels(target_q: np.ndarray, n_actions: int) -> np.ndarray:
    n_states = target_q.shape[0]
    labels = np.zeros((n_states, n_actions + 1, n_states), dtype=np.int32)
    for prev_idx in range(n_actions + 1):
        values = target_q.copy()
        if prev_idx < n_actions:
            values[:, prev_idx, :] += 1e-6
        labels[:, prev_idx, :] = np.argmax(values, axis=1)
    return labels


@jax.jit
def single_policy_mlp_train_step(
    state: train_state.TrainState,
    feats: jax.Array,
    labels: jax.Array,
    mask: jax.Array,
):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, feats)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(labels, logits.shape[-1])
        row_loss = -jnp.sum(one_hot * log_probs, axis=-1)
        return jnp.sum(row_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


def prev_policy_mlp_predict_probs(
    state: train_state.TrainState,
    features: np.ndarray,
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    n_prev = n_actions + 1
    flat = features.reshape(n_states * n_prev * n_states, features.shape[-1])
    logits = state.apply_fn({"params": state.params}, jnp.asarray(flat, dtype=jnp.float32))
    probs = np.asarray(jax.nn.softmax(logits, axis=-1), dtype=np.float64)
    return probs.reshape(n_states, n_prev, n_states, n_actions).transpose(0, 1, 3, 2)


def prev_policy_metrics(pred_scores: np.ndarray, target_q: np.ndarray, n_actions: int) -> ValueFitStats:
    labels = previous_action_policy_labels(target_q, n_actions)
    pred_actions = np.argmax(pred_scores, axis=2)
    mask = np.broadcast_to(~np.eye(target_q.shape[0], dtype=bool)[:, None, :], labels.shape)
    action_agreement = float(np.mean((pred_actions == labels)[mask])) if np.any(mask) else 1.0
    target_onehot = np.zeros_like(pred_scores)
    np.put_along_axis(target_onehot, labels[:, :, None, :], 1.0, axis=2)
    diff = pred_scores - target_onehot
    return ValueFitStats(
        loss=float(np.mean(diff * diff)),
        mse=float(np.mean(diff * diff)),
        max_abs=float(np.max(np.abs(diff))),
        action_agreement=action_agreement,
        train_seconds=0.0,
    )


def fit_raw_obs_prev_policy_mlp(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
    target_q: np.ndarray,
    steps: int,
    lr: float,
    seed: int,
    hidden_dims: tuple[int, ...],
    log_interval: int,
    method: str,
) -> tuple[np.ndarray, ValueFitStats]:
    features = build_raw_obs_prev_policy_features(env, train, topology)
    labels = previous_action_policy_labels(target_q, topology.n_actions)
    action_mask = np.broadcast_to(
        (~np.eye(topology.n_states, dtype=bool))[:, None, :],
        labels.shape,
    ).astype(np.float32)
    state = create_highlevel_policy_train_state(
        seed,
        int(features.shape[-1]),
        topology.n_actions,
        hidden_dims,
        lr,
    )
    feats_jax = jnp.asarray(features.reshape(-1, features.shape[-1]), dtype=jnp.float32)
    labels_jax = jnp.asarray(labels.reshape(-1), dtype=jnp.int32)
    mask_jax = jnp.asarray(action_mask.reshape(-1), dtype=jnp.float32)
    start = time.perf_counter()
    pred_scores = prev_policy_mlp_predict_probs(state, features, topology.n_states, topology.n_actions)
    for step in range(1, steps + 1):
        state, loss = single_policy_mlp_train_step(state, feats_jax, labels_jax, mask_jax)
        if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == steps):
            pred_scores = prev_policy_mlp_predict_probs(state, features, topology.n_states, topology.n_actions)
            metrics = prev_policy_metrics(pred_scores, target_q, topology.n_actions)
            print(
                f"[prev-policy-mlp] method={method} step={step} mse={metrics.mse:.6f} "
                f"max_abs={metrics.max_abs:.6f} action_agree={metrics.action_agreement:.3f} "
                f"loss={float(loss):.6f}",
                flush=True,
            )
    pred_scores = prev_policy_mlp_predict_probs(state, features, topology.n_states, topology.n_actions)
    for goal_state in range(topology.n_states):
        pred_scores[goal_state, :, :, goal_state] = 1.0 / topology.n_actions
    stats = prev_policy_metrics(pred_scores, target_q, topology.n_actions)
    stats.train_seconds = time.perf_counter() - start
    stats.loss = float(stats.mse)
    return pred_scores, stats


def cached_value_model_fit(
    env_name: str,
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
    value_model: str,
    method: str,
    target_q: np.ndarray,
    steps: int,
    lr: float,
    seed: int,
    hidden_dims: tuple[int, ...],
    log_interval: int,
    tie_tol: float,
    cache_dir: Path | None,
    use_cache: bool,
) -> tuple[np.ndarray, ValueFitStats, int]:
    target_digest = array_digest(target_q)
    hidden_token = "x".join(map(str, hidden_dims))
    expected = {
        "version": CACHE_VERSION,
        "kind": "antmaze_value_model_fit",
        "env_name": env_name,
        "value_model": value_model,
        "method": method,
        "target_q_digest": target_digest,
        "target_q_shape": tuple(int(x) for x in target_q.shape),
        "steps": int(steps),
        "lr": float(lr),
        "seed": int(seed),
        "hidden_dims": tuple(int(x) for x in hidden_dims),
        "tie_tol": float(tie_tol),
        "obs_shape": tuple(int(x) for x in np.asarray(train["observations"]).shape),
        "states": tuple(tuple(x) for x in topology.states),
    }
    cache_path = None
    if cache_dir is not None:
        cache_path = (
            cache_dir
            / (
                f"antmaze_value_v{CACHE_VERSION}_{safe_cache_token(env_name)}_"
                f"{safe_cache_token(value_model)}_{safe_cache_token(method)}_"
                f"{target_digest[:12]}_seed{seed}_steps{steps}_"
                f"lr{cache_float_token(lr)}_h{hidden_token}.pkl"
            )
        )

    start = time.perf_counter()
    if use_cache and cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        if cache_matches(payload, expected):
            elapsed = time.perf_counter() - start
            print(
                f"[cache] value hit method={method} model={value_model} "
                f"path={cache_path} seconds={elapsed:.2f}",
                flush=True,
            )
            return (
                np.asarray(payload["q"], dtype=np.float64),
                value_stats_from_payload(payload["value_stats"]),
                1,
            )
        print(f"[cache] value stale path={cache_path}", flush=True)

    if value_model == "raw_obs_tie_policy_mlp":
        fitted_q, value_stats = fit_raw_obs_tie_policy_mlp(
            env,
            train,
            topology,
            target_q,
            steps,
            lr,
            seed,
            hidden_dims,
            log_interval,
            method,
            tie_tol,
        )
    elif value_model == "raw_obs_prev_policy_mlp":
        fitted_q, value_stats = fit_raw_obs_prev_policy_mlp(
            env,
            train,
            topology,
            target_q,
            steps,
            lr,
            seed,
            hidden_dims,
            log_interval,
            method,
        )
    else:
        raise ValueError(f"unknown cached value model {value_model}")

    elapsed = time.perf_counter() - start
    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(
                {
                    "meta": expected,
                    "q": fitted_q,
                    "value_stats": value_stats_to_payload(value_stats),
                },
                f,
            )
        print(
            f"[cache] value wrote method={method} model={value_model} "
            f"path={cache_path} seconds={elapsed:.2f}",
            flush=True,
        )
    else:
        print(f"[cache] value disabled method={method} model={value_model} seconds={elapsed:.2f}", flush=True)
    return fitted_q, value_stats, 0


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


def greedy_path(
    topology: Topology,
    q: np.ndarray,
    start_state: int,
    goal_state: int,
    max_len: int = 64,
) -> list[int]:
    path = [start_state]
    state = start_state
    previous_action_id: int | None = None
    for _ in range(max_len):
        if state == goal_state:
            break
        if q.ndim == 4:
            prev_idx = topology.n_actions if previous_action_id is None else previous_action_id
            values = q[state, prev_idx, :, goal_state]
        else:
            values = q[state, :, goal_state].copy()
            if previous_action_id is not None:
                values[previous_action_id] += 1e-6
        action_id = int(np.argmax(values))
        previous_action_id = action_id
        nxt = int(topology.intended_next[state, action_id])
        path.append(nxt)
        if nxt == state:
            break
        state = nxt
    return path


def resolve_value_method(method: str, matched_iters: int, full_iters: int) -> tuple[int, bool]:
    if method == "bellman_matched":
        return matched_iters, False
    if method == "sto_trl_matched":
        return matched_iters, True
    if method == "bellman_full":
        return full_iters, False
    if method.startswith("bellman_") and method.endswith("_sweeps"):
        parts = method.split("_")
        if len(parts) == 3:
            return int(parts[1]), False
    if method.startswith("sto_trl_") and method.endswith("_sweeps"):
        parts = method.split("_")
        if len(parts) == 4:
            return int(parts[2]), True
    raise ValueError(
        f"unknown method {method}; expected bellman_matched, sto_trl_matched, "
        "bellman_full, support_trl_matched, bellman_<n>_sweeps, or sto_trl_<n>_sweeps"
    )


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
    parser.add_argument(
        "--transition-model",
        choices=["topology", "learned_softmax", "raw_obs_mlp"],
        default="topology",
    )
    parser.add_argument(
        "--transition-target-source",
        choices=["topology", "topology_samples", "dataset_cell_changes", "dataset_jump_changes"],
        default="topology_samples",
    )
    parser.add_argument("--samples-per-row", type=int, default=20)
    parser.add_argument("--transition-steps", type=int, default=1_000)
    parser.add_argument("--transition-lr", type=float, default=0.3)
    parser.add_argument("--transition-mlp-lr", type=float, default=3e-3)
    parser.add_argument("--transition-hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--transition-seed", type=int, default=0)
    parser.add_argument("--transition-log-interval", type=int, default=500)
    parser.add_argument(
        "--value-model",
        choices=["table", "raw_obs_tie_policy_mlp", "raw_obs_prev_policy_mlp"],
        default="table",
    )
    parser.add_argument("--value-steps", type=int, default=2_000)
    parser.add_argument("--value-mlp-lr", type=float, default=3e-3)
    parser.add_argument("--value-hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--value-seed", type=int, default=0)
    parser.add_argument("--value-log-interval", type=int, default=1_000)
    parser.add_argument("--value-tie-tol", type=float, default=1e-9)
    parser.add_argument("--bc-steps", type=int, default=5000)
    parser.add_argument("--bc-batch-size", type=int, default=1024)
    parser.add_argument("--bc-min-future", type=int, default=5)
    parser.add_argument("--bc-max-future", type=int, default=80)
    parser.add_argument("--bc-hidden-dims", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--bc-lr", type=float, default=3e-4)
    parser.add_argument("--bc-rel-scale", type=float, default=None)
    parser.add_argument("--goal-representation", choices=["xy", "full"], default="xy")
    parser.add_argument("--bc-seed", type=int, default=0)
    parser.add_argument("--bc-log-interval", type=int, default=500)
    parser.add_argument("--no-bc-layer-norm", action="store_true")
    parser.add_argument("--save-policy", type=Path, default=None)
    parser.add_argument("--load-policy", type=Path, default=None)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--waypoint-lookahead", type=int, default=2)
    parser.add_argument("--path-mode", choices=["greedy", "persistent"], default="greedy")
    parser.add_argument("--advance-distance", type=float, default=1.5)
    parser.add_argument("--goal-candidates-per-state", type=int, default=1)
    parser.add_argument("--goal-candidate-mode", choices=["nearest_xy", "body_nearest"], default="nearest_xy")
    parser.add_argument("--eval-action-repeat", type=int, default=1)
    parser.add_argument(
        "--policy-eval-backend",
        choices=["jax", "numpy"],
        default="jax",
        help=(
            "Use jax for exact claimed evaluations. The numpy backend is faster but can "
            "change long-horizon MuJoCo rollouts through tiny numerical drift."
        ),
    )
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--profile-eval", action="store_true")
    parser.add_argument("--print-paths", action="store_true")
    parser.add_argument("--print-paths-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results" / "cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bellman_matched", "sto_trl_matched", "bellman_full"],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "antmaze_bc_topology_planner.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=ROOT / "results" / "antmaze_bc_topology_planner_summary.json",
    )
    parser.add_argument("--progress-log", type=Path, default=None)
    args = parser.parse_args()
    if args.progress_log is not None:
        args.progress_log.parent.mkdir(parents=True, exist_ok=True)
        args.progress_log.write_text("")

    if args.train_only:
        if args.load_policy is not None:
            raise ValueError("--train-only cannot be combined with --load-policy")
        if args.save_policy is None:
            raise ValueError("--train-only requires --save-policy")

    wall_start = time.perf_counter()
    env_start = time.perf_counter()
    env, train, val = make_env_and_datasets(args.env_name, dataset_path=None)
    env_load_seconds = time.perf_counter() - env_start
    print(f"[time] env_load seconds={env_load_seconds:.2f}", flush=True)
    if args.train_only:
        policy = train_bc_policy(
            train=train,
            val=val,
            seed=args.bc_seed,
            steps=args.bc_steps,
            batch_size=args.bc_batch_size,
            min_future=args.bc_min_future,
            max_future=args.bc_max_future,
            hidden_dims=tuple(args.bc_hidden_dims),
            lr=args.bc_lr,
            rel_scale=args.bc_rel_scale,
            goal_representation=args.goal_representation,
            layer_norm=not args.no_bc_layer_norm,
            log_interval=args.bc_log_interval,
            progress_log=args.progress_log,
        )
        save_bc_policy(
            args.save_policy,
            policy,
            tuple(args.bc_hidden_dims),
            args.bc_lr,
            not args.no_bc_layer_norm,
        )
        summary = {
            "train_only": True,
            "env": args.env_name,
            "save_policy": str(args.save_policy),
            "metadata": policy.metadata,
        }
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
        append_progress(args.progress_log, {"type": "train_only_complete", **policy.metadata})
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return

    use_cache = not args.no_cache
    topology, topology_seconds, topology_cache_hit = cached_dataset_topology(
        args.env_name,
        env,
        train,
        args.min_jump_count,
        args.cache_dir,
        use_cache,
    )
    matched_iters = args.iters
    if matched_iters is None:
        matched_iters = max(1, int(math.ceil(math.log2(max(topology.n_states, 2)))))
    print(
        f"[topology] states={topology.n_states} actions={topology.n_actions} "
        f"matched_iters={matched_iters}",
        flush=True,
    )

    transition_fit_stats: TransitionFitStats | None = None
    transition_oracle_stats: TransitionFitStats | None = None
    planning_transitions = topology.transitions
    transition_seconds = 0.0
    transition_cache_hit = 0
    if args.transition_model in {"learned_softmax", "raw_obs_mlp"}:
        (
            planning_transitions,
            transition_fit_stats,
            transition_oracle_stats,
            transition_seconds,
            transition_cache_hit,
        ) = cached_transition_fit(
            args.env_name,
            env,
            train,
            topology,
            args.transition_model,
            args.transition_target_source,
            args.samples_per_row,
            args.transition_steps,
            args.transition_lr,
            args.transition_mlp_lr,
            args.transition_seed,
            tuple(args.transition_hidden_dims),
            args.transition_log_interval,
            args.cache_dir,
            use_cache,
        )
        print(
            f"[transition] final ce={transition_fit_stats.ce:.6f} "
            f"kl={transition_fit_stats.kl:.6f} l1={transition_fit_stats.mean_l1:.6f} "
            f"max_abs={transition_fit_stats.max_abs:.6f} top1={transition_fit_stats.top1_agreement:.3f}",
            flush=True,
        )
        print(
            f"[transition-oracle] kl={transition_oracle_stats.kl:.6f} "
            f"l1={transition_oracle_stats.mean_l1:.6f} "
            f"max_abs={transition_oracle_stats.max_abs:.6f} "
            f"top1={transition_oracle_stats.top1_agreement:.3f}",
            flush=True,
        )

    q_by_method: dict[str, np.ndarray] = {}
    iters_by_method: dict[str, int] = {}
    value_fit_by_method: dict[str, ValueFitStats] = {}
    value_cache_hit_by_method: dict[str, int] = {}
    value_cache: dict[tuple[int, bool], np.ndarray] = {}
    value_start = time.perf_counter()
    for method in dict.fromkeys(args.methods):
        if method == "support_trl_matched":
            value_iters = matched_iters
            table_q = train_support_transitive_value(planning_transitions, args.gamma, value_iters)
        else:
            value_iters, stochastic = resolve_value_method(method, matched_iters, args.full_iters)
            cache_key = (value_iters, stochastic)
            if cache_key not in value_cache:
                value_cache[cache_key] = train_model_value(planning_transitions, args.gamma, value_iters, stochastic)
            table_q = value_cache[cache_key]
        if args.value_model == "raw_obs_tie_policy_mlp":
            fitted_q, value_stats, cache_hit = cached_value_model_fit(
                args.env_name,
                env,
                train,
                topology,
                args.value_model,
                method,
                table_q,
                args.value_steps,
                args.value_mlp_lr,
                args.value_seed,
                tuple(args.value_hidden_dims),
                args.value_log_interval,
                args.value_tie_tol,
                args.cache_dir,
                use_cache,
            )
            q_by_method[method] = fitted_q
            value_fit_by_method[method] = value_stats
            value_cache_hit_by_method[method] = cache_hit
        elif args.value_model == "raw_obs_prev_policy_mlp":
            fitted_q, value_stats, cache_hit = cached_value_model_fit(
                args.env_name,
                env,
                train,
                topology,
                args.value_model,
                method,
                table_q,
                args.value_steps,
                args.value_mlp_lr,
                args.value_seed,
                tuple(args.value_hidden_dims),
                args.value_log_interval,
                args.value_tie_tol,
                args.cache_dir,
                use_cache,
            )
            q_by_method[method] = fitted_q
            value_fit_by_method[method] = value_stats
            value_cache_hit_by_method[method] = cache_hit
        else:
            q_by_method[method] = table_q
            value_fit_by_method[method] = ValueFitStats(
                loss=0.0,
                mse=0.0,
                max_abs=0.0,
                action_agreement=1.0,
                train_seconds=0.0,
            )
            value_cache_hit_by_method[method] = 0
        iters_by_method[method] = value_iters
    value_seconds = time.perf_counter() - value_start
    print(f"[time] value_training seconds={value_seconds:.2f}", flush=True)

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

    policy_start = time.perf_counter()
    if args.load_policy is not None:
        policy = load_bc_policy(args.load_policy)
        print(
            f"[bc] loaded_policy={args.load_policy} "
            f"metadata={json.dumps(policy.metadata, sort_keys=True, default=str)}",
            flush=True,
        )
    else:
        policy = train_bc_policy(
            train=train,
            val=val,
            seed=args.bc_seed,
            steps=args.bc_steps,
            batch_size=args.bc_batch_size,
            min_future=args.bc_min_future,
            max_future=args.bc_max_future,
            hidden_dims=tuple(args.bc_hidden_dims),
            lr=args.bc_lr,
            rel_scale=args.bc_rel_scale,
            goal_representation=args.goal_representation,
            layer_norm=not args.no_bc_layer_norm,
            log_interval=args.bc_log_interval,
            progress_log=args.progress_log,
        )
        if args.save_policy is not None:
            save_bc_policy(
                args.save_policy,
                policy,
                tuple(args.bc_hidden_dims),
                args.bc_lr,
                not args.no_bc_layer_norm,
            )
            print(f"[bc] saved_policy={args.save_policy}", flush=True)
    policy_seconds = time.perf_counter() - policy_start
    print(f"[time] policy_ready seconds={policy_seconds:.2f}", flush=True)
    goal_index, goal_index_seconds, goal_index_cache_hit = cached_goal_observation_index(
        args.env_name,
        env,
        train,
        topology,
        args.goal_candidates_per_state,
        args.cache_dir,
        use_cache,
    )
    setup_seconds = time.perf_counter() - wall_start
    print(f"[time] setup_total seconds={setup_seconds:.2f}", flush=True)
    reported_bc_steps = int(policy.metadata.get("bc_steps", args.bc_steps))
    reported_bc_batch_size = int(policy.metadata.get("bc_batch_size", args.bc_batch_size))
    reported_bc_min_future = int(policy.metadata.get("bc_min_future", args.bc_min_future))
    reported_bc_max_future = int(policy.metadata.get("bc_max_future", args.bc_max_future))
    reported_bc_seed = int(policy.metadata.get("bc_seed", args.bc_seed))
    reported_bc_hidden_dims = tuple(int(x) for x in policy.metadata.get("bc_hidden_dims", args.bc_hidden_dims))
    reported_bc_lr = float(policy.metadata.get("bc_lr", args.bc_lr))
    reported_bc_layer_norm = bool(policy.metadata.get("bc_layer_norm", not args.no_bc_layer_norm))
    reported_goal_representation = str(policy.metadata.get("goal_representation", args.goal_representation))

    rows: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        for method in args.methods:
            print(f"[run] seed={seed} method={method}", flush=True)
            agent = BCTopologyAgent(
                env,
                topology,
                q_by_method[method],
                policy,
                goal_index,
                args.waypoint_lookahead,
                args.path_mode,
                args.advance_distance,
                args.goal_candidate_mode,
                args.policy_eval_backend,
            )
            eval_start = time.perf_counter()
            metrics = evaluate_agent(
                agent,
                env,
                args.episodes,
                seed,
                args.task_ids,
                args.eval_action_repeat,
                args.max_episode_steps,
                args.profile_eval,
            )
            eval_seconds = time.perf_counter() - eval_start
            print(f"[time] eval seed={seed} method={method} seconds={eval_seconds:.2f}", flush=True)
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
                "transition_model": args.transition_model,
                "transition_target_source": args.transition_target_source,
                "transition_samples_per_row": args.samples_per_row,
                "transition_steps": args.transition_steps,
                "transition_lr": args.transition_lr,
                "transition_mlp_lr": args.transition_mlp_lr,
                "transition_hidden_dims": ",".join(map(str, args.transition_hidden_dims)),
                "transition_seed": args.transition_seed,
                "transition_seconds": transition_seconds,
                "transition_cache_hit": transition_cache_hit,
                "transition_ce": "" if transition_fit_stats is None else transition_fit_stats.ce,
                "transition_kl": "" if transition_fit_stats is None else transition_fit_stats.kl,
                "transition_l1": "" if transition_fit_stats is None else transition_fit_stats.mean_l1,
                "transition_max_abs": "" if transition_fit_stats is None else transition_fit_stats.max_abs,
                "transition_top1": "" if transition_fit_stats is None else transition_fit_stats.top1_agreement,
                "transition_oracle_kl": "" if transition_oracle_stats is None else transition_oracle_stats.kl,
                "transition_oracle_l1": "" if transition_oracle_stats is None else transition_oracle_stats.mean_l1,
                "transition_oracle_max_abs": "" if transition_oracle_stats is None else transition_oracle_stats.max_abs,
                "transition_oracle_top1": "" if transition_oracle_stats is None else transition_oracle_stats.top1_agreement,
                "value_model": args.value_model,
                "value_steps": args.value_steps,
                "value_mlp_lr": args.value_mlp_lr,
                "value_hidden_dims": ",".join(map(str, args.value_hidden_dims)),
                "value_seed": args.value_seed,
                "value_tie_tol": args.value_tie_tol,
                "value_cache_hit": value_cache_hit_by_method[method],
                "value_loss": value_fit_by_method[method].loss,
                "value_mse": value_fit_by_method[method].mse,
                "value_max_abs": value_fit_by_method[method].max_abs,
                "value_action_agreement": value_fit_by_method[method].action_agreement,
                "value_train_seconds": value_fit_by_method[method].train_seconds,
                "bc_steps": reported_bc_steps,
                "bc_batch_size": reported_bc_batch_size,
                "bc_min_future": reported_bc_min_future,
                "bc_max_future": reported_bc_max_future,
                "bc_seed": reported_bc_seed,
                "bc_hidden_dims": ",".join(map(str, reported_bc_hidden_dims)),
                "bc_lr": reported_bc_lr,
                "bc_rel_scale": policy.stats.rel_scale,
                "goal_representation": reported_goal_representation,
                "bc_layer_norm": int(reported_bc_layer_norm),
                "waypoint_lookahead": args.waypoint_lookahead,
                "path_mode": args.path_mode,
                "advance_distance": args.advance_distance,
                "goal_candidates_per_state": args.goal_candidates_per_state,
                "goal_candidate_mode": args.goal_candidate_mode,
                "eval_action_repeat": args.eval_action_repeat,
                "policy_eval_backend": args.policy_eval_backend,
                "max_episode_steps": "" if args.max_episode_steps is None else args.max_episode_steps,
                "env_load_seconds": env_load_seconds,
                "topology_seconds": topology_seconds,
                "topology_cache_hit": topology_cache_hit,
                "value_seconds": value_seconds,
                "policy_seconds": policy_seconds,
                "goal_index_seconds": goal_index_seconds,
                "goal_index_cache_hit": goal_index_cache_hit,
                "setup_seconds": setup_seconds,
                "eval_seconds": eval_seconds,
                **metrics,
            }
            rows.append(row)
            write_csv(rows, args.out)
            args.summary_out.parent.mkdir(parents=True, exist_ok=True)
            args.summary_out.write_text(json.dumps(summarize(rows), indent=2, sort_keys=True))
            append_progress(args.progress_log, {"type": "eval_row", **row})
    print(json.dumps(summarize(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
