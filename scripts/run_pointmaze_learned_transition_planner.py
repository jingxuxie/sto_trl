#!/usr/bin/env python3
"""Learned-transition diagnostic for PointMaze stochastic TRL.

This script fits a small differentiable categorical transition model on top of
the inferred PointMaze cell MDP, then runs the same matched-budget Bellman and
stochastic-TRL planners using the learned transition probabilities. It is a
fast bridge between the current topology diagnostic and a fully neural model:
the executor and cell abstraction are still shared with the topology runner,
but the transition table used by the value backup is learned by optimization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
SCRIPT_DIR = ROOT / "scripts"
TRL_DIR = ROOT / "external" / "trl"
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
from flax.training import train_state
from envs.env_utils import make_env_and_datasets
from run_pointmaze_topology_planner import (
    ACTION_DELTAS,
    Topology,
    build_dataset_topology,
    build_topology,
    evaluate,
    evaluate_model,
    summarize,
    write_csv,
)
from sto_trl.learners import train_model_value


@dataclass
class TransitionFitStats:
    loss: float
    ce: float
    kl: float
    mean_l1: float
    max_abs: float
    top1_agreement: float
    train_seconds: float
    target_source: str
    fitted_rows: int
    fallback_rows: int


@dataclass
class ValueFitStats:
    loss: float
    mse: float
    max_abs: float
    action_agreement: float
    train_seconds: float


class RawObsTransitionMLP(nn.Module):
    hidden_dims: tuple[int, ...]
    output_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        return nn.Dense(self.output_dim)(x)


class RawObsValueMLP(nn.Module):
    hidden_dims: tuple[int, ...]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


class RawObsPolicyMLP(nn.Module):
    hidden_dims: tuple[int, ...]
    n_actions: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        return nn.Dense(self.n_actions)(x)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def array_digest(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def cached_stats(stats: ValueFitStats) -> ValueFitStats:
    return ValueFitStats(
        loss=stats.loss,
        mse=stats.mse,
        max_abs=stats.max_abs,
        action_agreement=stats.action_agreement,
        train_seconds=0.0,
    )


def value_supervision_cache_key(
    value_model: str,
    target_q: np.ndarray,
    n_actions: int,
    tie_tol: float,
) -> str:
    if value_model == "raw_obs_policy_mlp":
        labels = np.argmax(target_q, axis=1).astype(np.int32)
        return f"{value_model}:{array_digest(labels)}"
    if value_model == "raw_obs_tie_policy_mlp":
        max_q = target_q.max(axis=1, keepdims=True)
        labels = (target_q >= (max_q - tie_tol)).astype(np.bool_)
        return f"{value_model}:tol={tie_tol}:{array_digest(labels)}"
    if value_model == "raw_obs_prev_policy_mlp":
        labels = previous_action_policy_labels(target_q, n_actions)
        return f"{value_model}:{array_digest(labels)}"
    return f"{value_model}:{array_digest(target_q)}"


def transition_metrics(probs: np.ndarray, target: np.ndarray, fitted_mask: np.ndarray, target_source: str, train_seconds: float) -> TransitionFitStats:
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
        loss=ce,
        ce=ce,
        kl=kl,
        mean_l1=mean_l1,
        max_abs=max_abs,
        top1_agreement=top1,
        train_seconds=train_seconds,
        target_source=target_source,
        fitted_rows=row_count,
        fallback_rows=fallback_rows,
    )


def infer_action_from_xy(env, state: tuple[int, int], action_xy: np.ndarray) -> int:
    current_xy = np.asarray(env.unwrapped.ij_to_xy(state), dtype=np.float64)
    scores = []
    for di, dj in ACTION_DELTAS:
        target_xy = np.asarray(env.unwrapped.ij_to_xy((state[0] + di, state[1] + dj)), dtype=np.float64)
        direction = target_xy - current_xy
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
        scores.append(float(np.dot(np.asarray(action_xy, dtype=np.float64), direction)))
    return int(np.argmax(scores))


def build_dataset_count_targets(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
    prior_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros_like(topology.transitions, dtype=np.float64)
    obs = np.asarray(train["observations"], dtype=np.float32)
    actions = np.asarray(train["actions"], dtype=np.float32)
    valids = np.asarray(train["valids"], dtype=np.float32) > 0
    valid_idxs = np.nonzero(valids[:-1])[0]

    def obs_to_state(x: np.ndarray) -> tuple[int, int]:
        return tuple(env.unwrapped.xy_to_ij(np.asarray(x[:2], dtype=np.float64)))

    delta_to_action = {delta: idx for idx, delta in enumerate(ACTION_DELTAS)}
    for idx in valid_idxs:
        state = obs_to_state(obs[idx])
        next_state = obs_to_state(obs[idx + 1])
        if state not in topology.state_to_idx or next_state not in topology.state_to_idx:
            continue
        s = topology.state_to_idx[state]
        sp = topology.state_to_idx[next_state]
        delta = (next_state[0] - state[0], next_state[1] - state[1])
        if abs(delta[0]) + abs(delta[1]) > 1 and state in topology.teleport_in:
            counts[s, :, sp] += 1.0
        elif delta in delta_to_action:
            counts[s, delta_to_action[delta], sp] += 1.0
        else:
            action_id = infer_action_from_xy(env, state, actions[idx])
            counts[s, action_id, sp] += 1.0

    fitted_mask = counts.sum(axis=-1) > 0.0
    if prior_weight > 0.0:
        counts = counts + prior_weight * topology.transitions
        fitted_mask = counts.sum(axis=-1) > 0.0

    target = topology.transitions.copy()
    row_sums = counts.sum(axis=-1, keepdims=True)
    np.divide(counts, row_sums, out=target, where=row_sums > 0.0)
    return target, fitted_mask


def build_dataset_cell_change_targets(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
    prior_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros_like(topology.transitions, dtype=np.float64)
    obs = np.asarray(train["observations"], dtype=np.float32)
    actions = np.asarray(train["actions"], dtype=np.float32)
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
            action_id = infer_action_from_xy(env, state, actions[idx])
            counts[s, action_id, sp] += 1.0

    fitted_mask = counts.sum(axis=-1) > 0.0
    if prior_weight > 0.0:
        counts = counts + prior_weight * topology.transitions
        fitted_mask = counts.sum(axis=-1) > 0.0

    target = topology.transitions.copy()
    row_sums = counts.sum(axis=-1, keepdims=True)
    np.divide(counts, row_sums, out=target, where=row_sums > 0.0)
    return target, fitted_mask


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


def build_transition_targets(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
    source: str,
    prior_weight: float,
    samples_per_row: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if source == "topology":
        return topology.transitions.copy(), np.ones(topology.transitions.shape[:2], dtype=bool)
    if source == "topology_samples":
        return build_topology_sample_targets(topology, samples_per_row, seed)
    if source == "dataset_counts":
        return build_dataset_count_targets(env, train, topology, prior_weight)
    if source == "dataset_cell_changes":
        return build_dataset_cell_change_targets(env, train, topology, prior_weight)
    raise ValueError(f"unknown target source {source}")


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
            metrics = transition_metrics(probs, target, fitted_mask, "pending", time.perf_counter() - start)
            print(
                f"[transition] step={step} ce={metrics.ce:.6f} "
                f"kl={metrics.kl:.6f} l1={metrics.mean_l1:.6f} top1={metrics.top1_agreement:.3f}",
                flush=True,
            )
    probs = stable_softmax(logits)
    probs = np.where(fitted_mask[..., None], probs, target)
    stats = transition_metrics(probs, target, fitted_mask, "pending", time.perf_counter() - start)
    return probs, stats


def build_raw_obs_row_features(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
) -> np.ndarray:
    obs = np.asarray(train["observations"], dtype=np.float32)
    obs_mean = obs.mean(axis=0)
    obs_std = np.maximum(obs.std(axis=0), 1e-3)
    centers = np.asarray(
        [env.unwrapped.ij_to_xy(state) for state in topology.states],
        dtype=np.float32,
    )
    norm_centers = (centers - obs_mean[None, :]) / obs_std[None, :]
    n_states = topology.n_states
    n_actions = topology.n_actions
    features = np.zeros((n_states, n_actions, obs.shape[1] + n_actions), dtype=np.float32)
    eye = np.eye(n_actions, dtype=np.float32)
    for action_id in range(n_actions):
        features[:, action_id, : obs.shape[1]] = norm_centers
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
        stats = transition_metrics(target, target, fitted_mask, "pending", 0.0)
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
    probs = transition_mlp_predict(state, features, topology.n_states, topology.n_actions)
    for step in range(1, steps + 1):
        state, loss = transition_mlp_train_step(state, feats_jax, targets_jax)
        if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == steps):
            probs = transition_mlp_predict(state, features, topology.n_states, topology.n_actions)
            probs_for_eval = np.where(fitted_mask[..., None], probs, target)
            metrics = transition_metrics(
                probs_for_eval,
                target,
                fitted_mask,
                "pending",
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
    stats = transition_metrics(probs, target, fitted_mask, "pending", time.perf_counter() - start)
    return probs, stats


def build_raw_obs_value_features(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
) -> np.ndarray:
    obs = np.asarray(train["observations"], dtype=np.float32)
    obs_mean = obs.mean(axis=0)
    obs_std = np.maximum(obs.std(axis=0), 1e-3)
    centers = np.asarray(
        [env.unwrapped.ij_to_xy(state) for state in topology.states],
        dtype=np.float32,
    )
    state_obs = np.tile(obs_mean[None, :], (topology.n_states, 1)).astype(np.float32)
    goal_obs = np.tile(obs_mean[None, :], (topology.n_states, 1)).astype(np.float32)
    state_obs[:, :2] = centers
    goal_obs[:, :2] = centers
    norm_state = (state_obs - obs_mean[None, :]) / obs_std[None, :]
    norm_goal = (goal_obs - obs_mean[None, :]) / obs_std[None, :]

    n_states = topology.n_states
    n_actions = topology.n_actions
    features = np.zeros((n_states, n_actions, n_states, obs.shape[1] * 3 + n_actions), dtype=np.float32)
    eye = np.eye(n_actions, dtype=np.float32)
    for action_id in range(n_actions):
        features[:, action_id, :, : obs.shape[1]] = norm_state[:, None, :]
        features[:, action_id, :, obs.shape[1] : 2 * obs.shape[1]] = norm_goal[None, :, :]
        features[:, action_id, :, 2 * obs.shape[1] : 3 * obs.shape[1]] = (
            norm_goal[None, :, :] - norm_state[:, None, :]
        )
        features[:, action_id, :, 3 * obs.shape[1] :] = eye[action_id]
    return features


def create_value_train_state(
    seed: int,
    feature_dim: int,
    hidden_dims: tuple[int, ...],
    lr: float,
) -> train_state.TrainState:
    model = RawObsValueMLP(hidden_dims=hidden_dims)
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, feature_dim), dtype=jnp.float32))["params"]
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(lr))


@jax.jit
def value_mlp_train_step(
    state: train_state.TrainState,
    feats: jax.Array,
    targets: jax.Array,
    action_labels: jax.Array,
    action_mask: jax.Array,
    policy_ce_weight: float,
):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, feats)
        preds = jax.nn.sigmoid(logits)
        value_loss = jnp.mean((preds - targets.reshape(-1)) ** 2)
        action_logits = logits.reshape(targets.shape).transpose(0, 2, 1)
        ce = optax.softmax_cross_entropy_with_integer_labels(action_logits, action_labels)
        policy_loss = jnp.sum(ce * action_mask) / jnp.maximum(jnp.sum(action_mask), 1.0)
        return value_loss + policy_ce_weight * policy_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


def value_mlp_predict(
    state: train_state.TrainState,
    features: np.ndarray,
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    flat = features.reshape(n_states * n_actions * n_states, features.shape[-1])
    logits = state.apply_fn({"params": state.params}, jnp.asarray(flat, dtype=jnp.float32))
    preds = np.asarray(jax.nn.sigmoid(logits), dtype=np.float64)
    return preds.reshape(n_states, n_actions, n_states)


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


def fit_raw_obs_value_mlp(
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
    policy_ce_weight: float,
) -> tuple[np.ndarray, ValueFitStats]:
    features = build_raw_obs_value_features(env, train, topology)
    flat_features = features.reshape(-1, features.shape[-1])
    flat_targets = target_q.reshape(-1).astype(np.float32)
    action_labels = np.argmax(target_q, axis=1).astype(np.int32)
    action_mask = (~np.eye(topology.n_states, dtype=bool)).astype(np.float32)
    state = create_value_train_state(seed, int(flat_features.shape[-1]), hidden_dims, lr)
    feats_jax = jnp.asarray(flat_features, dtype=jnp.float32)
    targets_jax = jnp.asarray(target_q.astype(np.float32), dtype=jnp.float32)
    action_labels_jax = jnp.asarray(action_labels, dtype=jnp.int32)
    action_mask_jax = jnp.asarray(action_mask, dtype=jnp.float32)
    start = time.perf_counter()
    pred = value_mlp_predict(state, features, topology.n_states, topology.n_actions)
    for step in range(1, steps + 1):
        state, loss = value_mlp_train_step(
            state,
            feats_jax,
            targets_jax,
            action_labels_jax,
            action_mask_jax,
            policy_ce_weight,
        )
        if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == steps):
            pred = value_mlp_predict(state, features, topology.n_states, topology.n_actions)
            metrics = value_metrics(pred, target_q)
            print(
                f"[value-mlp] method={method} step={step} mse={metrics.mse:.6f} "
                f"max_abs={metrics.max_abs:.6f} action_agree={metrics.action_agreement:.3f} "
                f"loss={float(loss):.6f}",
                flush=True,
            )
    pred = value_mlp_predict(state, features, topology.n_states, topology.n_actions)
    stats = value_metrics(pred, target_q)
    stats.train_seconds = time.perf_counter() - start
    stats.loss = float(stats.mse)
    return np.clip(pred, 0.0, 1.0), stats


def build_raw_obs_policy_features(
    env,
    train: dict[str, np.ndarray],
    topology: Topology,
) -> np.ndarray:
    obs = np.asarray(train["observations"], dtype=np.float32)
    obs_mean = obs.mean(axis=0)
    obs_std = np.maximum(obs.std(axis=0), 1e-3)
    centers = np.asarray(
        [env.unwrapped.ij_to_xy(state) for state in topology.states],
        dtype=np.float32,
    )
    state_obs = np.tile(obs_mean[None, :], (topology.n_states, 1)).astype(np.float32)
    goal_obs = np.tile(obs_mean[None, :], (topology.n_states, 1)).astype(np.float32)
    state_obs[:, :2] = centers
    goal_obs[:, :2] = centers
    norm_state = (state_obs - obs_mean[None, :]) / obs_std[None, :]
    norm_goal = (goal_obs - obs_mean[None, :]) / obs_std[None, :]
    features = np.zeros((topology.n_states, topology.n_states, obs.shape[1] * 3), dtype=np.float32)
    features[:, :, : obs.shape[1]] = norm_state[:, None, :]
    features[:, :, obs.shape[1] : 2 * obs.shape[1]] = norm_goal[None, :, :]
    features[:, :, 2 * obs.shape[1] :] = norm_goal[None, :, :] - norm_state[:, None, :]
    return features


def create_policy_train_state(
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
def policy_mlp_train_step(
    state: train_state.TrainState,
    feats: jax.Array,
    labels: jax.Array,
    mask: jax.Array,
):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, feats)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return jnp.sum(ce * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


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


def policy_mlp_predict(
    state: train_state.TrainState,
    features: np.ndarray,
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    flat = features.reshape(n_states * n_states, features.shape[-1])
    logits = state.apply_fn({"params": state.params}, jnp.asarray(flat, dtype=jnp.float32))
    probs = np.asarray(jax.nn.softmax(logits, axis=-1), dtype=np.float64).reshape(n_states, n_states, n_actions)
    return probs.transpose(0, 2, 1)


def fit_raw_obs_policy_mlp(
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
    features = build_raw_obs_policy_features(env, train, topology)
    action_labels = np.argmax(target_q, axis=1).astype(np.int32)
    action_mask = (~np.eye(topology.n_states, dtype=bool)).astype(np.float32)
    state = create_policy_train_state(seed, int(features.shape[-1]), topology.n_actions, hidden_dims, lr)
    feats_jax = jnp.asarray(features.reshape(-1, features.shape[-1]), dtype=jnp.float32)
    labels_jax = jnp.asarray(action_labels.reshape(-1), dtype=jnp.int32)
    mask_jax = jnp.asarray(action_mask.reshape(-1), dtype=jnp.float32)
    start = time.perf_counter()
    pred = policy_mlp_predict(state, features, topology.n_states, topology.n_actions)
    for step in range(1, steps + 1):
        state, loss = policy_mlp_train_step(state, feats_jax, labels_jax, mask_jax)
        if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == steps):
            pred = policy_mlp_predict(state, features, topology.n_states, topology.n_actions)
            metrics = value_metrics(pred, target_q)
            print(
                f"[policy-mlp] method={method} step={step} mse={metrics.mse:.6f} "
                f"max_abs={metrics.max_abs:.6f} action_agree={metrics.action_agreement:.3f} "
                f"loss={float(loss):.6f}",
                flush=True,
            )
    pred = policy_mlp_predict(state, features, topology.n_states, topology.n_actions)
    stats = value_metrics(pred, target_q)
    stats.train_seconds = time.perf_counter() - start
    stats.loss = float(stats.mse)
    return pred, stats


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
    state = create_policy_train_state(seed, int(features.shape[-1]), topology.n_actions, hidden_dims, lr)
    feats_jax = jnp.asarray(features.reshape(-1, features.shape[-1]), dtype=jnp.float32)
    targets_jax = jnp.asarray(tie_targets.reshape(-1, topology.n_actions), dtype=jnp.float32)
    mask_jax = jnp.asarray(action_mask.reshape(-1), dtype=jnp.float32)
    start = time.perf_counter()
    pred_probs = policy_mlp_predict(state, features, topology.n_states, topology.n_actions)
    for step in range(1, steps + 1):
        state, loss = tie_policy_mlp_train_step(state, feats_jax, targets_jax, mask_jax)
        if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == steps):
            pred_probs = policy_mlp_predict(state, features, topology.n_states, topology.n_actions)
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
    pred_probs = policy_mlp_predict(state, features, topology.n_states, topology.n_actions)
    pred_mask = (pred_probs >= 0.5).astype(np.float64)
    empty = pred_mask.sum(axis=1, keepdims=True) == 0.0
    fallback = np.zeros_like(pred_mask)
    np.put_along_axis(fallback, np.argmax(pred_probs, axis=1, keepdims=True), 1.0, axis=1)
    pred_q = np.where(empty, fallback, pred_mask)
    for g in range(topology.n_states):
        pred_q[g, :, g] = 1.0
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
    state = create_policy_train_state(
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
        "bellman_full, bellman_<n>_sweeps, or sto_trl_<n>_sweeps"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="pointmaze-teleport-stitch-v0")
    parser.add_argument("--topology-source", choices=["env", "dataset"], default="dataset")
    parser.add_argument(
        "--transition-model",
        choices=["table_softmax", "raw_obs_mlp"],
        default="table_softmax",
    )
    parser.add_argument(
        "--transition-target-source",
        choices=["topology", "topology_samples", "dataset_counts", "dataset_cell_changes"],
        default="topology",
    )
    parser.add_argument("--samples-per-row", type=int, default=50)
    parser.add_argument("--dataset-prior-weight", type=float, default=0.0)
    parser.add_argument("--transition-steps", type=int, default=2_000)
    parser.add_argument("--transition-lr", type=float, default=0.3)
    parser.add_argument("--transition-mlp-lr", type=float, default=3e-3)
    parser.add_argument("--transition-hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--transition-seed", type=int, default=0)
    parser.add_argument("--transition-log-interval", type=int, default=500)
    parser.add_argument(
        "--value-model",
        choices=[
            "table",
            "raw_obs_mlp",
            "raw_obs_policy_mlp",
            "raw_obs_tie_policy_mlp",
            "raw_obs_prev_policy_mlp",
        ],
        default="table",
    )
    parser.add_argument("--value-steps", type=int, default=3_000)
    parser.add_argument("--value-mlp-lr", type=float, default=3e-3)
    parser.add_argument("--value-hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--value-seed", type=int, default=0)
    parser.add_argument("--value-log-interval", type=int, default=1_000)
    parser.add_argument("--value-policy-ce-weight", type=float, default=0.0)
    parser.add_argument("--value-tie-tol", type=float, default=1e-9)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--full-iters", type=int, default=180)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
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
        default=ROOT / "results" / "pointmaze_learned_transition_planner.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=ROOT / "results" / "pointmaze_learned_transition_planner_summary.json",
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
    if args.transition_model == "table_softmax":
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
    fit_stats.target_source = args.transition_target_source
    row_sums = learned_transitions.sum(axis=-1, keepdims=True)
    learned_transitions = np.divide(
        learned_transitions,
        row_sums,
        out=np.zeros_like(learned_transitions),
        where=row_sums > 0.0,
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
        f"[transition] final ce={fit_stats.ce:.6f} kl={fit_stats.kl:.6f} "
        f"l1={fit_stats.mean_l1:.6f} max_abs={fit_stats.max_abs:.6f} "
        f"top1={fit_stats.top1_agreement:.3f} fitted_rows={fit_stats.fitted_rows} "
        f"fallback_rows={fit_stats.fallback_rows} seconds={fit_stats.train_seconds:.2f}",
        flush=True,
    )
    oracle_stats = transition_metrics(
        learned_transitions,
        topology.transitions,
        np.ones(topology.transitions.shape[:2], dtype=bool),
        "topology_oracle",
        fit_stats.train_seconds,
    )
    print(
        f"[transition-oracle] kl={oracle_stats.kl:.6f} l1={oracle_stats.mean_l1:.6f} "
        f"max_abs={oracle_stats.max_abs:.6f} top1={oracle_stats.top1_agreement:.3f}",
        flush=True,
    )

    q_by_method: dict[str, np.ndarray] = {}
    iters_by_method: dict[str, int] = {}
    value_fit_by_method: dict[str, ValueFitStats] = {}
    value_cache_hit_by_method: dict[str, int] = {}
    value_supervision_cache: dict[str, tuple[np.ndarray, ValueFitStats]] = {}
    for method in args.methods:
        value_iters, use_transitive = resolve_value_method(method, matched_iters, args.full_iters)
        table_q = train_model_value(
            learned_topology.transitions,
            args.gamma,
            value_iters,
            use_transitive=use_transitive,
        )
        if args.value_model == "table":
            q_by_method[method] = table_q
            value_fit_by_method[method] = ValueFitStats(
                loss=0.0,
                mse=0.0,
                max_abs=0.0,
                action_agreement=1.0,
                train_seconds=0.0,
            )
            value_cache_hit_by_method[method] = 0
        else:
            cache_key = value_supervision_cache_key(
                args.value_model,
                table_q,
                learned_topology.n_actions,
                args.value_tie_tol,
            )
            if cache_key in value_supervision_cache:
                fitted_q, source_stats = value_supervision_cache[cache_key]
                value_stats = cached_stats(source_stats)
                value_cache_hit_by_method[method] = 1
                print(
                    f"[value-cache] supervision hit method={method} "
                    f"model={args.value_model} key={cache_key[:32]}",
                    flush=True,
                )
            elif args.value_model == "raw_obs_mlp":
                fitted_q, value_stats = fit_raw_obs_value_mlp(
                    env,
                    train,
                    learned_topology,
                    table_q,
                    args.value_steps,
                    args.value_mlp_lr,
                    args.value_seed,
                    tuple(args.value_hidden_dims),
                    args.value_log_interval,
                    method,
                    args.value_policy_ce_weight,
                )
                value_supervision_cache[cache_key] = (fitted_q, value_stats)
                value_cache_hit_by_method[method] = 0
            elif args.value_model == "raw_obs_policy_mlp":
                fitted_q, value_stats = fit_raw_obs_policy_mlp(
                    env,
                    train,
                    learned_topology,
                    table_q,
                    args.value_steps,
                    args.value_mlp_lr,
                    args.value_seed,
                    tuple(args.value_hidden_dims),
                    args.value_log_interval,
                    method,
                )
                value_supervision_cache[cache_key] = (fitted_q, value_stats)
                value_cache_hit_by_method[method] = 0
            elif args.value_model == "raw_obs_tie_policy_mlp":
                fitted_q, value_stats = fit_raw_obs_tie_policy_mlp(
                    env,
                    train,
                    learned_topology,
                    table_q,
                    args.value_steps,
                    args.value_mlp_lr,
                    args.value_seed,
                    tuple(args.value_hidden_dims),
                    args.value_log_interval,
                    method,
                    args.value_tie_tol,
                )
                value_supervision_cache[cache_key] = (fitted_q, value_stats)
                value_cache_hit_by_method[method] = 0
            elif args.value_model == "raw_obs_prev_policy_mlp":
                fitted_q, value_stats = fit_raw_obs_prev_policy_mlp(
                    env,
                    train,
                    learned_topology,
                    table_q,
                    args.value_steps,
                    args.value_mlp_lr,
                    args.value_seed,
                    tuple(args.value_hidden_dims),
                    args.value_log_interval,
                    method,
                )
                value_supervision_cache[cache_key] = (fitted_q, value_stats)
                value_cache_hit_by_method[method] = 0
            else:
                raise ValueError(f"unknown value model {args.value_model}")
            q_by_method[method] = fitted_q
            value_fit_by_method[method] = value_stats
        iters_by_method[method] = value_iters

    rows: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        for method in args.methods:
            print(f"[run] seed={seed} method={method}", flush=True)
            if args.eval_mode == "model":
                metrics = evaluate_model(
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
                metrics = evaluate(
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
            row = {
                "method": method,
                "env": args.env_name,
                "seed": seed,
                "episodes_per_task": args.episodes,
                "gamma": args.gamma,
                "iters": iters_by_method[method],
                "full_iters": args.full_iters,
                "n_states": topology.n_states,
                "n_actions": topology.n_actions,
                "action_scale": args.action_scale,
                "eval_mode": args.eval_mode,
                "model_rollout_mode": args.model_rollout_mode if args.eval_mode == "model" else "",
                "model_max_steps": "" if args.model_max_steps is None else args.model_max_steps,
                "max_episode_steps": "" if args.max_episode_steps is None else args.max_episode_steps,
                "topology_source": args.topology_source,
                "task_ids": "all" if args.task_ids is None else ",".join(map(str, args.task_ids)),
                "transition_model": args.transition_model,
                "transition_target_source": args.transition_target_source,
                "transition_steps": args.transition_steps,
                "transition_lr": args.transition_lr,
                "transition_mlp_lr": args.transition_mlp_lr,
                "transition_hidden_dims": ",".join(map(str, args.transition_hidden_dims)),
                "transition_seed": args.transition_seed,
                "transition_samples_per_row": args.samples_per_row,
                "transition_ce": fit_stats.ce,
                "transition_kl": fit_stats.kl,
                "transition_l1": fit_stats.mean_l1,
                "transition_max_abs": fit_stats.max_abs,
                "transition_top1": fit_stats.top1_agreement,
                "transition_oracle_kl": oracle_stats.kl,
                "transition_oracle_l1": oracle_stats.mean_l1,
                "transition_oracle_max_abs": oracle_stats.max_abs,
                "transition_oracle_top1": oracle_stats.top1_agreement,
                "transition_train_seconds": fit_stats.train_seconds,
                "transition_fitted_rows": fit_stats.fitted_rows,
                "transition_fallback_rows": fit_stats.fallback_rows,
                "value_model": args.value_model,
                "value_steps": args.value_steps,
                "value_mlp_lr": args.value_mlp_lr,
                "value_hidden_dims": ",".join(map(str, args.value_hidden_dims)),
                "value_seed": args.value_seed,
                "value_policy_ce_weight": args.value_policy_ce_weight,
                "value_tie_tol": args.value_tie_tol,
                "value_cache_hit": value_cache_hit_by_method[method],
                "value_loss": value_fit_by_method[method].loss,
                "value_mse": value_fit_by_method[method].mse,
                "value_max_abs": value_fit_by_method[method].max_abs,
                "value_action_agreement": value_fit_by_method[method].action_agreement,
                "value_train_seconds": value_fit_by_method[method].train_seconds,
                **metrics,
            }
            rows.append(row)
            write_csv(rows, args.out)
            args.summary_out.parent.mkdir(parents=True, exist_ok=True)
            args.summary_out.write_text(json.dumps(summarize(rows), indent=2, sort_keys=True))

    print(json.dumps(summarize(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
