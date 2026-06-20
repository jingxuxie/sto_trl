from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import partial

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from sto_trl.learners import bellman_backup, stochastic_transitive_backup, train_support_transitive_value


@dataclass(frozen=True)
class NeuralToyConfig:
    hidden_dims: tuple[int, ...] = (128, 128)
    lr: float = 3e-3
    warmup_steps: int = 250
    steps_per_iter: int = 120
    positive_weight: float = 4.0
    diag_weight: float = 4.0
    rank_ce_weight: float = 0.05
    log_interval: int = 0


@dataclass(frozen=True)
class NeuralToyStats:
    final_loss: float
    train_seconds: float
    operator_iters: int
    warmup_steps: int
    steps_per_iter: int


class ReachabilityMLP(nn.Module):
    hidden_dims: tuple[int, ...]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


class ActionHeadReachabilityMLP(nn.Module):
    hidden_dims: tuple[int, ...]
    n_actions: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        return nn.Dense(self.n_actions)(x)


def build_onehot_features(n_states: int, n_actions: int) -> np.ndarray:
    state_eye = np.eye(n_states, dtype=np.float32)
    action_eye = np.eye(n_actions, dtype=np.float32)
    goal_eye = np.eye(n_states, dtype=np.float32)
    feats = np.zeros(
        (n_states, n_actions, n_states, 2 * n_states + n_actions),
        dtype=np.float32,
    )
    for state in range(n_states):
        feats[state, :, :, :n_states] = state_eye[state]
    for action in range(n_actions):
        feats[:, action, :, n_states : n_states + n_actions] = action_eye[action]
    for goal in range(n_states):
        feats[:, :, goal, n_states + n_actions :] = goal_eye[goal]
    return feats.reshape(n_states * n_actions * n_states, -1)


def build_state_goal_onehot_features(n_states: int) -> np.ndarray:
    state_eye = np.eye(n_states, dtype=np.float32)
    goal_eye = np.eye(n_states, dtype=np.float32)
    feats = np.zeros((n_states, n_states, 2 * n_states), dtype=np.float32)
    feats[:, :, :n_states] = state_eye[:, None, :]
    feats[:, :, n_states:] = goal_eye[None, :, :]
    return feats.reshape(n_states * n_states, -1)


def one_step_q(transitions: np.ndarray, gamma: float) -> np.ndarray:
    n_states, _n_actions, _ = transitions.shape
    q = gamma * transitions.copy()
    for goal in range(n_states):
        q[goal, :, goal] = 1.0
    return np.clip(q, 0.0, 1.0)


def support_one_step_q(transitions: np.ndarray, gamma: float) -> np.ndarray:
    n_states, _n_actions, _ = transitions.shape
    q = np.zeros_like(transitions, dtype=np.float64)
    q[transitions > 0.0] = gamma
    for goal in range(n_states):
        q[goal, :, goal] = 1.0
    return np.clip(q, 0.0, 1.0)


def target_weights(target: np.ndarray, config: NeuralToyConfig) -> np.ndarray:
    n_states, n_actions, _ = target.shape
    weights = np.ones_like(target, dtype=np.float32)
    weights += config.positive_weight * (target > 1e-8).astype(np.float32)
    for goal in range(n_states):
        weights[goal, :, goal] += config.diag_weight
    return weights.reshape(n_states * n_actions * n_states)


def action_labels_and_mask(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_states = target.shape[0]
    labels = np.argmax(target, axis=1).astype(np.int32)
    spread = np.max(target, axis=1) - np.min(target, axis=1)
    mask = ((~np.eye(n_states, dtype=bool)) & (spread > 1e-6)).astype(np.float32)
    return labels, mask


def create_train_state(
    seed: int,
    feature_dim: int,
    hidden_dims: tuple[int, ...],
    lr: float,
) -> train_state.TrainState:
    model = ReachabilityMLP(hidden_dims=hidden_dims)
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, feature_dim), dtype=jnp.float32))["params"]
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(lr))


def create_action_head_train_state(
    seed: int,
    feature_dim: int,
    n_actions: int,
    hidden_dims: tuple[int, ...],
    lr: float,
) -> train_state.TrainState:
    model = ActionHeadReachabilityMLP(hidden_dims=hidden_dims, n_actions=n_actions)
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, feature_dim), dtype=jnp.float32))["params"]
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(lr))


@partial(jax.jit, static_argnames=("n_actions",))
def train_step(
    state: train_state.TrainState,
    features: jax.Array,
    targets: jax.Array,
    weights: jax.Array,
    labels: jax.Array,
    label_mask: jax.Array,
    rank_ce_weight: float,
    n_actions: int,
) -> tuple[train_state.TrainState, jax.Array]:
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, features)
        preds = jax.nn.sigmoid(logits)
        sqerr = (preds - targets) ** 2
        value_loss = jnp.sum(weights * sqerr) / jnp.maximum(jnp.sum(weights), 1.0)
        action_logits = logits.reshape((labels.shape[0], n_actions, labels.shape[1])).transpose(0, 2, 1)
        ce = optax.softmax_cross_entropy_with_integer_labels(action_logits, labels)
        rank_loss = jnp.sum(ce * label_mask) / jnp.maximum(jnp.sum(label_mask), 1.0)
        return value_loss + rank_ce_weight * rank_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@jax.jit
def action_head_train_step(
    state: train_state.TrainState,
    features: jax.Array,
    targets: jax.Array,
    row_weights: jax.Array,
    labels: jax.Array,
    label_mask: jax.Array,
    rank_ce_weight: float,
) -> tuple[train_state.TrainState, jax.Array]:
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, features)
        preds = jax.nn.sigmoid(logits)
        row_loss = jnp.mean((preds - targets) ** 2, axis=-1)
        value_loss = jnp.sum(row_weights * row_loss) / jnp.maximum(jnp.sum(row_weights), 1.0)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        rank_loss = jnp.sum(ce * label_mask) / jnp.maximum(jnp.sum(label_mask), 1.0)
        return value_loss + rank_ce_weight * rank_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


def predict_q(
    state: train_state.TrainState,
    features: np.ndarray,
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    logits = state.apply_fn({"params": state.params}, jnp.asarray(features, dtype=jnp.float32))
    values = np.asarray(jax.nn.sigmoid(logits), dtype=np.float64)
    q = values.reshape(n_states, n_actions, n_states)
    for goal in range(n_states):
        q[goal, :, goal] = 1.0
    return np.clip(q, 0.0, 1.0)


def predict_action_head_q(
    state: train_state.TrainState,
    features: np.ndarray,
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    logits = state.apply_fn({"params": state.params}, jnp.asarray(features, dtype=jnp.float32))
    values = np.asarray(jax.nn.sigmoid(logits), dtype=np.float64)
    q = values.reshape(n_states, n_states, n_actions).transpose(0, 2, 1)
    for goal in range(n_states):
        q[goal, :, goal] = 1.0
    return np.clip(q, 0.0, 1.0)


def fit_target(
    state: train_state.TrainState,
    features_jax: jax.Array,
    target: np.ndarray,
    steps: int,
    config: NeuralToyConfig,
) -> tuple[train_state.TrainState, float]:
    _n_states, n_actions, _ = target.shape
    targets_jax = jnp.asarray(target.reshape(-1), dtype=jnp.float32)
    weights_jax = jnp.asarray(target_weights(target, config), dtype=jnp.float32)
    labels, label_mask = action_labels_and_mask(target)
    labels_jax = jnp.asarray(labels, dtype=jnp.int32)
    label_mask_jax = jnp.asarray(label_mask, dtype=jnp.float32)
    loss = jnp.asarray(0.0)
    for _ in range(max(0, int(steps))):
        state, loss = train_step(
            state,
            features_jax,
            targets_jax,
            weights_jax,
            labels_jax,
            label_mask_jax,
            float(config.rank_ce_weight),
            n_actions,
        )
    return state, float(loss)


def action_head_row_weights(target: np.ndarray, config: NeuralToyConfig) -> np.ndarray:
    n_states, _n_actions, _ = target.shape
    rows = target.transpose(0, 2, 1)
    weights = np.ones((n_states, n_states), dtype=np.float32)
    weights += config.positive_weight * np.max(rows > 1e-8, axis=-1).astype(np.float32)
    diag = np.arange(n_states)
    weights[diag, diag] += config.diag_weight
    return weights.reshape(n_states * n_states)


def fit_action_head_target(
    state: train_state.TrainState,
    features_jax: jax.Array,
    target: np.ndarray,
    steps: int,
    config: NeuralToyConfig,
    label_target: np.ndarray | None = None,
) -> tuple[train_state.TrainState, float]:
    n_states, _n_actions, _ = target.shape
    if label_target is None:
        label_target = target
    targets = target.transpose(0, 2, 1).reshape(n_states * n_states, -1)
    targets_jax = jnp.asarray(targets, dtype=jnp.float32)
    row_weights_jax = jnp.asarray(action_head_row_weights(target, config), dtype=jnp.float32)
    labels = np.argmax(label_target, axis=1).astype(np.int32)
    spread = np.max(label_target, axis=1) - np.min(label_target, axis=1)
    label_mask = ((~np.eye(n_states, dtype=bool)) & (spread > 1e-6)).astype(np.float32)
    labels_jax = jnp.asarray(labels.reshape(-1), dtype=jnp.int32)
    label_mask_jax = jnp.asarray(label_mask.reshape(-1), dtype=jnp.float32)
    loss = jnp.asarray(0.0)
    for _ in range(max(0, int(steps))):
        state, loss = action_head_train_step(
            state,
            features_jax,
            targets_jax,
            row_weights_jax,
            labels_jax,
            label_mask_jax,
            float(config.rank_ce_weight),
        )
    return state, float(loss)


def stochastic_transitive_backup_topk_bridge(q: np.ndarray, gamma: float, waypoint_k: int) -> np.ndarray:
    n_states, _n_actions, _ = q.shape
    if waypoint_k <= 0 or waypoint_k >= n_states:
        return stochastic_transitive_backup(q, gamma)

    v = q.max(axis=1)
    target = np.zeros_like(q)
    # Retrieve a small waypoint set per (state, goal) by value composition,
    # then compute the action-value transitive target only on that set.
    bridge_scores = v[:, :, None] * v[None, :, :]
    for state in range(n_states):
        bridge_scores[state, state, :] = -np.inf
    for state in range(n_states):
        for goal in range(n_states):
            candidates = np.argpartition(bridge_scores[state, :, goal], -waypoint_k)[-waypoint_k:]
            vals = np.take(q[state], candidates, axis=1) * v[candidates, goal][None, :]
            target[state, :, goal] = np.max(vals, axis=1)
    for goal in range(n_states):
        target[goal, :, goal] = 1.0
    return np.clip(target, 0.0, 1.0)


def sample_next_states_from_transitions(
    transitions: np.ndarray,
    samples_per_row: int,
    seed: int,
) -> np.ndarray:
    n_states, n_actions, _ = transitions.shape
    if samples_per_row <= 0:
        raise ValueError("samples_per_row must be positive")
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_states, n_actions, samples_per_row), dtype=np.int32)
    for state in range(n_states):
        for action in range(n_actions):
            probs = np.asarray(transitions[state, action], dtype=np.float64)
            total = float(np.sum(probs))
            if total <= 0.0:
                samples[state, action, :] = state
                continue
            probs = probs / total
            samples[state, action, :] = rng.choice(n_states, size=samples_per_row, replace=True, p=probs)
    return samples


def sample_waypoint_candidates(
    n_states: int,
    candidates_per_pair: int,
    seed: int,
) -> np.ndarray:
    if candidates_per_pair <= 0:
        raise ValueError("candidates_per_pair must be positive")
    rng = np.random.default_rng(seed)
    candidates = np.zeros((n_states, n_states, candidates_per_pair), dtype=np.int32)
    all_states = np.arange(n_states, dtype=np.int32)
    for state in range(n_states):
        valid = all_states[all_states != state]
        replace = candidates_per_pair > len(valid)
        for goal in range(n_states):
            candidates[state, goal] = rng.choice(valid, size=candidates_per_pair, replace=replace)
    return candidates


def one_step_q_from_samples(next_state_samples: np.ndarray, gamma: float) -> np.ndarray:
    n_states, n_actions, _samples_per_row = next_state_samples.shape
    q = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    for state in range(n_states):
        for action in range(n_actions):
            counts = np.bincount(next_state_samples[state, action], minlength=n_states).astype(np.float64)
            q[state, action] = gamma * counts / max(float(np.sum(counts)), 1.0)
    for goal in range(n_states):
        q[goal, :, goal] = 1.0
    return np.clip(q, 0.0, 1.0)


def support_one_step_q_from_samples(next_state_samples: np.ndarray, gamma: float) -> np.ndarray:
    n_states, n_actions, _samples_per_row = next_state_samples.shape
    q = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    for state in range(n_states):
        for action in range(n_actions):
            q[state, action, np.unique(next_state_samples[state, action])] = gamma
    for goal in range(n_states):
        q[goal, :, goal] = 1.0
    return np.clip(q, 0.0, 1.0)


def sampled_bellman_backup(q: np.ndarray, next_state_samples: np.ndarray, gamma: float) -> np.ndarray:
    n_states, _n_actions, _samples_per_row = next_state_samples.shape
    v = q.max(axis=1)
    diag = np.arange(n_states)
    v[diag, diag] = 1.0
    target = gamma * np.mean(v[next_state_samples], axis=2)
    target[diag, :, diag] = 1.0
    return np.clip(target, 0.0, 1.0)


def sampled_transitive_backup_bridge(
    q: np.ndarray,
    waypoint_candidates: np.ndarray,
    waypoint_k: int,
) -> np.ndarray:
    n_states, _n_actions, _ = q.shape
    v = q.max(axis=1)
    target = np.zeros_like(q)
    for state in range(n_states):
        for goal in range(n_states):
            candidates = waypoint_candidates[state, goal]
            if waypoint_k > 0 and waypoint_k < len(candidates):
                scores = v[state, candidates] * v[candidates, goal]
                top = np.argpartition(scores, -waypoint_k)[-waypoint_k:]
                candidates = candidates[top]
            vals = np.take(q[state], candidates, axis=1) * v[candidates, goal][None, :]
            target[state, :, goal] = np.max(vals, axis=1)
    for goal in range(n_states):
        target[goal, :, goal] = 1.0
    return np.clip(target, 0.0, 1.0)


def sampled_operator_target(
    q: np.ndarray,
    next_state_samples: np.ndarray,
    gamma: float,
    method: str,
    support_q: np.ndarray,
    waypoint_candidates: np.ndarray,
    waypoint_k: int,
) -> np.ndarray:
    transitive = sampled_transitive_backup_bridge(q, waypoint_candidates, waypoint_k)
    if method == "neural_bellman_td":
        return sampled_bellman_backup(q, next_state_samples, gamma)
    if method == "neural_sto_trl":
        bellman = sampled_bellman_backup(q, next_state_samples, gamma)
        return np.maximum(bellman, transitive)
    if method == "neural_sto_trl_monotone":
        bellman = sampled_bellman_backup(q, next_state_samples, gamma)
        return np.maximum.reduce([q, bellman, transitive])
    if method == "neural_support_trl":
        return np.maximum.reduce([q, support_q, transitive])
    raise ValueError(f"unknown neural method {method}")


def operator_target(
    q: np.ndarray,
    transitions: np.ndarray,
    gamma: float,
    method: str,
    support_q: np.ndarray,
    waypoint_mode: str = "all",
    waypoint_k: int = 0,
) -> np.ndarray:
    if waypoint_mode == "all":
        transitive = stochastic_transitive_backup(q, gamma)
    elif waypoint_mode == "topk_bridge":
        transitive = stochastic_transitive_backup_topk_bridge(q, gamma, waypoint_k)
    else:
        raise ValueError(f"unknown waypoint mode {waypoint_mode}")
    if method == "neural_bellman_td":
        return bellman_backup(q, transitions, gamma)
    if method == "neural_sto_trl":
        bellman = bellman_backup(q, transitions, gamma)
        return np.maximum(bellman, transitive)
    if method == "neural_sto_trl_monotone":
        bellman = bellman_backup(q, transitions, gamma)
        return np.maximum.reduce([q, bellman, transitive])
    if method == "neural_support_trl":
        return np.maximum.reduce([q, support_q, transitive])
    raise ValueError(f"unknown neural method {method}")


def train_neural_toy_critic(
    transitions: np.ndarray,
    gamma: float,
    method: str,
    operator_iters: int,
    seed: int,
    config: NeuralToyConfig,
) -> tuple[np.ndarray, NeuralToyStats]:
    n_states, n_actions, _ = transitions.shape
    features = build_onehot_features(n_states, n_actions)
    return train_neural_critic_from_features(
        transitions,
        gamma,
        method,
        operator_iters,
        seed,
        config,
        features,
    )


def train_action_head_neural_critic_from_features(
    transitions: np.ndarray,
    gamma: float,
    method: str,
    operator_iters: int,
    seed: int,
    config: NeuralToyConfig,
    features: np.ndarray,
) -> tuple[np.ndarray, NeuralToyStats]:
    n_states, n_actions, _ = transitions.shape
    expected_rows = n_states * n_states
    if features.shape[0] != expected_rows:
        raise ValueError(f"features has {features.shape[0]} rows, expected {expected_rows}")
    features_jax = jnp.asarray(features, dtype=jnp.float32)
    state = create_action_head_train_state(seed, features.shape[-1], n_actions, config.hidden_dims, config.lr)

    start = time.perf_counter()
    support_q = support_one_step_q(transitions, gamma)
    init_q = support_q if method == "neural_support_trl" else one_step_q(transitions, gamma)
    state, loss = fit_action_head_target(state, features_jax, init_q, config.warmup_steps, config)

    if method == "neural_support_trl":
        target = train_support_transitive_value(transitions, gamma, operator_iters)
        total_steps = operator_iters * config.steps_per_iter
        state, loss = fit_action_head_target(state, features_jax, target, total_steps, config)
        q = predict_action_head_q(state, features, n_states, n_actions)
        stats = NeuralToyStats(
            final_loss=float(loss),
            train_seconds=time.perf_counter() - start,
            operator_iters=int(operator_iters),
            warmup_steps=int(config.warmup_steps),
            steps_per_iter=int(config.steps_per_iter),
        )
        return q, stats

    for iteration in range(1, operator_iters + 1):
        q = predict_action_head_q(state, features, n_states, n_actions)
        target = operator_target(q, transitions, gamma, method, support_q)
        state, loss = fit_action_head_target(state, features_jax, target, config.steps_per_iter, config)
        if config.log_interval > 0 and (iteration == 1 or iteration % config.log_interval == 0):
            print(
                f"[qhead] method={method} iter={iteration} loss={loss:.6f}",
                flush=True,
            )

    q = predict_action_head_q(state, features, n_states, n_actions)
    stats = NeuralToyStats(
        final_loss=float(loss),
        train_seconds=time.perf_counter() - start,
        operator_iters=int(operator_iters),
        warmup_steps=int(config.warmup_steps),
        steps_per_iter=int(config.steps_per_iter),
    )
    return q, stats


def train_action_head_buffered_neural_critic_from_features(
    transitions: np.ndarray,
    gamma: float,
    method: str,
    operator_iters: int,
    seed: int,
    config: NeuralToyConfig,
    features: np.ndarray,
    final_steps: int = 0,
    monotone: bool = True,
    reset_final: bool = False,
    waypoint_mode: str = "all",
    waypoint_k: int = 0,
) -> tuple[np.ndarray, NeuralToyStats]:
    n_states, n_actions, _ = transitions.shape
    expected_rows = n_states * n_states
    if features.shape[0] != expected_rows:
        raise ValueError(f"features has {features.shape[0]} rows, expected {expected_rows}")
    if method != "neural_sto_trl":
        raise ValueError("buffered action-head critic currently supports neural_sto_trl only")
    features_jax = jnp.asarray(features, dtype=jnp.float32)
    state = create_action_head_train_state(seed, features.shape[-1], n_actions, config.hidden_dims, config.lr)

    start = time.perf_counter()
    support_q = support_one_step_q(transitions, gamma)
    target_buffer = one_step_q(transitions, gamma)
    state, loss = fit_action_head_target(state, features_jax, target_buffer, config.warmup_steps, config)

    for iteration in range(1, operator_iters + 1):
        proposed = operator_target(
            target_buffer,
            transitions,
            gamma,
            method,
            support_q,
            waypoint_mode=waypoint_mode,
            waypoint_k=waypoint_k,
        )
        target_buffer = np.maximum(target_buffer, proposed) if monotone else proposed
        state, loss = fit_action_head_target(state, features_jax, target_buffer, config.steps_per_iter, config)
        if config.log_interval > 0 and (iteration == 1 or iteration % config.log_interval == 0):
            q = predict_action_head_q(state, features, n_states, n_actions)
            fit_mse = float(np.mean((q - target_buffer) ** 2))
            print(
                f"[qhead-buffer] method={method} iter={iteration} "
                f"loss={loss:.6f} fit_mse={fit_mse:.6f}",
                flush=True,
            )

    if final_steps > 0:
        if reset_final:
            state = create_action_head_train_state(
                seed + 1_000_003,
                features.shape[-1],
                n_actions,
                config.hidden_dims,
                config.lr,
            )
        state, loss = fit_action_head_target(state, features_jax, target_buffer, final_steps, config)
        if config.log_interval > 0:
            q = predict_action_head_q(state, features, n_states, n_actions)
            fit_mse = float(np.mean((q - target_buffer) ** 2))
            print(
                f"[qhead-buffer] method={method} final_steps={final_steps} "
                f"reset_final={int(reset_final)} loss={loss:.6f} fit_mse={fit_mse:.6f}",
                flush=True,
            )

    q = predict_action_head_q(state, features, n_states, n_actions)
    stats = NeuralToyStats(
        final_loss=float(loss),
        train_seconds=time.perf_counter() - start,
        operator_iters=int(operator_iters),
        warmup_steps=int(config.warmup_steps),
        steps_per_iter=int(config.steps_per_iter),
    )
    return q, stats


def train_action_head_sampled_buffered_neural_critic_from_features(
    sample_transitions: np.ndarray,
    gamma: float,
    method: str,
    operator_iters: int,
    seed: int,
    config: NeuralToyConfig,
    features: np.ndarray,
    final_steps: int = 0,
    reset_final: bool = True,
    next_samples_per_row: int = 32,
    waypoint_candidates_per_pair: int = 16,
    waypoint_k: int = 4,
) -> tuple[np.ndarray, NeuralToyStats]:
    n_states, n_actions, _ = sample_transitions.shape
    expected_rows = n_states * n_states
    if features.shape[0] != expected_rows:
        raise ValueError(f"features has {features.shape[0]} rows, expected {expected_rows}")
    if method != "neural_sto_trl":
        raise ValueError("sampled-buffered action-head critic currently supports neural_sto_trl only")
    features_jax = jnp.asarray(features, dtype=jnp.float32)
    state = create_action_head_train_state(seed, features.shape[-1], n_actions, config.hidden_dims, config.lr)

    start = time.perf_counter()
    next_state_samples = sample_next_states_from_transitions(
        sample_transitions,
        next_samples_per_row,
        seed + 17_017,
    )
    support_q = support_one_step_q_from_samples(next_state_samples, gamma)
    target_buffer = one_step_q_from_samples(next_state_samples, gamma)
    state, loss = fit_action_head_target(state, features_jax, target_buffer, config.warmup_steps, config)

    for iteration in range(1, operator_iters + 1):
        waypoint_candidates = sample_waypoint_candidates(
            n_states,
            waypoint_candidates_per_pair,
            seed + 1_000_003 + 7919 * iteration,
        )
        proposed = sampled_operator_target(
            target_buffer,
            next_state_samples,
            gamma,
            method,
            support_q,
            waypoint_candidates,
            waypoint_k,
        )
        target_buffer = np.maximum(target_buffer, proposed)
        state, loss = fit_action_head_target(state, features_jax, target_buffer, config.steps_per_iter, config)
        if config.log_interval > 0 and (iteration == 1 or iteration % config.log_interval == 0):
            q = predict_action_head_q(state, features, n_states, n_actions)
            fit_mse = float(np.mean((q - target_buffer) ** 2))
            print(
                f"[qhead-sampled-buffer] method={method} iter={iteration} "
                f"loss={loss:.6f} fit_mse={fit_mse:.6f}",
                flush=True,
            )

    if final_steps > 0:
        if reset_final:
            state = create_action_head_train_state(
                seed + 2_000_003,
                features.shape[-1],
                n_actions,
                config.hidden_dims,
                config.lr,
            )
        state, loss = fit_action_head_target(state, features_jax, target_buffer, final_steps, config)
        if config.log_interval > 0:
            q = predict_action_head_q(state, features, n_states, n_actions)
            fit_mse = float(np.mean((q - target_buffer) ** 2))
            print(
                f"[qhead-sampled-buffer] method={method} final_steps={final_steps} "
                f"reset_final={int(reset_final)} loss={loss:.6f} fit_mse={fit_mse:.6f}",
                flush=True,
            )

    q = predict_action_head_q(state, features, n_states, n_actions)
    stats = NeuralToyStats(
        final_loss=float(loss),
        train_seconds=time.perf_counter() - start,
        operator_iters=int(operator_iters),
        warmup_steps=int(config.warmup_steps),
        steps_per_iter=int(config.steps_per_iter),
    )
    return q, stats


def train_action_head_sampled_target_network_neural_critic_from_features(
    sample_transitions: np.ndarray,
    gamma: float,
    method: str,
    operator_iters: int,
    seed: int,
    config: NeuralToyConfig,
    features: np.ndarray,
    next_samples_per_row: int = 32,
    waypoint_candidates_per_pair: int = 16,
    waypoint_k: int = 4,
    monotone_targets: bool = True,
) -> tuple[np.ndarray, NeuralToyStats]:
    n_states, n_actions, _ = sample_transitions.shape
    expected_rows = n_states * n_states
    if features.shape[0] != expected_rows:
        raise ValueError(f"features has {features.shape[0]} rows, expected {expected_rows}")
    if method != "neural_sto_trl":
        raise ValueError("sampled target-network action-head critic currently supports neural_sto_trl only")
    features_jax = jnp.asarray(features, dtype=jnp.float32)
    state = create_action_head_train_state(seed, features.shape[-1], n_actions, config.hidden_dims, config.lr)

    start = time.perf_counter()
    next_state_samples = sample_next_states_from_transitions(
        sample_transitions,
        next_samples_per_row,
        seed + 23_017,
    )
    support_q = support_one_step_q_from_samples(next_state_samples, gamma)
    init_target = one_step_q_from_samples(next_state_samples, gamma)
    state, loss = fit_action_head_target(state, features_jax, init_target, config.warmup_steps, config)

    for iteration in range(1, operator_iters + 1):
        q_target = predict_action_head_q(state, features, n_states, n_actions)
        waypoint_candidates = sample_waypoint_candidates(
            n_states,
            waypoint_candidates_per_pair,
            seed + 3_000_007 + 7919 * iteration,
        )
        target = sampled_operator_target(
            q_target,
            next_state_samples,
            gamma,
            method,
            support_q,
            waypoint_candidates,
            waypoint_k,
        )
        if monotone_targets:
            target = np.maximum(q_target, target)
        state, loss = fit_action_head_target(state, features_jax, target, config.steps_per_iter, config)
        if config.log_interval > 0 and (iteration == 1 or iteration % config.log_interval == 0):
            q = predict_action_head_q(state, features, n_states, n_actions)
            fit_mse = float(np.mean((q - target) ** 2))
            target_agreement = float(np.mean(np.argmax(q, axis=1) == np.argmax(target, axis=1)))
            print(
                f"[qhead-sampled-target-net] method={method} iter={iteration} "
                f"loss={loss:.6f} fit_mse={fit_mse:.6f} "
                f"target_action_agreement={target_agreement:.3f}",
                flush=True,
            )

    q = predict_action_head_q(state, features, n_states, n_actions)
    stats = NeuralToyStats(
        final_loss=float(loss),
        train_seconds=time.perf_counter() - start,
        operator_iters=int(operator_iters),
        warmup_steps=int(config.warmup_steps),
        steps_per_iter=int(config.steps_per_iter),
    )
    return q, stats


def train_action_head_sampled_target_replay_neural_critic_from_features(
    sample_transitions: np.ndarray,
    gamma: float,
    method: str,
    operator_iters: int,
    seed: int,
    config: NeuralToyConfig,
    features: np.ndarray,
    next_samples_per_row: int = 32,
    waypoint_candidates_per_pair: int = 16,
    waypoint_k: int = 4,
) -> tuple[np.ndarray, NeuralToyStats]:
    n_states, n_actions, _ = sample_transitions.shape
    expected_rows = n_states * n_states
    if features.shape[0] != expected_rows:
        raise ValueError(f"features has {features.shape[0]} rows, expected {expected_rows}")
    if method != "neural_sto_trl":
        raise ValueError("sampled target-replay action-head critic currently supports neural_sto_trl only")
    features_jax = jnp.asarray(features, dtype=jnp.float32)

    start = time.perf_counter()
    next_state_samples = sample_next_states_from_transitions(
        sample_transitions,
        next_samples_per_row,
        seed + 29_017,
    )
    support_q = support_one_step_q_from_samples(next_state_samples, gamma)
    replay_target = one_step_q_from_samples(next_state_samples, gamma)
    state = create_action_head_train_state(seed, features.shape[-1], n_actions, config.hidden_dims, config.lr)
    state, loss = fit_action_head_target(state, features_jax, replay_target, config.warmup_steps, config)

    for iteration in range(1, operator_iters + 1):
        q_target = predict_action_head_q(state, features, n_states, n_actions)
        waypoint_candidates = sample_waypoint_candidates(
            n_states,
            waypoint_candidates_per_pair,
            seed + 4_000_009 + 7919 * iteration,
        )
        proposed = sampled_operator_target(
            q_target,
            next_state_samples,
            gamma,
            method,
            support_q,
            waypoint_candidates,
            waypoint_k,
        )
        replay_target = np.maximum.reduce([replay_target, q_target, proposed])
        state = create_action_head_train_state(
            seed + 4099 * iteration,
            features.shape[-1],
            n_actions,
            config.hidden_dims,
            config.lr,
        )
        state, loss = fit_action_head_target(state, features_jax, replay_target, config.steps_per_iter, config)
        if config.log_interval > 0 and (iteration == 1 or iteration % config.log_interval == 0):
            q = predict_action_head_q(state, features, n_states, n_actions)
            fit_mse = float(np.mean((q - replay_target) ** 2))
            replay_agreement = float(np.mean(np.argmax(q, axis=1) == np.argmax(replay_target, axis=1)))
            print(
                f"[qhead-sampled-target-replay] method={method} iter={iteration} "
                f"loss={loss:.6f} fit_mse={fit_mse:.6f} "
                f"replay_action_agreement={replay_agreement:.3f}",
                flush=True,
            )

    q = predict_action_head_q(state, features, n_states, n_actions)
    stats = NeuralToyStats(
        final_loss=float(loss),
        train_seconds=time.perf_counter() - start,
        operator_iters=int(operator_iters),
        warmup_steps=int(config.warmup_steps),
        steps_per_iter=int(config.steps_per_iter),
    )
    return q, stats


def train_action_head_self_buffered_neural_critic_from_features(
    transitions: np.ndarray,
    gamma: float,
    method: str,
    operator_iters: int,
    seed: int,
    config: NeuralToyConfig,
    features: np.ndarray,
    final_steps: int = 0,
) -> tuple[np.ndarray, NeuralToyStats]:
    n_states, n_actions, _ = transitions.shape
    expected_rows = n_states * n_states
    if features.shape[0] != expected_rows:
        raise ValueError(f"features has {features.shape[0]} rows, expected {expected_rows}")
    if method != "neural_sto_trl":
        raise ValueError("self-buffered action-head critic currently supports neural_sto_trl only")
    features_jax = jnp.asarray(features, dtype=jnp.float32)
    state = create_action_head_train_state(seed, features.shape[-1], n_actions, config.hidden_dims, config.lr)

    start = time.perf_counter()
    support_q = support_one_step_q(transitions, gamma)
    target_buffer = one_step_q(transitions, gamma)
    state, loss = fit_action_head_target(state, features_jax, target_buffer, config.warmup_steps, config)

    for iteration in range(1, operator_iters + 1):
        q = predict_action_head_q(state, features, n_states, n_actions)
        proposed = operator_target(q, transitions, gamma, method, support_q)
        target_buffer = np.maximum(target_buffer, proposed)
        state, loss = fit_action_head_target(state, features_jax, target_buffer, config.steps_per_iter, config)
        if config.log_interval > 0 and (iteration == 1 or iteration % config.log_interval == 0):
            q = predict_action_head_q(state, features, n_states, n_actions)
            fit_mse = float(np.mean((q - target_buffer) ** 2))
            print(
                f"[qhead-self-buffer] method={method} iter={iteration} "
                f"loss={loss:.6f} fit_mse={fit_mse:.6f}",
                flush=True,
            )

    if final_steps > 0:
        state, loss = fit_action_head_target(state, features_jax, target_buffer, final_steps, config)
        if config.log_interval > 0:
            q = predict_action_head_q(state, features, n_states, n_actions)
            fit_mse = float(np.mean((q - target_buffer) ** 2))
            print(
                f"[qhead-self-buffer] method={method} final_steps={final_steps} "
                f"loss={loss:.6f} fit_mse={fit_mse:.6f}",
                flush=True,
            )

    q = predict_action_head_q(state, features, n_states, n_actions)
    stats = NeuralToyStats(
        final_loss=float(loss),
        train_seconds=time.perf_counter() - start,
        operator_iters=int(operator_iters),
        warmup_steps=int(config.warmup_steps),
        steps_per_iter=int(config.steps_per_iter),
    )
    return q, stats


def train_action_head_fresh_iter_neural_critic_from_features(
    transitions: np.ndarray,
    gamma: float,
    method: str,
    operator_iters: int,
    seed: int,
    config: NeuralToyConfig,
    features: np.ndarray,
) -> tuple[np.ndarray, NeuralToyStats]:
    n_states, n_actions, _ = transitions.shape
    expected_rows = n_states * n_states
    if features.shape[0] != expected_rows:
        raise ValueError(f"features has {features.shape[0]} rows, expected {expected_rows}")
    if method != "neural_sto_trl":
        raise ValueError("fresh-iter action-head critic currently supports neural_sto_trl only")
    features_jax = jnp.asarray(features, dtype=jnp.float32)

    start = time.perf_counter()
    support_q = support_one_step_q(transitions, gamma)
    init_q = one_step_q(transitions, gamma)
    state = create_action_head_train_state(seed, features.shape[-1], n_actions, config.hidden_dims, config.lr)
    state, loss = fit_action_head_target(state, features_jax, init_q, config.warmup_steps, config)

    for iteration in range(1, operator_iters + 1):
        q = predict_action_head_q(state, features, n_states, n_actions)
        target = operator_target(q, transitions, gamma, method, support_q)
        state = create_action_head_train_state(
            seed + 1009 * iteration,
            features.shape[-1],
            n_actions,
            config.hidden_dims,
            config.lr,
        )
        state, loss = fit_action_head_target(state, features_jax, target, config.steps_per_iter, config)
        if config.log_interval > 0 and (iteration == 1 or iteration % config.log_interval == 0):
            q = predict_action_head_q(state, features, n_states, n_actions)
            fit_mse = float(np.mean((q - target) ** 2))
            print(
                f"[qhead-fresh-iter] method={method} iter={iteration} "
                f"loss={loss:.6f} fit_mse={fit_mse:.6f}",
                flush=True,
            )

    q = predict_action_head_q(state, features, n_states, n_actions)
    stats = NeuralToyStats(
        final_loss=float(loss),
        train_seconds=time.perf_counter() - start,
        operator_iters=int(operator_iters),
        warmup_steps=int(config.warmup_steps),
        steps_per_iter=int(config.steps_per_iter),
    )
    return q, stats


def train_action_head_guided_rank_neural_critic_from_features(
    transitions: np.ndarray,
    gamma: float,
    method: str,
    operator_iters: int,
    seed: int,
    config: NeuralToyConfig,
    features: np.ndarray,
) -> tuple[np.ndarray, NeuralToyStats]:
    n_states, n_actions, _ = transitions.shape
    expected_rows = n_states * n_states
    if features.shape[0] != expected_rows:
        raise ValueError(f"features has {features.shape[0]} rows, expected {expected_rows}")
    if method != "neural_sto_trl":
        raise ValueError("guided-rank action-head critic currently supports neural_sto_trl only")
    features_jax = jnp.asarray(features, dtype=jnp.float32)

    start = time.perf_counter()
    support_q = support_one_step_q(transitions, gamma)
    guide_target = one_step_q(transitions, gamma)
    for _ in range(operator_iters):
        guide_target = operator_target(guide_target, transitions, gamma, method, support_q)

    state = create_action_head_train_state(seed, features.shape[-1], n_actions, config.hidden_dims, config.lr)
    init_q = one_step_q(transitions, gamma)
    state, loss = fit_action_head_target(
        state,
        features_jax,
        init_q,
        config.warmup_steps,
        config,
        label_target=guide_target,
    )

    for iteration in range(1, operator_iters + 1):
        q = predict_action_head_q(state, features, n_states, n_actions)
        target = operator_target(q, transitions, gamma, method, support_q)
        state, loss = fit_action_head_target(
            state,
            features_jax,
            target,
            config.steps_per_iter,
            config,
            label_target=guide_target,
        )
        if config.log_interval > 0 and (iteration == 1 or iteration % config.log_interval == 0):
            q = predict_action_head_q(state, features, n_states, n_actions)
            fit_mse = float(np.mean((q - target) ** 2))
            guide_action_agreement = float(np.mean(np.argmax(q, axis=1) == np.argmax(guide_target, axis=1)))
            print(
                f"[qhead-guided-rank] method={method} iter={iteration} "
                f"loss={loss:.6f} fit_mse={fit_mse:.6f} guide_action_agreement={guide_action_agreement:.3f}",
                flush=True,
            )

    q = predict_action_head_q(state, features, n_states, n_actions)
    stats = NeuralToyStats(
        final_loss=float(loss),
        train_seconds=time.perf_counter() - start,
        operator_iters=int(operator_iters),
        warmup_steps=int(config.warmup_steps),
        steps_per_iter=int(config.steps_per_iter),
    )
    return q, stats


def train_action_head_generated_target_neural_critic_from_features(
    transitions: np.ndarray,
    gamma: float,
    method: str,
    operator_iters: int,
    seed: int,
    config: NeuralToyConfig,
    features: np.ndarray,
    fit_steps: int,
) -> tuple[np.ndarray, NeuralToyStats]:
    n_states, n_actions, _ = transitions.shape
    expected_rows = n_states * n_states
    if features.shape[0] != expected_rows:
        raise ValueError(f"features has {features.shape[0]} rows, expected {expected_rows}")
    if method != "neural_sto_trl":
        raise ValueError("generated-target action-head critic currently supports neural_sto_trl only")

    support_q = support_one_step_q(transitions, gamma)
    target = one_step_q(transitions, gamma)
    for _ in range(operator_iters):
        target = operator_target(target, transitions, gamma, method, support_q)

    features_jax = jnp.asarray(features, dtype=jnp.float32)
    state = create_action_head_train_state(seed, features.shape[-1], n_actions, config.hidden_dims, config.lr)
    start = time.perf_counter()
    state, loss = fit_action_head_target(state, features_jax, target, fit_steps, config)
    q = predict_action_head_q(state, features, n_states, n_actions)
    stats = NeuralToyStats(
        final_loss=float(loss),
        train_seconds=time.perf_counter() - start,
        operator_iters=int(operator_iters),
        warmup_steps=0,
        steps_per_iter=int(fit_steps),
    )
    return q, stats


def train_neural_critic_from_features(
    transitions: np.ndarray,
    gamma: float,
    method: str,
    operator_iters: int,
    seed: int,
    config: NeuralToyConfig,
    features: np.ndarray,
) -> tuple[np.ndarray, NeuralToyStats]:
    n_states, n_actions, _ = transitions.shape
    expected_rows = n_states * n_actions * n_states
    if features.shape[0] != expected_rows:
        raise ValueError(f"features has {features.shape[0]} rows, expected {expected_rows}")
    features_jax = jnp.asarray(features, dtype=jnp.float32)
    state = create_train_state(seed, features.shape[-1], config.hidden_dims, config.lr)

    start = time.perf_counter()
    support_q = support_one_step_q(transitions, gamma)
    init_q = support_q if method == "neural_support_trl" else one_step_q(transitions, gamma)
    state, loss = fit_target(state, features_jax, init_q, config.warmup_steps, config)

    if method == "neural_support_trl":
        target = train_support_transitive_value(transitions, gamma, operator_iters)
        total_steps = operator_iters * config.steps_per_iter
        state, loss = fit_target(state, features_jax, target, total_steps, config)
        q = predict_q(state, features, n_states, n_actions)
        stats = NeuralToyStats(
            final_loss=float(loss),
            train_seconds=time.perf_counter() - start,
            operator_iters=int(operator_iters),
            warmup_steps=int(config.warmup_steps),
            steps_per_iter=int(config.steps_per_iter),
        )
        return q, stats

    for iteration in range(1, operator_iters + 1):
        q = predict_q(state, features, n_states, n_actions)
        target = operator_target(q, transitions, gamma, method, support_q)
        state, loss = fit_target(state, features_jax, target, config.steps_per_iter, config)
        if config.log_interval > 0 and (iteration == 1 or iteration % config.log_interval == 0):
            print(
                f"[neural] method={method} iter={iteration} loss={loss:.6f}",
                flush=True,
            )

    q = predict_q(state, features, n_states, n_actions)
    stats = NeuralToyStats(
        final_loss=float(loss),
        train_seconds=time.perf_counter() - start,
        operator_iters=int(operator_iters),
        warmup_steps=int(config.warmup_steps),
        steps_per_iter=int(config.steps_per_iter),
    )
    return q, stats


def fit_action_head_neural_critic_to_target(
    transitions: np.ndarray,
    target: np.ndarray,
    seed: int,
    config: NeuralToyConfig,
    features: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, NeuralToyStats]:
    n_states, n_actions, _ = transitions.shape
    expected_rows = n_states * n_states
    if features.shape[0] != expected_rows:
        raise ValueError(f"features has {features.shape[0]} rows, expected {expected_rows}")
    if target.shape != transitions.shape:
        raise ValueError(f"target shape {target.shape} does not match transitions {transitions.shape}")

    features_jax = jnp.asarray(features, dtype=jnp.float32)
    state = create_action_head_train_state(seed, features.shape[-1], n_actions, config.hidden_dims, config.lr)
    start = time.perf_counter()
    state, loss = fit_action_head_target(state, features_jax, target, steps, config)
    q = predict_action_head_q(state, features, n_states, n_actions)
    stats = NeuralToyStats(
        final_loss=float(loss),
        train_seconds=time.perf_counter() - start,
        operator_iters=0,
        warmup_steps=0,
        steps_per_iter=int(steps),
    )
    return q, stats


def fit_neural_critic_to_target(
    transitions: np.ndarray,
    target: np.ndarray,
    seed: int,
    config: NeuralToyConfig,
    features: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, NeuralToyStats]:
    n_states, n_actions, _ = transitions.shape
    expected_rows = n_states * n_actions * n_states
    if features.shape[0] != expected_rows:
        raise ValueError(f"features has {features.shape[0]} rows, expected {expected_rows}")
    if target.shape != transitions.shape:
        raise ValueError(f"target shape {target.shape} does not match transitions {transitions.shape}")

    features_jax = jnp.asarray(features, dtype=jnp.float32)
    state = create_train_state(seed, features.shape[-1], config.hidden_dims, config.lr)
    start = time.perf_counter()
    state, loss = fit_target(state, features_jax, target, steps, config)
    q = predict_q(state, features, n_states, n_actions)
    stats = NeuralToyStats(
        final_loss=float(loss),
        train_seconds=time.perf_counter() - start,
        operator_iters=0,
        warmup_steps=0,
        steps_per_iter=int(steps),
    )
    return q, stats
