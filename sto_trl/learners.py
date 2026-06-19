from __future__ import annotations

import math

import numpy as np

from .data import Trajectory


def _future_first_hits(states: np.ndarray, n_states: int, start_idx: int) -> np.ndarray:
    hits = np.full(n_states, -1, dtype=np.int64)
    for j in range(start_idx + 1, len(states)):
        g = int(states[j])
        if hits[g] < 0:
            hits[g] = j - start_idx
    return hits


def train_mc(
    trajectories: list[Trajectory],
    n_states: int,
    n_actions: int,
    gamma: float,
    include_unreached_goals: bool,
) -> np.ndarray:
    sums = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    counts = np.zeros_like(sums)
    for traj in trajectories:
        for i, (s, a) in enumerate(zip(traj.states[:-1], traj.actions)):
            s = int(s)
            a = int(a)
            hits = _future_first_hits(traj.states, n_states, i)
            if include_unreached_goals:
                labels = np.zeros(n_states, dtype=np.float64)
                reached = hits >= 0
                labels[reached] = gamma ** hits[reached]
                sums[s, a, :] += labels
                counts[s, a, :] += 1.0
            else:
                for g, dt in enumerate(hits):
                    if dt >= 0:
                        sums[s, a, g] += gamma**dt
                        counts[s, a, g] += 1.0
    q = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    for g in range(n_states):
        q[g, :, g] = 1.0
    return q


def train_trl_realized(
    trajectories: list[Trajectory],
    n_states: int,
    n_actions: int,
    gamma: float,
    mode: str = "raw",
    expectile: float = 0.9,
    lr: float = 0.35,
    epochs: int = 40,
    updates_per_epoch: int = 5_000,
    seed: int = 0,
) -> np.ndarray:
    """In-trajectory TRL table learner.

    This intentionally follows the deterministic/realized-trajectory semantics:
    only goals that appear later in the same trajectory produce targets. That is
    the failure mode we want to diagnose under stochastic dynamics.
    """
    del lr, updates_per_epoch, seed
    if mode not in {"raw", "log"}:
        raise ValueError("mode must be 'raw' or 'log'")

    if mode == "raw":
        q = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
        for traj in trajectories:
            for s, a, sp in zip(traj.states[:-1], traj.actions, traj.states[1:]):
                q[int(s), int(a), int(sp)] = max(q[int(s), int(a), int(sp)], gamma)
        for g in range(n_states):
            q[g, :, g] = 1.0
    else:
        large_d = 1e6
        d = np.full((n_states, n_actions, n_states), large_d, dtype=np.float64)
        for g in range(n_states):
            d[g, :, g] = 0.0
        for traj in trajectories:
            for s, a, sp in zip(traj.states[:-1], traj.actions, traj.states[1:]):
                d[int(s), int(a), int(sp)] = min(d[int(s), int(a), int(sp)], 1.0)

    # Ideal tabular TRL: solve shorter in-trajectory chunks before longer ones.
    # This is the deterministic divide-and-conquer sanity target; in stochastic
    # MDPs it still treats lucky realized paths as reliable, exposing the bias.
    max_span = max((len(traj.actions) for traj in trajectories), default=0)
    passes = max(1, min(3, epochs))
    for _pass in range(passes):
        if mode == "raw":
            v = q.max(axis=1)
        else:
            v_d = d.min(axis=1)
        for span in range(2, max_span + 1):
            for traj in trajectories:
                if len(traj.states) <= span:
                    continue
                for i in range(0, len(traj.states) - span):
                    j = i + span
                    s = int(traj.states[i])
                    a = int(traj.actions[i])
                    g = int(traj.states[j])
                    if mode == "raw":
                        target = 0.0
                        for k in range(i + 1, j):
                            w = int(traj.states[k])
                            left = gamma if k - i <= 1 else q[s, a, w]
                            right = gamma if j - k <= 1 else v[w, g]
                            target = max(target, float(left * right))
                        pred = q[s, a, g]
                        weight = expectile if pred < target else 1.0 - expectile
                        q[s, a, g] = np.clip(pred + weight * (target - pred), 0.0, 1.0)
                    else:
                        target_d = float("inf")
                        for k in range(i + 1, j):
                            w = int(traj.states[k])
                            left_d = 1.0 if k - i <= 1 else d[s, a, w]
                            right_d = 1.0 if j - k <= 1 else v_d[w, g]
                            target_d = min(target_d, float(left_d + right_d))
                        pred_d = d[s, a, g]
                        weight = expectile if pred_d > target_d else 1.0 - expectile
                        d[s, a, g] = max(0.0, pred_d + weight * (target_d - pred_d))

    if mode == "raw":
        return q
    q = np.zeros_like(d)
    finite = d < 1e5
    q[finite] = gamma ** d[finite]
    for g in range(n_states):
        q[g, :, g] = 1.0
    return q


def bellman_backup(q: np.ndarray, transitions: np.ndarray, gamma: float) -> np.ndarray:
    n_states, _n_actions, _ = transitions.shape
    v = q.max(axis=1)
    diag = np.arange(n_states)
    v[diag, diag] = 1.0
    target = gamma * np.einsum("san,ng->sag", transitions, v)
    target[diag, :, diag] = 1.0
    return np.clip(target, 0.0, 1.0)


def stochastic_transitive_backup(q: np.ndarray, gamma: float) -> np.ndarray:
    n_states, n_actions, _ = q.shape
    v = q.max(axis=1)
    target = np.zeros_like(q)
    for g in range(n_states):
        best = np.zeros((n_states, n_actions), dtype=np.float64)
        for w in range(n_states):
            candidate = q[:, :, w] * v[w, g]
            # For action-values, w == current state is a tautological first leg:
            # Q(s,a,s)=1 would copy V(s,g) into every action regardless of a.
            candidate[w, :] = 0.0
            best = np.maximum(best, candidate)
        best[g, :] = 1.0
        target[:, :, g] = best
    return np.clip(target, 0.0, 1.0)


def train_model_value(
    transitions: np.ndarray,
    gamma: float,
    n_iters: int,
    use_transitive: bool,
    init: str = "one_step",
) -> np.ndarray:
    n_states, n_actions, _ = transitions.shape
    if init == "zero":
        q = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
        for g in range(n_states):
            q[g, :, g] = 1.0
    elif init == "one_step":
        q = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
        for g in range(n_states):
            q[:, :, g] = gamma * transitions[:, :, g]
            q[g, :, g] = 1.0
    else:
        raise ValueError("init must be 'zero' or 'one_step'")

    for _ in range(n_iters):
        bellman = bellman_backup(q, transitions, gamma)
        if use_transitive:
            tr = stochastic_transitive_backup(q, gamma)
            q = np.maximum(bellman, tr)
        else:
            q = bellman
    return np.clip(q, 0.0, 1.0)


def train_support_transitive_value(
    transitions: np.ndarray,
    gamma: float,
    n_iters: int,
) -> np.ndarray:
    """Optimistic deterministic-TRL value on transition support.

    This baseline treats every observed stochastic next state as if it were a
    reliable one-step transition. It is intentionally optimistic under
    stochastic dynamics and diagnoses the deterministic TRL failure mode.
    """
    n_states, _n_actions, _ = transitions.shape
    q = np.zeros_like(transitions, dtype=np.float64)
    q[transitions > 0.0] = gamma
    for g in range(n_states):
        q[g, :, g] = 1.0
    for _ in range(n_iters):
        q = np.maximum(q, stochastic_transitive_backup(q, gamma))
    return np.clip(q, 0.0, 1.0)


def recommended_transitive_iters(longest_horizon: int) -> int:
    return max(1, int(math.ceil(math.log2(max(longest_horizon, 2)))) + 1)
