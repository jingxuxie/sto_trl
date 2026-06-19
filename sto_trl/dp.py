from __future__ import annotations

import numpy as np


def optimal_reachability(
    transitions: np.ndarray,
    gamma: float,
    n_iters: int = 3_000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Optimal discounted hitting reachability U(s,a,g) for every goal."""
    n_states, n_actions, _ = transitions.shape
    q = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    for g in range(n_states):
        q[g, :, g] = 1.0

    for _ in range(n_iters):
        old = q.copy()
        v = old.max(axis=1)
        diag = np.arange(n_states)
        v[diag, diag] = 1.0
        target = gamma * np.einsum("san,ng->sag", transitions, v)
        target[diag, :, diag] = 1.0
        q = target
        if float(np.max(np.abs(q - old))) < tol:
            break
    return q


def policy_reachability(
    transitions: np.ndarray,
    policy: np.ndarray,
    gamma: float,
    goal: int,
    n_iters: int = 10_000,
    tol: float = 1e-12,
) -> np.ndarray:
    n_states = transitions.shape[0]
    v = np.zeros(n_states, dtype=np.float64)
    v[goal] = 1.0
    for _ in range(n_iters):
        old = v.copy()
        continuation = old.copy()
        continuation[goal] = 1.0
        for s in range(n_states):
            if s == goal:
                v[s] = 1.0
            else:
                v[s] = gamma * float(transitions[s, policy[s]].dot(continuation))
        if float(np.max(np.abs(v - old))) < tol:
            break
    return v


def rollout_policy_success(
    transitions: np.ndarray,
    policy: np.ndarray,
    start: int,
    goal: int,
    max_steps: int,
    episodes: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    steps_to_success = []
    successes = 0
    n_states = transitions.shape[0]
    for _ in range(episodes):
        state = start
        for step in range(1, max_steps + 1):
            action = int(policy[state])
            state = int(rng.choice(n_states, p=transitions[state, action]))
            if state == goal:
                successes += 1
                steps_to_success.append(step)
                break
    median_steps = float(np.median(steps_to_success)) if steps_to_success else float("inf")
    return successes / episodes, median_steps


def greedy_policy(q: np.ndarray, goal: int) -> np.ndarray:
    return np.argmax(q[:, :, goal], axis=1).astype(np.int64)


def discounted_distance(q: np.ndarray, gamma: float, eps: float = 1e-12) -> np.ndarray:
    return -np.log(np.maximum(q, eps)) / (-np.log(gamma))
