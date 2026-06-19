from __future__ import annotations

import numpy as np

from .dp import discounted_distance, greedy_policy, policy_reachability, rollout_policy_success


def triangle_violation_rate(
    q: np.ndarray,
    gamma: float,
    samples: int = 20_000,
    seed: int = 0,
    eps: float = 1e-6,
) -> float:
    rng = np.random.default_rng(seed)
    v = q.max(axis=1)
    d = discounted_distance(v, gamma)
    n_states = v.shape[0]
    violations = 0
    total = 0
    for _ in range(samples):
        s = int(rng.integers(n_states))
        w = int(rng.integers(n_states))
        g = int(rng.integers(n_states))
        if not np.isfinite(d[s, g] + d[s, w] + d[w, g]):
            continue
        violations += int(d[s, g] > d[s, w] + d[w, g] + eps)
        total += 1
    return violations / max(total, 1)


def start_goal_metrics(
    q: np.ndarray,
    q_star: np.ndarray,
    transitions: np.ndarray,
    gamma: float,
    start: int,
    goal: int,
    max_steps: int,
    rollout_seed: int,
    rollout_episodes: int = 500,
) -> dict[str, float]:
    policy = greedy_policy(q, goal)
    v_pi = policy_reachability(transitions, policy, gamma, goal)
    v_star = q_star.max(axis=1)[:, goal]
    success, median_steps = rollout_policy_success(
        transitions,
        policy,
        start,
        goal,
        max_steps=max_steps,
        episodes=rollout_episodes,
        seed=rollout_seed,
    )
    safe_q = float(q[start, 0, goal])
    risky_q = float(q[start, 1, goal]) if q.shape[1] > 1 else float("nan")
    safe_star = float(q_star[start, 0, goal])
    risky_star = float(q_star[start, 1, goal]) if q.shape[1] > 1 else float("nan")
    return {
        "pred_start_safe": safe_q,
        "pred_start_risky": risky_q,
        "true_start_safe": safe_star,
        "true_start_risky": risky_star,
        "start_value": float(v_pi[start]),
        "optimal_start_value": float(v_star[start]),
        "regret": float(v_star[start] - v_pi[start]),
        "success_rate": float(success),
        "median_steps": float(median_steps),
        "start_action": float(policy[start]),
        "risky_action": float(policy[start] == 1),
        "risky_overestimate_ratio": float(risky_q / max(risky_star, 1e-12)),
    }


def value_metrics(
    q: np.ndarray,
    q_star: np.ndarray,
    gamma: float,
    long_horizon_threshold: float = 8.0,
    triangle_samples: int = 20_000,
) -> dict[str, float]:
    err = q - q_star
    d = discounted_distance(q, gamma)
    d_star = discounted_distance(q_star, gamma)
    mask = q_star > 1e-10
    long_mask = mask & (d_star > long_horizon_threshold)
    return {
        "value_mse": float(np.mean(err[mask] ** 2)) if np.any(mask) else 0.0,
        "overestimate_mean": float(np.mean(np.maximum(err[mask], 0.0))) if np.any(mask) else 0.0,
        "underestimate_mean": float(np.mean(np.maximum(-err[mask], 0.0))) if np.any(mask) else 0.0,
        "distance_mse": float(np.mean((d[mask] - d_star[mask]) ** 2)) if np.any(mask) else 0.0,
        "long_horizon_mse": float(np.mean(err[long_mask] ** 2)) if np.any(long_mask) else 0.0,
        "triangle_violation_rate": triangle_violation_rate(q, gamma, samples=triangle_samples)
        if triangle_samples > 0
        else float("nan"),
    }
