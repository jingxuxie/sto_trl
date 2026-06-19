from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .envs import TabularMDP


@dataclass(frozen=True)
class Trajectory:
    states: np.ndarray
    actions: np.ndarray


def rollout_policy(
    mdp: TabularMDP,
    policy,
    rng: np.random.Generator,
    start_state: int | None = None,
    max_steps: int | None = None,
) -> Trajectory:
    state = mdp.start_state if start_state is None else int(start_state)
    horizon = mdp.max_steps if max_steps is None else int(max_steps)
    states = [state]
    actions = []
    for t in range(horizon):
        action = int(policy(state, t))
        next_state = mdp.sample_next(state, action, rng)
        actions.append(action)
        states.append(next_state)
        state = next_state
        if state == mdp.goal_state:
            break
    return Trajectory(np.asarray(states, dtype=np.int64), np.asarray(actions, dtype=np.int64))


def generate_chain_shortest_path_data(
    mdp: TabularMDP,
    repeats: int = 3,
    noise: float = 0.05,
    seed: int = 0,
) -> list[Trajectory]:
    rng = np.random.default_rng(seed)
    trajectories: list[Trajectory] = []
    for _ in range(repeats):
        for start in range(mdp.n_states):
            for target in range(mdp.n_states):
                if start == target:
                    continue

                def policy(state: int, _t: int, target: int = target) -> int:
                    if rng.random() < noise:
                        return int(rng.integers(mdp.n_actions))
                    return 1 if state < target else 0

                traj = rollout_policy(
                    mdp,
                    policy,
                    rng,
                    start_state=start,
                    max_steps=2 * mdp.n_states,
                )
                trajectories.append(traj)
    return trajectories


def generate_chain_start_goal_data(
    mdp: TabularMDP,
    repeats: int = 64,
    noise: float = 0.0,
    seed: int = 0,
) -> list[Trajectory]:
    rng = np.random.default_rng(seed)
    trajectories: list[Trajectory] = []
    target = mdp.goal_state
    for _ in range(repeats):

        def policy(state: int, _t: int) -> int:
            if rng.random() < noise:
                return int(rng.integers(mdp.n_actions))
            return 1 if state < target else 0

        trajectories.append(
            rollout_policy(
                mdp,
                policy,
                rng,
                start_state=mdp.start_state,
                max_steps=2 * mdp.n_states,
            )
        )
    return trajectories


def generate_risky_data(
    mdp: TabularMDP,
    n_trajectories: int = 500,
    p_choose_risky: float = 0.5,
    action_noise: float = 0.02,
    seed: int = 0,
) -> list[Trajectory]:
    rng = np.random.default_rng(seed)
    trajectories: list[Trajectory] = []

    for _ in range(n_trajectories):
        chose_risky = rng.random() < p_choose_risky
        state = mdp.start_state
        states = [state]
        actions = []
        for _t in range(mdp.max_steps):
            if rng.random() < action_noise:
                action = int(rng.integers(mdp.n_actions))
            elif state == mdp.start_state:
                action = 1 if chose_risky else 0
            else:
                action = 0
            next_state = mdp.sample_next(state, action, rng)
            actions.append(action)
            states.append(next_state)
            state = next_state
            if state == mdp.goal_state or mdp.state_names[state].startswith("trap_"):
                break
        trajectories.append(
            Trajectory(np.asarray(states, dtype=np.int64), np.asarray(actions, dtype=np.int64))
        )
    return trajectories


def _forward_actions_to_goal(mdp: TabularMDP) -> np.ndarray:
    """Action taking state i to i+1 for corridor-like MDPs."""
    actions = np.zeros(mdp.n_states, dtype=np.int64)
    for state in range(mdp.goal_state):
        for action in range(mdp.n_actions):
            probs = mdp.transitions[state, action]
            if int(np.argmax(probs)) == state + 1 and probs[state + 1] >= 1.0 - 1e-12:
                actions[state] = action
                break
    return actions


def generate_grid_shortcut_data(
    mdp: TabularMDP,
    n_trajectories: int = 1_000,
    p_choose_portal: float = 0.5,
    action_noise: float = 0.02,
    seed: int = 0,
) -> list[Trajectory]:
    rng = np.random.default_rng(seed)
    trajectories: list[Trajectory] = []
    forward_actions = _forward_actions_to_goal(mdp)
    portal_action = mdp.n_actions - 1

    for _ in range(n_trajectories):
        chose_portal = rng.random() < p_choose_portal
        state = mdp.start_state
        states = [state]
        actions = []
        for _t in range(mdp.max_steps):
            if rng.random() < action_noise:
                action = int(rng.integers(mdp.n_actions))
            elif state == mdp.start_state and chose_portal:
                action = portal_action
            else:
                action = int(forward_actions[state])
            next_state = mdp.sample_next(state, action, rng)
            actions.append(action)
            states.append(next_state)
            state = next_state
            if state == mdp.goal_state or mdp.state_names[state].startswith("trap_"):
                break
        trajectories.append(
            Trajectory(np.asarray(states, dtype=np.int64), np.asarray(actions, dtype=np.int64))
        )
    return trajectories


def transition_counts(
    trajectories: list[Trajectory], n_states: int, n_actions: int
) -> np.ndarray:
    counts = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    for traj in trajectories:
        for s, a, sp in zip(traj.states[:-1], traj.actions, traj.states[1:]):
            counts[int(s), int(a), int(sp)] += 1.0
    return counts


def estimate_transitions(
    trajectories: list[Trajectory],
    n_states: int,
    n_actions: int,
    fallback_self_loop: bool = True,
) -> np.ndarray:
    counts = transition_counts(trajectories, n_states, n_actions)
    totals = counts.sum(axis=-1, keepdims=True)
    probs = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    if fallback_self_loop:
        unseen = totals[..., 0] == 0
        for s in range(n_states):
            for a in range(n_actions):
                if unseen[s, a]:
                    probs[s, a, s] = 1.0
    return probs


def coverage_summary(
    trajectories: list[Trajectory], n_states: int, n_actions: int, goal_state: int
) -> dict[str, float]:
    counts = transition_counts(trajectories, n_states, n_actions)
    state_counts = np.zeros(n_states, dtype=np.float64)
    reached_goal = 0
    lengths = []
    for traj in trajectories:
        np.add.at(state_counts, traj.states, 1)
        reached_goal += int(np.any(traj.states == goal_state))
        lengths.append(len(traj.actions))
    return {
        "n_trajectories": float(len(trajectories)),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "goal_reach_frac": float(reached_goal / max(len(trajectories), 1)),
        "covered_state_frac": float(np.mean(state_counts > 0)),
        "covered_sa_frac": float(np.mean(counts.sum(axis=-1) > 0)),
    }
