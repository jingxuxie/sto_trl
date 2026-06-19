from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TabularMDP:
    name: str
    transitions: np.ndarray  # [state, action, next_state]
    start_state: int
    goal_state: int
    max_steps: int
    action_names: tuple[str, ...]
    state_names: tuple[str, ...]

    @property
    def n_states(self) -> int:
        return int(self.transitions.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.transitions.shape[1])

    def sample_next(self, state: int, action: int, rng: np.random.Generator) -> int:
        return int(rng.choice(self.n_states, p=self.transitions[state, action]))


def _self_loop(n_states: int, n_actions: int) -> np.ndarray:
    transitions = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    for s in range(n_states):
        transitions[s, :, s] = 1.0
    return transitions


def deterministic_chain(n_states: int = 32) -> TabularMDP:
    if n_states < 3:
        raise ValueError("n_states must be at least 3")
    transitions = _self_loop(n_states, 2)
    for s in range(n_states):
        transitions[s, 0, :] = 0.0
        transitions[s, 0, max(s - 1, 0)] = 1.0
        transitions[s, 1, :] = 0.0
        transitions[s, 1, min(s + 1, n_states - 1)] = 1.0
    names = tuple(f"x{s}" for s in range(n_states))
    return TabularMDP(
        name=f"det_chain_n{n_states}",
        transitions=transitions,
        start_state=0,
        goal_state=n_states - 1,
        max_steps=2 * n_states,
        action_names=("left", "right"),
        state_names=names,
    )


def risky_shortcut(
    safe_length: int = 16,
    p_success: float = 0.2,
    trap_escape_length: int | None = None,
) -> TabularMDP:
    """Two-route MDP.

    From start, action 0 enters a deterministic safe chain of `safe_length`
    steps to the goal. Action 1 enters a risky two-step shortcut: the next
    action succeeds with probability p_success and otherwise enters a trap.
    """
    if safe_length < 2:
        raise ValueError("safe_length must be at least 2")
    if not 0.0 <= p_success <= 1.0:
        raise ValueError("p_success must be in [0, 1]")

    start = 0
    safe_states = list(range(1, safe_length))
    risk_state = safe_length

    if trap_escape_length is None:
        trap_states = [safe_length + 1]
    else:
        if trap_escape_length < 2:
            raise ValueError("trap_escape_length must be >= 2 or None")
        trap_states = list(range(safe_length + 1, safe_length + trap_escape_length))
    goal = trap_states[-1] + 1
    n_states = goal + 1
    transitions = _self_loop(n_states, 2)

    # Start: safe route or risky route.
    transitions[start, 0, :] = 0.0
    transitions[start, 0, safe_states[0]] = 1.0
    transitions[start, 1, :] = 0.0
    transitions[start, 1, risk_state] = 1.0

    # Safe chain. Action 0 progresses; action 1 stalls to make policy errors visible.
    for idx, state in enumerate(safe_states):
        next_state = safe_states[idx + 1] if idx + 1 < len(safe_states) else goal
        transitions[state, 0, :] = 0.0
        transitions[state, 0, next_state] = 1.0
        transitions[state, 1, :] = 0.0
        transitions[state, 1, state] = 1.0

    # Risk branch. Action 0 exposes the stochastic outcome; action 1 stalls.
    transitions[risk_state, 0, :] = 0.0
    transitions[risk_state, 0, goal] = p_success
    transitions[risk_state, 0, trap_states[0]] = 1.0 - p_success
    transitions[risk_state, 1, :] = 0.0
    transitions[risk_state, 1, risk_state] = 1.0

    # Trap is absorbing by default, or a long escape chain if requested.
    if trap_escape_length is not None:
        for idx, state in enumerate(trap_states):
            next_state = trap_states[idx + 1] if idx + 1 < len(trap_states) else goal
            transitions[state, 0, :] = 0.0
            transitions[state, 0, next_state] = 1.0
            transitions[state, 1, :] = 0.0
            transitions[state, 1, state] = 1.0

    transitions[goal, :, :] = 0.0
    transitions[goal, :, goal] = 1.0

    names = ["start"]
    names.extend(f"safe_{i}" for i in range(1, safe_length))
    names.append("risk")
    names.extend(f"trap_{i}" for i in range(len(trap_states)))
    names.append("goal")
    max_steps = max(4 * safe_length, safe_length + (trap_escape_length or 0) + 8)
    return TabularMDP(
        name=f"risky_L{safe_length}_p{p_success:.2f}",
        transitions=transitions,
        start_state=start,
        goal_state=goal,
        max_steps=max_steps,
        action_names=("safe/forward", "risky/stall"),
        state_names=tuple(names),
    )


def stochastic_grid_shortcut(
    width: int = 16,
    height: int = 8,
    p_success: float = 0.05,
) -> TabularMDP:
    """2D snake-corridor goal task with a stochastic risky portal.

    The safe route is a deterministic Hamiltonian snake path through a 2D grid.
    Compass actions only connect consecutive cells on that path, so the task is
    a long-horizon corridor with turns rather than a shortcut-rich open grid.
    At the start, the portal action reaches the goal with probability
    `p_success` and an absorbing trap otherwise.
    """
    if width < 2 or height < 2:
        raise ValueError("width and height must both be at least 2")
    if not 0.0 <= p_success <= 1.0:
        raise ValueError("p_success must be in [0, 1]")

    path: list[tuple[int, int]] = []
    for y in range(height):
        xs = range(width) if y % 2 == 0 else range(width - 1, -1, -1)
        for x in xs:
            path.append((x, y))

    n_path = len(path)
    trap = n_path
    n_states = n_path + 1
    n_actions = 5
    start = 0
    goal = n_path - 1
    portal = 4
    transitions = _self_loop(n_states, n_actions)

    action_for_delta = {
        (0, -1): 0,  # north
        (1, 0): 1,   # east
        (0, 1): 2,   # south
        (-1, 0): 3,  # west
    }

    for i, (x, y) in enumerate(path):
        for j in (i - 1, i + 1):
            if j < 0 or j >= n_path:
                continue
            nx, ny = path[j]
            action = action_for_delta[(nx - x, ny - y)]
            transitions[i, action, :] = 0.0
            transitions[i, action, j] = 1.0

    transitions[start, portal, :] = 0.0
    transitions[start, portal, goal] = p_success
    transitions[start, portal, trap] = 1.0 - p_success

    transitions[goal, :, :] = 0.0
    transitions[goal, :, goal] = 1.0
    transitions[trap, :, :] = 0.0
    transitions[trap, :, trap] = 1.0

    names = [f"cell_{i}_x{x}_y{y}" for i, (x, y) in enumerate(path)]
    names.append("trap_portal_fail")
    return TabularMDP(
        name=f"grid_shortcut_{width}x{height}_p{p_success:.2f}",
        transitions=transitions,
        start_state=start,
        goal_state=goal,
        max_steps=2 * n_path,
        action_names=("north", "east", "south", "west", "portal"),
        state_names=tuple(names),
    )
