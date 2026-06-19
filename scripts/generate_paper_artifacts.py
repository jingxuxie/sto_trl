#!/usr/bin/env python3
"""Generate paper-facing tables and simple SVG figures from current results."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLE_DIR = RESULTS / "paper_tables"
FIG_DIR = RESULTS / "figures"


TABULAR_L128 = RESULTS / "tabular_L128_fast.csv"
TABULAR_COMPACT = RESULTS / "fast_tabular_compact.csv"
TABULAR_SAFE_HORIZON = RESULTS / "tabular_safe_horizon_L16_128_5seed.csv"
GRID_SHORTCUT = RESULTS / "grid_shortcut_2d_5seed.csv"
GRID_REALIZED_DIAGNOSTIC = RESULTS / "grid_shortcut_realized_diagnostic.csv"
GRID_BUDGET_CURVE = RESULTS / "grid_budget_curve_16x8_p005_5seed.csv"
POINTMAZE_TOPOLOGY = RESULTS / "pointmaze_topology_dataset_5seed_ep50.csv"
POINTMAZE_TOPOLOGY_STITCH = RESULTS / "pointmaze_topology_stitch_5seed_ep50.csv"
POINTMAZE_TOPOLOGY_STITCH_SUPPORT = RESULTS / "pointmaze_topology_stitch_support_baseline_5seed_ep50.csv"
POINTMAZE_STITCH_TASK5 = RESULTS / "pointmaze_stitch_task5_ep100_seed01234.csv"
POINTMAZE_LEARNED_TRANSITION_FILES = [
    {
        "env": "pointmaze-teleport-navigate-v0",
        "transition_model": "dataset cell-change softmax",
        "paths": [RESULTS / "pointmaze_learned_transition_navigate_cellchanges_1k_ep50_seed0.csv"],
    },
    {
        "env": "pointmaze-teleport-stitch-v0",
        "transition_model": "dataset cell-change softmax",
        "paths": [RESULTS / "pointmaze_learned_transition_stitch_cellchanges_1k_ep50_seed0.csv"],
    },
    {
        "env": "pointmaze-teleport-navigate-v0",
        "transition_model": "raw-observation MLP cell-change",
        "paths": [
            RESULTS / "pointmaze_navigate_rawobs_mlp_cellchanges_ep50_seed0.csv",
            RESULTS / "pointmaze_navigate_rawobs_mlp_cellchanges_ep50_tseed1.csv",
            RESULTS / "pointmaze_navigate_rawobs_mlp_cellchanges_ep50_tseed2.csv",
        ],
    },
    {
        "env": "pointmaze-teleport-stitch-v0",
        "transition_model": "raw-observation MLP cell-change",
        "paths": [
            RESULTS / "pointmaze_stitch_rawobs_mlp_cellchanges_ep50_seed0.csv",
            RESULTS / "pointmaze_stitch_rawobs_mlp_cellchanges_ep50_tseed1.csv",
            RESULTS / "pointmaze_stitch_rawobs_mlp_cellchanges_ep50_tseed2.csv",
        ],
    },
]
POINTMAZE_BC_CONTROLLER_FILES = [
    {
        "env": "pointmaze-teleport-navigate-v0",
        "controller": "5k full-goal BC + body-nearest k16",
        "paths": [
            RESULTS / "pointmaze_bc_topology_navigate_fullgoal_5k_ep20_seed0.csv",
            RESULTS / "pointmaze_bc_topology_navigate_fullgoal_5k_ep20_seed12.csv",
        ],
    },
    {
        "env": "pointmaze-teleport-stitch-v0",
        "controller": "5k full-goal BC + body-nearest k16",
        "paths": [
            RESULTS / "pointmaze_bc_topology_stitch_fullgoal_5k_ep20_seed0.csv",
            RESULTS / "pointmaze_bc_topology_stitch_fullgoal_5k_ep20_seed12.csv",
        ],
    },
]
POINTMAZE_DIRECT_ACTOR_FILES = [
    {
        "execution_path": "GCFBC direct final-goal actor",
        "env": "pointmaze-teleport-navigate-v0",
        "controller_or_actor": "GCFBC 128,128",
        "eval_setting": "seed 0, 5 episodes/task",
        "path": RESULTS
        / "ogbench_screen_exp"
        / "dummy"
        / "pointmaze_gcfbc_10k_actor_diag_cpu"
        / "sd000_20260619_104221"
        / "eval.csv",
    },
    {
        "execution_path": "MSEBC direct final-goal actor",
        "env": "pointmaze-teleport-navigate-v0",
        "controller_or_actor": "MSEBC LN 256,256",
        "eval_setting": "seed 0, 5 episodes/task",
        "path": RESULTS
        / "ogbench_screen_exp"
        / "dummy"
        / "pointmaze_msebc_10k_ln256_actor_diag_cpu"
        / "sd000_20260619_104714"
        / "eval.csv",
    },
]
FAST_EVAL_PROFILE_FILES = [
    {
        "screen": "antmaze navigate hard slice",
        "role": "recommended fast screen",
        "path": RESULTS / "antmaze_navigate_fast_profile_repeat1_ep5_seed0_task45.csv",
    },
    {
        "screen": "antmaze stitch hard slice",
        "role": "recommended fast screen",
        "path": RESULTS / "antmaze_stitch_fast_profile_repeat1_ep5_seed0_task45.csv",
    },
    {
        "screen": "antmaze stitch hard slice",
        "role": "two-episode baseline",
        "path": RESULTS / "antmaze_stitch_fast_profile_repeat1_ep2_seed0_task45.csv",
    },
    {
        "screen": "antmaze stitch hard slice",
        "role": "action-repeat ablation",
        "path": RESULTS / "antmaze_stitch_fast_profile_repeat2_ep2_seed0_task45.csv",
    },
]
ANTMAZE_SUPPORT_ABLATION_FILES = [
    {
        "env": "antmaze-teleport-navigate-v0",
        "controller": "50k full-goal BC + body-nearest k16",
        "path": RESULTS / "antmaze_navigate_support_ablation_ep5_seed0_task45.csv",
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "controller": "20k full-goal BC + body-nearest k16",
        "path": RESULTS / "antmaze_stitch_support_ablation_ep5_seed0_task45.csv",
    },
]
POINTMAZE_TIE_POLICY_HEAD_FILES = [
    RESULTS / "pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv",
    RESULTS / "pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv",
]
POINTMAZE_PREV_POLICY_HEAD_FILES = [
    RESULTS / "pointmaze_navigate_rawobs_mlp_transition_prev_policy_ep20_seed0.csv",
    RESULTS / "pointmaze_stitch_rawobs_mlp_transition_prev_policy_ep20_seed0.csv",
]
POINTMAZE_TIE_POLICY_HEAD_EVAL_SEED_FILES = [
    {
        "env": "pointmaze-teleport-navigate-v0",
        "paths": [
            RESULTS / "pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv",
            RESULTS / "pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_evalseed12.csv",
        ],
    },
    {
        "env": "pointmaze-teleport-stitch-v0",
        "paths": [
            RESULTS / "pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv",
            RESULTS / "pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_evalseed12.csv",
        ],
    },
]
POINTMAZE_TIE_POLICY_HEAD_TRANSITION_SEED_FILES = [
    {
        "env": "pointmaze-teleport-navigate-v0",
        "paths": [
            RESULTS / "pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv",
            RESULTS / "pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_tseed1.csv",
            RESULTS / "pointmaze_navigate_rawobs_mlp_transition_tie_policy_methods_ep20_tseed2.csv",
        ],
    },
    {
        "env": "pointmaze-teleport-stitch-v0",
        "paths": [
            RESULTS / "pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_seed0.csv",
            RESULTS / "pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_tseed1.csv",
            RESULTS / "pointmaze_stitch_rawobs_mlp_transition_tie_policy_methods_ep20_tseed2.csv",
        ],
    },
]
ANTMAZE_TIE_POLICY_HEAD_FILES = [
    {
        "env": "antmaze-teleport-navigate-v0",
        "controller": "50k full-goal BC + body-nearest k16",
        "path": RESULTS / "antmaze_navigate_tie_policy_head_hard_tasks_ep20_seed0.csv",
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "controller": "20k full-goal BC + body-nearest k16",
        "path": RESULTS / "antmaze_stitch_tie_policy_head_hard_tasks_ep20_seed0.csv",
    },
]
ANTMAZE_RAWOBS_TIE_POLICY_HEAD_FILES = [
    {
        "env": "antmaze-teleport-navigate-v0",
        "controller": "50k full-goal BC + body-nearest k16",
        "paths": [
            RESULTS / "antmaze_navigate_rawobs_transition_tie_policy_head_ep10_seed0.csv",
            RESULTS / "antmaze_navigate_rawobs_transition_tie_policy_head_ep10_tseed1.csv",
            RESULTS / "antmaze_navigate_rawobs_transition_tie_policy_head_ep10_tseed2.csv",
        ],
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "controller": "20k full-goal BC + body-nearest k16",
        "paths": [
            RESULTS / "antmaze_stitch_rawobs_transition_tie_policy_head_ep10_seed0.csv",
            RESULTS / "antmaze_stitch_rawobs_transition_tie_policy_head_ep10_tseed1.csv",
            RESULTS / "antmaze_stitch_rawobs_transition_tie_policy_head_ep10_tseed2.csv",
        ],
    },
]
ANTMAZE_RAWOBS_TIE_POLICY_EVAL_SEED_FILES = [
    {
        "env": "antmaze-teleport-navigate-v0",
        "controller": "50k full-goal BC + body-nearest k16",
        "paths": [
            RESULTS / "antmaze_navigate_rawobs_transition_tie_policy_head_ep10_seed0.csv",
            RESULTS / "antmaze_navigate_rawobs_transition_tie_policy_head_ep10_tseed0_evalseed12.csv",
        ],
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "controller": "20k full-goal BC + body-nearest k16",
        "paths": [
            RESULTS / "antmaze_stitch_rawobs_transition_tie_policy_head_ep10_seed0.csv",
            RESULTS / "antmaze_stitch_rawobs_transition_tie_policy_head_ep10_tseed0_evalseed12.csv",
        ],
    },
]
ANTMAZE_RAWOBS_PREV_POLICY_EVAL_SEED_FILES = [
    {
        "env": "antmaze-teleport-navigate-v0",
        "controller": "50k full-goal BC + body-nearest k16",
        "paths": [
            RESULTS / "antmaze_navigate_rawobs_transition_prev_policy_head_ep10_seed012.csv",
        ],
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "controller": "20k full-goal BC + body-nearest k16",
        "paths": [
            RESULTS / "antmaze_stitch_rawobs_transition_prev_policy_head_ep10_seed012.csv",
        ],
    },
]
ANTMAZE_BC_TOPOLOGY = RESULTS / "antmaze_bc_topology_fullgoal_20k_ep3_seed0.csv"
ANTMAZE_NAVIGATE_BODYK16_FILES = [
    RESULTS / "antmaze_bc_topology_navigate_fullgoal_50k_ep10_seed0_bodyk16_matched.csv",
    RESULTS / "antmaze_bc_topology_navigate_fullgoal_50k_ep10_seed0_bodyk16_sto_full.csv",
]
ANTMAZE_BODYK16_MULTI_FILES = [
    RESULTS / "antmaze_bc_topology_navigate_fullgoal_50k_ep5_seed012_bodyk16.csv",
    RESULTS / "antmaze_bc_topology_stitch_fullgoal_20k_ep5_seed012_bodyk16.csv",
]
ANTMAZE_BCSEED1_FILES = [
    RESULTS / "antmaze_bc_topology_navigate_fullgoal_50k_bcseed1_ep20_seed012_bodyk16_cpu.csv",
    RESULTS / "antmaze_bc_topology_stitch_fullgoal_20k_bcseed1_ep20_seed012_bodyk16_cpu.csv",
]
ANTMAZE_STITCH_CONTROLLER_SEED_FILES = [
    ("0", RESULTS / "antmaze_bc_topology_stitch_fullgoal_20k_ep20_seed012_bodyk16_cpu.csv"),
    ("1", RESULTS / "antmaze_bc_topology_stitch_fullgoal_20k_bcseed1_ep20_seed012_bodyk16_cpu.csv"),
    ("2", RESULTS / "antmaze_bc_topology_stitch_fullgoal_20k_bcseed2_ep20_seed012_bodyk16_cpu.csv"),
]
ANTMAZE_NAVIGATE_CONTROLLER_SEED_FILES = [
    ("0", RESULTS / "antmaze_bc_topology_navigate_fullgoal_50k_ep20_seed012_bodyk16_cpu.csv"),
    ("1", RESULTS / "antmaze_bc_topology_navigate_fullgoal_50k_bcseed1_ep20_seed012_bodyk16_cpu.csv"),
    ("2", RESULTS / "antmaze_bc_topology_navigate_fullgoal_50k_bcseed2_ep20_seed012_bodyk16_cpu.csv"),
]
ANTMAZE_BODYK16_EP20_FILES = [
    RESULTS / "antmaze_bc_topology_navigate_fullgoal_50k_ep20_seed012_bodyk16_cpu.csv",
    RESULTS / "antmaze_bc_topology_stitch_fullgoal_20k_ep20_seed012_bodyk16_cpu.csv",
]
ANTMAZE_BUDGET_FILES = [
    RESULTS / "antmaze_bc_topology_navigate_budget_ep3_seed0_bodyk16.csv",
    RESULTS / "antmaze_bc_topology_stitch_budget_ep3_seed0_bodyk16.csv",
]
ANTMAZE_STITCH_EXECUTOR_ABLATION_FILES = [
    ("nearest_xy", RESULTS / "antmaze_bc_topology_stitch_fullgoal_20k_ep3_seed0_nearestxy_sto_full.csv"),
    ("body_nearest_k16", RESULTS / "antmaze_bc_topology_stitch_fullgoal_20k_ep3_seed0_bodyk16_sto_full.csv"),
]
ANTMAZE_LEARNED_TRANSITION_FILES = [
    {
        "env": "antmaze-teleport-navigate-v0",
        "controller": "50k BC",
        "transition_model": "table softmax, 20 samples/row",
        "paths": [
            RESULTS / "antmaze_navigate_learned_transition_samples20_ep10_tseed0.csv",
            RESULTS / "antmaze_navigate_learned_transition_samples20_ep10_tseed1.csv",
            RESULTS / "antmaze_navigate_learned_transition_samples20_ep10_tseed2.csv",
        ],
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "controller": "20k BC",
        "transition_model": "table softmax, 20 samples/row",
        "paths": [
            RESULTS / "antmaze_stitch_learned_transition_samples20_ep10_tseed0.csv",
            RESULTS / "antmaze_stitch_learned_transition_samples20_ep10_tseed1.csv",
            RESULTS / "antmaze_stitch_learned_transition_samples20_ep10_tseed2.csv",
        ],
    },
    {
        "env": "antmaze-teleport-navigate-v0",
        "controller": "50k BC",
        "transition_model": "raw-observation MLP jump-change",
        "paths": [
            RESULTS / "antmaze_navigate_rawobs_mlp_jumpchanges_ep10_seed0.csv",
            RESULTS / "antmaze_navigate_rawobs_mlp_jumpchanges_ep10_tseed1.csv",
            RESULTS / "antmaze_navigate_rawobs_mlp_jumpchanges_ep10_tseed2.csv",
        ],
    },
    {
        "env": "antmaze-teleport-stitch-v0",
        "controller": "20k BC",
        "transition_model": "raw-observation MLP jump-change",
        "paths": [
            RESULTS / "antmaze_stitch_rawobs_mlp_jumpchanges_ep10_seed0.csv",
            RESULTS / "antmaze_stitch_rawobs_mlp_jumpchanges_ep10_tseed1.csv",
            RESULTS / "antmaze_stitch_rawobs_mlp_jumpchanges_ep10_tseed2.csv",
        ],
    },
]
TELEPORT_STITCH_SCREEN = TABLE_DIR / "teleport_stitch_screen_seed0.csv"
GRAPH_MAIN_FILES = [
    RESULTS / "pointmaze_graph_persistent_waypoint_matched_independent_seeded_paired50.csv",
]
HARD_TASK_STRESS_SPECS = [
    (
        "pointmaze-teleport-stitch-v0",
        "tasks 4,5",
        "seed 0, 50 episodes/task",
        "0",
        RESULTS / "pointmaze_stitch_hard_task45_ep50_seed0_fastfocus.csv",
    ),
    (
        "antmaze-teleport-navigate-v0",
        "tasks 4,5",
        "seed 0, 10 episodes/task",
        "50000",
        RESULTS / "antmaze_navigate_hard_task45_ep10_seed0_fastfocus.csv",
    ),
    (
        "antmaze-teleport-stitch-v0",
        "tasks 4,5",
        "seed 0, 10 episodes/task",
        "20000",
        RESULTS / "antmaze_stitch_hard_task45_ep10_seed0_fastfocus.csv",
    ),
]


METHOD_ORDER = [
    "mc_positive",
    "mc_all_goals",
    "trl_raw_realized",
    "trl_log_realized",
    "bellman_matched",
    "support_trl_matched",
    "sto_trl_matched",
    "bellman_full",
]
METHOD_LABELS = {
    "mc_positive": "MC positive",
    "mc_all_goals": "MC all goals",
    "trl_raw_realized": "Realized TRL",
    "trl_log_realized": "Log realized TRL",
    "bellman_matched": "Bellman matched",
    "support_trl_matched": "Support TRL",
    "sto_trl_matched": "Stochastic TRL",
    "bellman_full": "Bellman full",
    "bellman_polished": "Bellman polished",
    "sto_trl_polished": "Stochastic TRL polished",
    "bellman": "Bellman",
    "sto_trl": "Stochastic TRL",
}
COLORS = {
    "mc_positive": "#54A24B",
    "mc_all_goals": "#B279A2",
    "bellman": "#4C78A8",
    "bellman_matched": "#4C78A8",
    "bellman_polished": "#4C78A8",
    "sto_trl": "#F58518",
    "sto_trl_matched": "#F58518",
    "sto_trl_polished": "#F58518",
    "bellman_full": "#72B7B2",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() == "inf":
        return None
    return float(value)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def pstdev(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) <= 1:
        return 0.0
    mu = mean(vals)
    return math.sqrt(sum((x - mu) ** 2 for x in vals) / len(vals))


def sample_stdev(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) <= 1:
        return 0.0
    mu = mean(vals)
    return math.sqrt(sum((x - mu) ** 2 for x in vals) / (len(vals) - 1))


def t_critical_95(n: int) -> float:
    # Two-sided 95% critical values for small seed counts used here.
    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }
    return table.get(n, 1.96)


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def method_family(method: str) -> str:
    if method.startswith("bellman_") and method.endswith("_sweeps"):
        return "bellman_matched"
    if method.startswith("sto_trl_") and method.endswith("_sweeps"):
        return "sto_trl_matched"
    return method


def method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)


def action_name(value: float | None) -> str:
    if value is None:
        return ""
    return "risky" if value >= 0.5 else "safe"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row.get(name, "") for name in fieldnames) + " |")
    return "\n".join(lines)


@dataclass
class Bar:
    label: str
    value: float
    color: str
    stderr: float = 0.0


def label_lines(label: str) -> list[str]:
    if "\n" in label:
        return label.split("\n")
    words = label.split()
    if len(words) <= 1:
        return [label]
    if len(label) <= 14:
        return [label]
    mid = len(words) // 2
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def svg_text(x: float, y: float, label: str, size: int, anchor: str = "middle") -> str:
    lines = label_lines(label)
    if len(lines) == 1:
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-size="{size}" fill="#222">{escape(lines[0])}</text>'
        )
    tspans = []
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else size + 2
        tspans.append(f'<tspan x="{x:.1f}" dy="{dy}">{escape(line)}</tspan>')
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size}" fill="#222">' + "".join(tspans) + "</text>"
    )


def write_bar_svg(path: Path, title: str, bars: list[Bar], ylabel: str = "Success rate") -> None:
    width, height = 760, 420
    left, right, top, bottom = 72, 24, 54, 92
    chart_w = width - left - right
    chart_h = height - top - bottom
    ymax = max(1.0, max((b.value + b.stderr for b in bars), default=1.0))
    ymax = min(1.0, ymax) if ymax <= 1.0 else ymax * 1.08

    def y(value: float) -> float:
        return top + chart_h - (value / ymax) * chart_h

    step = chart_w / max(1, len(bars))
    bar_w = min(78, step * 0.58)
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, title, 18),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{width - right}" y2="{top + chart_h}" stroke="#333" stroke-width="1"/>',
    ]
    for i in range(6):
        val = i / 5
        yy = y(val)
        elems.append(f'<line x1="{left - 5}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="#e5e5e5" stroke-width="1"/>')
        elems.append(f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end" font-size="12" fill="#333">{val:.1f}</text>')
    elems.append(
        f'<text x="18" y="{top + chart_h / 2:.1f}" transform="rotate(-90 18 {top + chart_h / 2:.1f})" '
        f'text-anchor="middle" font-size="13" fill="#333">{escape(ylabel)}</text>'
    )
    for i, bar in enumerate(bars):
        cx = left + step * (i + 0.5)
        x0 = cx - bar_w / 2
        y0 = y(bar.value)
        h = top + chart_h - y0
        elems.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{bar.color}"/>')
        if bar.stderr > 0:
            ey0 = y(min(ymax, bar.value + bar.stderr))
            ey1 = y(max(0.0, bar.value - bar.stderr))
            elems.append(f'<line x1="{cx:.1f}" y1="{ey0:.1f}" x2="{cx:.1f}" y2="{ey1:.1f}" stroke="#222" stroke-width="1.5"/>')
            elems.append(f'<line x1="{cx - 8:.1f}" y1="{ey0:.1f}" x2="{cx + 8:.1f}" y2="{ey0:.1f}" stroke="#222" stroke-width="1.5"/>')
            elems.append(f'<line x1="{cx - 8:.1f}" y1="{ey1:.1f}" x2="{cx + 8:.1f}" y2="{ey1:.1f}" stroke="#222" stroke-width="1.5"/>')
        elems.append(svg_text(cx, y0 - 8, fmt(bar.value), 12))
        elems.append(svg_text(cx, top + chart_h + 22, bar.label, 12))
    elems.append("</svg>")
    path.write_text("\n".join(elems) + "\n")


def write_grouped_svg(
    path: Path,
    title: str,
    groups: list[str],
    series: list[str],
    values: dict[tuple[str, str], float],
) -> None:
    width, height = 820, 430
    left, right, top, bottom = 72, 28, 54, 108
    chart_w = width - left - right
    chart_h = height - top - bottom

    def y(value: float) -> float:
        return top + chart_h - value * chart_h

    group_w = chart_w / max(1, len(groups))
    bar_w = min(56, group_w / (len(series) + 1.5))
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, title, 18),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{width - right}" y2="{top + chart_h}" stroke="#333" stroke-width="1"/>',
    ]
    for i in range(6):
        val = i / 5
        yy = y(val)
        elems.append(f'<line x1="{left - 5}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="#e5e5e5" stroke-width="1"/>')
        elems.append(f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end" font-size="12" fill="#333">{val:.1f}</text>')
    elems.append(
        f'<text x="18" y="{top + chart_h / 2:.1f}" transform="rotate(-90 18 {top + chart_h / 2:.1f})" '
        'text-anchor="middle" font-size="13" fill="#333">Success rate</text>'
    )
    for gi, group in enumerate(groups):
        center = left + group_w * (gi + 0.5)
        start = center - (len(series) * bar_w) / 2
        for si, method in enumerate(series):
            value = values[(group, method)]
            x0 = start + si * bar_w
            y0 = y(value)
            elems.append(
                f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bar_w * 0.82:.1f}" '
                f'height="{top + chart_h - y0:.1f}" fill="{COLORS.get(method, "#999")}"/>'
            )
            elems.append(svg_text(x0 + bar_w * 0.41, y0 - 8, fmt(value), 11))
        elems.append(svg_text(center, top + chart_h + 24, group, 13))

    legend_x = left
    legend_y = height - 48
    for i, method in enumerate(series):
        x = legend_x + i * 220
        elems.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{COLORS.get(method, "#999")}"/>')
        elems.append(
            f'<text x="{x + 20}" y="{legend_y + 12}" font-size="13" fill="#222">'
            f'{escape(METHOD_LABELS.get(method, method))}</text>'
        )
    elems.append("</svg>")
    path.write_text("\n".join(elems) + "\n")


def write_horizon_svg(
    path: Path,
    title: str,
    table_rows: list[dict[str, str]],
    p_values: list[str],
    series: list[str],
) -> None:
    width, height = 940, 440
    left, right, top, bottom = 64, 24, 58, 86
    gap = 54
    panel_w = (width - left - right - gap) / 2
    chart_h = height - top - bottom
    horizons = sorted({int(row["safe_length"]) for row in table_rows})
    x_index = {horizon: i for i, horizon in enumerate(horizons)}
    data = {
        (row["p_success"], row["method_key"], int(row["safe_length"])): float(row["success_rate"])
        for row in table_rows
    }

    def panel_x(panel: int, horizon: int) -> float:
        x0 = left + panel * (panel_w + gap)
        if len(horizons) == 1:
            return x0 + panel_w / 2
        return x0 + x_index[horizon] * panel_w / (len(horizons) - 1)

    def y(value: float) -> float:
        return top + chart_h - value * chart_h

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 30, title, 18),
    ]
    for panel, p_success in enumerate(p_values):
        x0 = left + panel * (panel_w + gap)
        x1 = x0 + panel_w
        elems.append(svg_text((x0 + x1) / 2, 52, f"p={p_success}", 13))
        elems.append(f'<line x1="{x0:.1f}" y1="{top}" x2="{x0:.1f}" y2="{top + chart_h}" stroke="#333" stroke-width="1"/>')
        elems.append(f'<line x1="{x0:.1f}" y1="{top + chart_h}" x2="{x1:.1f}" y2="{top + chart_h}" stroke="#333" stroke-width="1"/>')
        for tick in range(6):
            val = tick / 5
            yy = y(val)
            elems.append(f'<line x1="{x0 - 4:.1f}" y1="{yy:.1f}" x2="{x1:.1f}" y2="{yy:.1f}" stroke="#e5e5e5" stroke-width="1"/>')
            if panel == 0:
                elems.append(f'<text x="{x0 - 10:.1f}" y="{yy + 4:.1f}" text-anchor="end" font-size="12" fill="#333">{val:.1f}</text>')
        for horizon in horizons:
            xx = panel_x(panel, horizon)
            elems.append(f'<text x="{xx:.1f}" y="{top + chart_h + 22:.1f}" text-anchor="middle" font-size="12" fill="#333">{horizon}</text>')
        elems.append(svg_text((x0 + x1) / 2, top + chart_h + 48, "Safe path length", 13))
        for method in series:
            pts = []
            for horizon in horizons:
                value = data.get((p_success, method, horizon))
                if value is not None:
                    pts.append((panel_x(panel, horizon), y(value), value))
            if len(pts) >= 2:
                point_str = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy, _ in pts)
                elems.append(f'<polyline points="{point_str}" fill="none" stroke="{COLORS.get(method, "#999")}" stroke-width="2.4"/>')
            for xx, yy, value in pts:
                elems.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{COLORS.get(method, "#999")}"/>')
                if value in {0.0, 1.0}:
                    elems.append(svg_text(xx, yy - 9, fmt(value), 10))
    elems.append(
        f'<text x="17" y="{top + chart_h / 2:.1f}" transform="rotate(-90 17 {top + chart_h / 2:.1f})" '
        'text-anchor="middle" font-size="13" fill="#333">Success rate</text>'
    )
    legend_y = height - 30
    legend_x = left
    for i, method in enumerate(series):
        x = legend_x + i * 190
        elems.append(f'<line x1="{x}" y1="{legend_y - 4}" x2="{x + 18}" y2="{legend_y - 4}" stroke="{COLORS.get(method, "#999")}" stroke-width="3"/>')
        elems.append(f'<circle cx="{x + 9}" cy="{legend_y - 4}" r="4" fill="{COLORS.get(method, "#999")}"/>')
        elems.append(
            f'<text x="{x + 24}" y="{legend_y}" font-size="13" fill="#222">'
            f'{escape(METHOD_LABELS.get(method, method))}</text>'
        )
    elems.append("</svg>")
    path.write_text("\n".join(elems) + "\n")


def write_budget_curve_svg(path: Path, title: str, table_rows: list[dict[str, str]]) -> None:
    width, height = 800, 420
    left, right, top, bottom = 72, 28, 58, 82
    chart_w = width - left - right
    chart_h = height - top - bottom
    budgets = sorted({int(row["sweeps"]) for row in table_rows})
    min_log = math.log2(min(budgets))
    max_log = math.log2(max(budgets))
    data = {
        (row["method_key"], int(row["sweeps"])): float(row["success_rate"])
        for row in table_rows
    }
    series = ["bellman", "sto_trl"]

    def x(sweeps: int) -> float:
        if max_log == min_log:
            return left + chart_w / 2
        return left + (math.log2(sweeps) - min_log) * chart_w / (max_log - min_log)

    def y(value: float) -> float:
        return top + chart_h - value * chart_h

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 30, title, 18),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{width - right}" y2="{top + chart_h}" stroke="#333" stroke-width="1"/>',
    ]
    for tick in range(6):
        val = tick / 5
        yy = y(val)
        elems.append(f'<line x1="{left - 5}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="#e5e5e5" stroke-width="1"/>')
        elems.append(f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end" font-size="12" fill="#333">{val:.1f}</text>')
    for budget in budgets:
        xx = x(budget)
        elems.append(f'<line x1="{xx:.1f}" y1="{top + chart_h}" x2="{xx:.1f}" y2="{top + chart_h + 5}" stroke="#333" stroke-width="1"/>')
        elems.append(f'<text x="{xx:.1f}" y="{top + chart_h + 22}" text-anchor="middle" font-size="12" fill="#333">{budget}</text>')
    elems.append(svg_text(left + chart_w / 2, top + chart_h + 50, "Planning sweeps", 13))
    elems.append(
        f'<text x="18" y="{top + chart_h / 2:.1f}" transform="rotate(-90 18 {top + chart_h / 2:.1f})" '
        'text-anchor="middle" font-size="13" fill="#333">Success rate</text>'
    )

    for method in series:
        pts = []
        for budget in budgets:
            value = data.get((method, budget))
            if value is not None:
                pts.append((x(budget), y(value), value))
        if len(pts) >= 2:
            point_str = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy, _ in pts)
            elems.append(f'<polyline points="{point_str}" fill="none" stroke="{COLORS.get(method, "#999")}" stroke-width="2.6"/>')
        for xx, yy, value in pts:
            elems.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{COLORS.get(method, "#999")}"/>')
            if value in {0.0, 1.0}:
                elems.append(svg_text(xx, yy - 9, fmt(value), 10))
    legend_y = height - 26
    legend_x = left
    for i, method in enumerate(series):
        xx = legend_x + i * 210
        elems.append(f'<line x1="{xx}" y1="{legend_y - 4}" x2="{xx + 18}" y2="{legend_y - 4}" stroke="{COLORS.get(method, "#999")}" stroke-width="3"/>')
        elems.append(f'<circle cx="{xx + 9}" cy="{legend_y - 4}" r="4" fill="{COLORS.get(method, "#999")}"/>')
        elems.append(f'<text x="{xx + 24}" y="{legend_y}" font-size="13" fill="#222">{escape(METHOD_LABELS.get(method, method))}</text>')
    elems.append("</svg>")
    path.write_text("\n".join(elems) + "\n")


def antmaze_env_label(env: str) -> str:
    if "navigate" in env:
        return "Navigate"
    if "stitch" in env:
        return "Stitch"
    return env


def write_antmaze_multiseed_svg(path: Path, rows: list[dict[str, str]]) -> None:
    width, height = 860, 430
    left, right, top, bottom = 72, 24, 54, 108
    chart_w = width - left - right
    chart_h = height - top - bottom
    envs = ["antmaze-teleport-navigate-v0", "antmaze-teleport-stitch-v0"]
    series = ["Bellman matched", "Stochastic TRL", "Bellman full"]
    colors = {
        "Bellman matched": COLORS["bellman_matched"],
        "Stochastic TRL": COLORS["sto_trl_matched"],
        "Bellman full": COLORS["bellman_full"],
    }
    by_key = {(row["env"], row["method"]): row for row in rows}

    def y(value: float) -> float:
        return top + chart_h - value * chart_h

    group_w = chart_w / len(envs)
    bar_w = min(58, group_w / (len(series) + 1.6))
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, "AntMaze body-candidate multi-seed screen", 18),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{width - right}" y2="{top + chart_h}" stroke="#333" stroke-width="1"/>',
    ]
    for i in range(6):
        val = i / 5
        yy = y(val)
        elems.append(f'<line x1="{left - 5}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="#e5e5e5" stroke-width="1"/>')
        elems.append(f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end" font-size="12" fill="#333">{val:.1f}</text>')
    elems.append(
        f'<text x="18" y="{top + chart_h / 2:.1f}" transform="rotate(-90 18 {top + chart_h / 2:.1f})" '
        'text-anchor="middle" font-size="13" fill="#333">Success rate</text>'
    )
    for gi, env in enumerate(envs):
        center = left + group_w * (gi + 0.5)
        start = center - (len(series) * bar_w) / 2
        for si, method in enumerate(series):
            row = by_key[(env, method)]
            value = float(row["mean_success"])
            stderr = float(row["success_std"])
            cx = start + si * bar_w + bar_w * 0.41
            x0 = start + si * bar_w
            y0 = y(value)
            elems.append(
                f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bar_w * 0.82:.1f}" '
                f'height="{top + chart_h - y0:.1f}" fill="{colors[method]}"/>'
            )
            if stderr > 0:
                ey0 = y(min(1.0, value + stderr))
                ey1 = y(max(0.0, value - stderr))
                elems.append(f'<line x1="{cx:.1f}" y1="{ey0:.1f}" x2="{cx:.1f}" y2="{ey1:.1f}" stroke="#222" stroke-width="1.4"/>')
                elems.append(f'<line x1="{cx - 7:.1f}" y1="{ey0:.1f}" x2="{cx + 7:.1f}" y2="{ey0:.1f}" stroke="#222" stroke-width="1.4"/>')
                elems.append(f'<line x1="{cx - 7:.1f}" y1="{ey1:.1f}" x2="{cx + 7:.1f}" y2="{ey1:.1f}" stroke="#222" stroke-width="1.4"/>')
            elems.append(svg_text(cx, y0 - 8, fmt(value), 11))
        elems.append(svg_text(center, top + chart_h + 24, antmaze_env_label(env), 13))

    legend_y = height - 48
    for i, method in enumerate(series):
        xx = left + i * 220
        elems.append(f'<rect x="{xx}" y="{legend_y}" width="14" height="14" fill="{colors[method]}"/>')
        elems.append(f'<text x="{xx + 20}" y="{legend_y + 12}" font-size="13" fill="#222">{escape(method)}</text>')
    elems.append("</svg>")
    path.write_text("\n".join(elems) + "\n")


def write_antmaze_budget_svg(path: Path, rows: list[dict[str, str]]) -> None:
    width, height = 940, 460
    left, right, top, bottom = 72, 28, 58, 120
    gap = 58
    panel_w = (width - left - right - gap) / 2
    chart_h = height - top - bottom
    envs = ["antmaze-teleport-navigate-v0", "antmaze-teleport-stitch-v0"]
    budgets = sorted({int(row["sweeps"]) for row in rows})
    min_log = math.log2(min(budgets))
    max_log = math.log2(max(budgets))
    by_key = {(row["env"], row["planner"], int(row["sweeps"])): float(row["mean_success"]) for row in rows}

    def x(panel: int, sweeps: int) -> float:
        x0 = left + panel * (panel_w + gap)
        if max_log == min_log:
            return x0 + panel_w / 2
        return x0 + (math.log2(sweeps) - min_log) * panel_w / (max_log - min_log)

    def y(value: float) -> float:
        return top + chart_h - value * chart_h

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 30, "AntMaze budget", 18),
    ]
    for panel, env in enumerate(envs):
        x0 = left + panel * (panel_w + gap)
        x1 = x0 + panel_w
        elems.append(svg_text((x0 + x1) / 2, 52, antmaze_env_label(env), 13))
        elems.append(f'<line x1="{x0:.1f}" y1="{top}" x2="{x0:.1f}" y2="{top + chart_h}" stroke="#333" stroke-width="1"/>')
        elems.append(f'<line x1="{x0:.1f}" y1="{top + chart_h}" x2="{x1:.1f}" y2="{top + chart_h}" stroke="#333" stroke-width="1"/>')
        for tick in range(6):
            val = tick / 5
            yy = y(val)
            elems.append(f'<line x1="{x0 - 4:.1f}" y1="{yy:.1f}" x2="{x1:.1f}" y2="{yy:.1f}" stroke="#e5e5e5" stroke-width="1"/>')
            if panel == 0:
                elems.append(f'<text x="{x0 - 10:.1f}" y="{yy + 4:.1f}" text-anchor="end" font-size="12" fill="#333">{val:.1f}</text>')
        for budget in budgets:
            xx = x(panel, budget)
            elems.append(f'<line x1="{xx:.1f}" y1="{top + chart_h}" x2="{xx:.1f}" y2="{top + chart_h + 5}" stroke="#333" stroke-width="1"/>')
            elems.append(f'<text x="{xx:.1f}" y="{top + chart_h + 22}" text-anchor="middle" font-size="11" fill="#333">{budget}</text>')
        elems.append(svg_text((x0 + x1) / 2, top + chart_h + 48, "Planning sweeps", 13))

        bellman_pts = []
        for budget in budgets:
            value = by_key.get((env, "Bellman", budget))
            if value is not None:
                bellman_pts.append((x(panel, budget), y(value), value))
        if len(bellman_pts) >= 2:
            point_str = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy, _ in bellman_pts)
            elems.append(f'<polyline points="{point_str}" fill="none" stroke="{COLORS["bellman_matched"]}" stroke-width="2.6"/>')
        for xx, yy, value in bellman_pts:
            elems.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{COLORS["bellman_matched"]}"/>')
            if value in {0.0, 1.0}:
                elems.append(svg_text(xx, yy - 9, fmt(value), 10))

        sto_value = by_key.get((env, "Stochastic TRL", 6))
        if sto_value is not None:
            xx = x(panel, 6)
            yy = y(sto_value)
            elems.append(f'<rect x="{xx - 5:.1f}" y="{yy - 5:.1f}" width="10" height="10" fill="{COLORS["sto_trl_matched"]}"/>')
            elems.append(svg_text(xx + 18, yy - 9, fmt(sto_value), 10, anchor="start"))
        full_value = by_key.get((env, "Bellman full", 180))
        if full_value is not None:
            xx = x(panel, 180)
            yy = y(full_value)
            elems.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{COLORS["bellman_full"]}"/>')

    elems.append(
        f'<text x="18" y="{top + chart_h / 2:.1f}" transform="rotate(-90 18 {top + chart_h / 2:.1f})" '
        'text-anchor="middle" font-size="13" fill="#333">Success rate</text>'
    )
    legend_y = height - 26
    legend = [
        ("Bellman", COLORS["bellman_matched"], "line"),
        ("Stochastic TRL at 6", COLORS["sto_trl_matched"], "square"),
        ("Bellman full", COLORS["bellman_full"], "dot"),
    ]
    for i, (label, color, marker) in enumerate(legend):
        xx = left + i * 230
        if marker == "line":
            elems.append(f'<line x1="{xx}" y1="{legend_y - 4}" x2="{xx + 18}" y2="{legend_y - 4}" stroke="{color}" stroke-width="3"/>')
            elems.append(f'<circle cx="{xx + 9}" cy="{legend_y - 4}" r="4" fill="{color}"/>')
        elif marker == "square":
            elems.append(f'<rect x="{xx + 4}" y="{legend_y - 9}" width="10" height="10" fill="{color}"/>')
        else:
            elems.append(f'<circle cx="{xx + 9}" cy="{legend_y - 4}" r="4" fill="{color}"/>')
        elems.append(f'<text x="{xx + 24}" y="{legend_y}" font-size="13" fill="#222">{escape(label)}</text>')
    elems.append("</svg>")
    path.write_text("\n".join(elems) + "\n")


def build_tabular_l128() -> tuple[list[str], list[dict[str, str]], dict[tuple[str, str], float]]:
    rows = read_rows(TABULAR_L128)
    grouped: dict[tuple[float, str], list[dict[str, str]]] = {}
    for row in rows:
        p_success = to_float(row.get("p_success"))
        if p_success is None:
            continue
        family = method_family(row["method"])
        if family not in {"mc_positive", "mc_all_goals", "bellman_matched", "sto_trl_matched"}:
            continue
        grouped.setdefault((p_success, family), []).append(row)

    out: list[dict[str, str]] = []
    safe_success_values: dict[tuple[str, str], float] = {}
    for (p_success, family), vals in sorted(grouped.items(), key=lambda x: (x[0][0], method_sort_key(x[0][1]))):
        def avg_col(key: str) -> float:
            return mean(x for x in (to_float(v.get(key)) for v in vals) if x is not None)

        avg = {
            "optimal_start_action": avg_col("optimal_start_action"),
            "success_rate": avg_col("success_rate"),
            "risky_action": avg_col("risky_action"),
            "regret": avg_col("regret"),
            "pred_start_safe": avg_col("pred_start_safe"),
            "pred_start_risky": avg_col("pred_start_risky"),
            "long_horizon_mse": avg_col("long_horizon_mse"),
        }
        optimal = action_name(avg.get("optimal_start_action"))
        row = {
            "p_success": fmt(p_success, 2),
            "optimal_action": optimal,
            "method": METHOD_LABELS.get(family, family),
            "n": str(len(vals)),
            "success_rate": fmt(avg.get("success_rate")),
            "risky_action_rate": fmt(avg.get("risky_action")),
            "regret": fmt(avg.get("regret")),
            "pred_safe": fmt(avg.get("pred_start_safe")),
            "pred_risky": fmt(avg.get("pred_start_risky")),
            "long_horizon_mse": fmt(avg.get("long_horizon_mse"), 5),
        }
        out.append(row)
        if p_success in {0.02, 0.05} and family in {"mc_positive", "bellman_matched", "sto_trl_matched"}:
            safe_success_values[(f"p={p_success:.2f}", family)] = avg["success_rate"]
    fields = [
        "p_success",
        "optimal_action",
        "method",
        "n",
        "success_rate",
        "risky_action_rate",
        "regret",
        "pred_safe",
        "pred_risky",
        "long_horizon_mse",
    ]
    return fields, out, safe_success_values


def build_tabular_risky_aggregate() -> tuple[list[str], list[dict[str, str]]]:
    rows = [row for row in read_rows(TABULAR_COMPACT) if row.get("stage") == "risky"]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(method_family(row["method"]), []).append(row)

    out: list[dict[str, str]] = []
    for family in sorted(grouped, key=method_sort_key):
        vals = grouped[family]

        def avg_col(key: str, subset: list[dict[str, str]] = vals) -> float:
            return mean(x for x in (to_float(v.get(key)) for v in subset) if x is not None)

        safe_opt = [v for v in vals if to_float(v.get("optimal_start_action")) == 0.0]
        risky_opt = [v for v in vals if to_float(v.get("optimal_start_action")) == 1.0]
        out.append(
            {
                "method": METHOD_LABELS.get(family, family),
                "n": str(len(vals)),
                "success_rate": fmt(avg_col("success_rate")),
                "regret": fmt(avg_col("regret")),
                "long_horizon_mse": fmt(avg_col("long_horizon_mse"), 5),
                "safe_opt_risky_rate": fmt(avg_col("risky_action", safe_opt)) if safe_opt else "",
                "risky_opt_risky_rate": fmt(avg_col("risky_action", risky_opt)) if risky_opt else "",
                "risky_overestimate_ratio": fmt(avg_col("risky_overestimate_ratio")),
            }
        )
    fields = [
        "method",
        "n",
        "success_rate",
        "regret",
        "long_horizon_mse",
        "safe_opt_risky_rate",
        "risky_opt_risky_rate",
        "risky_overestimate_ratio",
    ]
    return fields, out


def build_tabular_horizon() -> tuple[list[str], list[dict[str, str]]]:
    rows = read_rows(TABULAR_SAFE_HORIZON)
    wanted = {"mc_positive", "mc_all_goals", "bellman_matched", "sto_trl_matched", "bellman_full"}
    grouped: dict[tuple[int, float, str], list[dict[str, str]]] = {}
    matched_sweeps: dict[tuple[int, float, str], str] = {}
    for row in rows:
        p_success = to_float(row.get("p_success"))
        safe_length = to_float(row.get("safe_length"))
        if p_success is None or safe_length is None:
            continue
        family = method_family(row["method"])
        if family not in wanted:
            continue
        key = (int(safe_length), p_success, family)
        grouped.setdefault(key, []).append(row)
        if row["method"].startswith("bellman_") and row["method"].endswith("_sweeps"):
            matched_sweeps[key] = row["method"].split("_")[1]
        if row["method"].startswith("sto_trl_") and row["method"].endswith("_sweeps"):
            matched_sweeps[key] = row["method"].split("_")[2]

    out: list[dict[str, str]] = []
    for (safe_length, p_success, family), vals in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], method_sort_key(item[0][2]))
    ):
        def avg_col(key: str) -> float:
            return mean(x for x in (to_float(v.get(key)) for v in vals) if x is not None)

        row = {
            "safe_length": str(safe_length),
            "p_success": fmt(p_success, 2),
            "method_key": family,
            "method": METHOD_LABELS.get(family, family),
            "n": str(len(vals)),
            "matched_sweeps": matched_sweeps.get((safe_length, p_success, family), ""),
            "success_rate": fmt(avg_col("success_rate")),
            "risky_action_rate": fmt(avg_col("risky_action")),
            "regret": fmt(avg_col("regret")),
            "long_horizon_mse": fmt(avg_col("long_horizon_mse"), 5),
        }
        out.append(row)
    fields = [
        "safe_length",
        "p_success",
        "method",
        "n",
        "matched_sweeps",
        "success_rate",
        "risky_action_rate",
        "regret",
        "long_horizon_mse",
    ]
    return fields, out


def build_grid_shortcut() -> tuple[list[str], list[dict[str, str]]]:
    rows = read_rows(GRID_SHORTCUT)
    wanted = {"mc_positive", "mc_all_goals", "bellman_matched", "sto_trl_matched", "bellman_full"}
    grouped: dict[tuple[int, float, str], list[dict[str, str]]] = {}
    for row in rows:
        path_length = to_float(row.get("path_length"))
        p_success = to_float(row.get("p_success"))
        if path_length is None or p_success is None:
            continue
        family = method_family(row["method"])
        if family not in wanted:
            continue
        grouped.setdefault((int(path_length), p_success, family), []).append(row)

    out: list[dict[str, str]] = []
    for (path_length, p_success, family), vals in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], method_sort_key(item[0][2]))
    ):
        def avg_col(key: str) -> float:
            return mean(x for x in (to_float(v.get(key)) for v in vals) if x is not None)

        width = int(float(vals[0]["width"]))
        height = int(float(vals[0]["height"]))
        row = {
            "grid": f"{width}x{height}",
            "safe_length": str(path_length),
            "p_success": fmt(p_success, 2),
            "method_key": family,
            "method": METHOD_LABELS.get(family, family),
            "n": str(len(vals)),
            "matched_sweeps": vals[0].get("matched_sweeps", ""),
            "success_rate": fmt(avg_col("success_rate")),
            "safe_action_rate": fmt(avg_col("safe_action_rate")),
            "portal_action_rate": fmt(avg_col("portal_action_rate")),
            "regret": fmt(avg_col("regret")),
            "portal_overestimate_ratio": fmt(avg_col("portal_overestimate_ratio")),
            "long_horizon_mse": fmt(avg_col("long_horizon_mse"), 5),
        }
        out.append(row)
    fields = [
        "grid",
        "safe_length",
        "p_success",
        "method",
        "n",
        "matched_sweeps",
        "success_rate",
        "safe_action_rate",
        "portal_action_rate",
        "regret",
        "portal_overestimate_ratio",
        "long_horizon_mse",
    ]
    return fields, out


def build_grid_realized_diagnostic() -> tuple[list[str], list[dict[str, str]]]:
    rows = read_rows(GRID_REALIZED_DIAGNOSTIC)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(method_family(row["method"]), []).append(row)

    out: list[dict[str, str]] = []
    for family in sorted(grouped, key=method_sort_key):
        vals = grouped[family]

        def avg_col(key: str) -> float:
            return mean(x for x in (to_float(v.get(key)) for v in vals) if x is not None)

        out.append(
            {
                "method": METHOD_LABELS.get(family, family),
                "n": str(len(vals)),
                "success_rate": fmt(avg_col("success_rate")),
                "safe_action_rate": fmt(avg_col("safe_action_rate")),
                "portal_action_rate": fmt(avg_col("portal_action_rate")),
                "regret": fmt(avg_col("regret")),
                "portal_overestimate_ratio": fmt(avg_col("portal_overestimate_ratio")),
            }
        )
    fields = [
        "method",
        "n",
        "success_rate",
        "safe_action_rate",
        "portal_action_rate",
        "regret",
        "portal_overestimate_ratio",
    ]
    return fields, out


def build_grid_budget_curve() -> tuple[list[str], list[dict[str, str]]]:
    rows = read_rows(GRID_BUDGET_CURVE)
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], int(float(row["sweeps"]))), []).append(row)

    out: list[dict[str, str]] = []
    for (method, sweeps), vals in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        def avg_col(key: str) -> float:
            return mean(x for x in (to_float(v.get(key)) for v in vals) if x is not None)

        out.append(
            {
                "method_key": method,
                "method": METHOD_LABELS.get(method, method),
                "sweeps": str(sweeps),
                "n": str(len(vals)),
                "success_rate": fmt(avg_col("success_rate")),
                "safe_action_rate": fmt(avg_col("safe_action_rate")),
                "portal_action_rate": fmt(avg_col("portal_action_rate")),
                "regret": fmt(avg_col("regret")),
                "pred_safe": fmt(avg_col("pred_start_safe")),
                "pred_portal": fmt(avg_col("pred_start_portal")),
                "long_horizon_mse": fmt(avg_col("long_horizon_mse"), 5),
            }
        )
    fields = [
        "method",
        "sweeps",
        "n",
        "success_rate",
        "safe_action_rate",
        "portal_action_rate",
        "regret",
        "pred_safe",
        "pred_portal",
        "long_horizon_mse",
    ]
    return fields, out


def build_graph_summary() -> tuple[list[str], list[dict[str, str]], list[Bar]]:
    wanted = ["bellman_matched", "sto_trl_matched", "bellman_full"]
    per_method: dict[str, list[dict[str, str]]] = {method: [] for method in wanted}
    for path in GRAPH_MAIN_FILES:
        for row in read_rows(path):
            if row["method"] in per_method:
                per_method[row["method"]].append(row)

    out: list[dict[str, str]] = []
    bars: list[Bar] = []
    all_seeds = sorted(
        {
            int(float(row["seed"]))
            for vals in per_method.values()
            for row in vals
            if row.get("seed") not in {"", None}
        }
    )
    for method in wanted:
        vals = sorted(per_method[method], key=lambda row: int(float(row["seed"])))
        by_seed = {int(float(row["seed"])): to_float(row["overall_success"]) for row in vals}
        scores = [by_seed[s] for s in all_seeds if by_seed.get(s) is not None]
        task_means = []
        for task_i in range(1, 6):
            key = f"task{task_i}_success"
            task_means.append(mean(x for x in (to_float(row.get(key)) for row in vals) if x is not None))
        mu = mean(scores)
        sd = pstdev(scores)
        row = {
            "method": METHOD_LABELS.get(method, method),
            "sweeps": str(int(float(vals[0]["iters"]))) if vals else "",
            "executor": vals[0].get("action_mode", "") if vals else "",
        }
        for seed in all_seeds:
            row[f"seed{seed}_success"] = fmt(by_seed.get(seed))
        row.update(
            {
                "mean_success": fmt(mu),
                "std_success": fmt(sd),
                "task1_mean": fmt(task_means[0]),
                "task2_mean": fmt(task_means[1]),
                "task3_mean": fmt(task_means[2]),
                "task4_mean": fmt(task_means[3]),
                "task5_mean": fmt(task_means[4]),
            }
        )
        out.append(row)
        bars.append(
            Bar(
                label=METHOD_LABELS.get(method, method),
                value=mu,
                stderr=sd,
                color=COLORS.get(method, "#999"),
            )
        )
    fields = [
        "method",
        "sweeps",
        "executor",
        *[f"seed{seed}_success" for seed in all_seeds],
        "mean_success",
        "std_success",
        "task1_mean",
        "task2_mean",
        "task3_mean",
        "task4_mean",
        "task5_mean",
    ]
    return fields, out, bars


def graph_rows_by_method() -> dict[str, list[dict[str, str]]]:
    wanted = ["bellman_matched", "sto_trl_matched", "bellman_full"]
    per_method: dict[str, list[dict[str, str]]] = {method: [] for method in wanted}
    for path in GRAPH_MAIN_FILES:
        for row in read_rows(path):
            if row["method"] in per_method:
                per_method[row["method"]].append(row)
    for method, rows in per_method.items():
        per_method[method] = sorted(rows, key=lambda row: int(float(row["seed"])))
    return per_method


def build_graph_paired_stats() -> tuple[list[str], list[dict[str, str]]]:
    per_method = graph_rows_by_method()

    def by_seed(method: str) -> dict[int, float]:
        return {
            int(float(row["seed"])): float(row["overall_success"])
            for row in per_method[method]
        }

    baseline = by_seed("bellman_matched")
    sto = by_seed("sto_trl_matched")
    full = by_seed("bellman_full")

    def paired_row(left: str, right: str, label: str) -> dict[str, str]:
        left_scores = by_seed(left)
        right_scores = by_seed(right)
        seeds = sorted(set(left_scores) & set(right_scores))
        diffs = [left_scores[seed] - right_scores[seed] for seed in seeds]
        n = len(diffs)
        mu = mean(diffs)
        sd = sample_stdev(diffs)
        sem = sd / math.sqrt(n) if n else float("nan")
        half_width = t_critical_95(n) * sem if n else float("nan")
        return {
            "comparison": label,
            "n_seeds": str(n),
            "mean_diff": fmt(mu),
            "sample_sd_diff": fmt(sd),
            "sem_diff": fmt(sem),
            "ci95_low": fmt(mu - half_width),
            "ci95_high": fmt(mu + half_width),
            "min_seed_diff": fmt(min(diffs) if diffs else None),
            "max_seed_diff": fmt(max(diffs) if diffs else None),
            "all_seed_diffs_positive": str(all(diff > 0 for diff in diffs)),
            "relative_gain_vs_right": fmt(mu / mean(right_scores.values()) if right_scores else None),
            "full_gap_recovery": "",
        }

    rows = [
        paired_row(
            "sto_trl_matched",
            "bellman_matched",
            "Stochastic TRL - Bellman matched",
        ),
        paired_row(
            "bellman_full",
            "sto_trl_matched",
            "Bellman full - Stochastic TRL",
        ),
    ]

    matched_mean = mean(baseline.values())
    sto_mean = mean(sto.values())
    full_mean = mean(full.values())
    gap = full_mean - matched_mean
    if gap > 0:
        rows[0]["full_gap_recovery"] = fmt((sto_mean - matched_mean) / gap)
        rows[1]["full_gap_recovery"] = fmt((full_mean - sto_mean) / gap)

    fields = [
        "comparison",
        "n_seeds",
        "mean_diff",
        "sample_sd_diff",
        "sem_diff",
        "ci95_low",
        "ci95_high",
        "min_seed_diff",
        "max_seed_diff",
        "all_seed_diffs_positive",
        "relative_gain_vs_right",
        "full_gap_recovery",
    ]
    return fields, rows


def build_graph_task_deltas() -> tuple[list[str], list[dict[str, str]]]:
    per_method = graph_rows_by_method()
    rows: list[dict[str, str]] = []
    for task_i in range(1, 6):
        key = f"task{task_i}_success"
        means = {
            method: mean(float(row[key]) for row in vals)
            for method, vals in per_method.items()
            if vals
        }
        matched = means.get("bellman_matched")
        sto = means.get("sto_trl_matched")
        full = means.get("bellman_full")
        rows.append(
            {
                "task": f"task{task_i}",
                "bellman_matched": fmt(matched),
                "sto_trl": fmt(sto),
                "bellman_full": fmt(full),
                "sto_minus_matched": fmt(sto - matched if sto is not None and matched is not None else None),
                "full_minus_sto": fmt(full - sto if full is not None and sto is not None else None),
            }
        )
    fields = [
        "task",
        "bellman_matched",
        "sto_trl",
        "bellman_full",
        "sto_minus_matched",
        "full_minus_sto",
    ]
    return fields, rows


def build_pointmaze_topology_summary_for(
    path: Path,
    order: list[str],
) -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "method",
        "sweeps",
        "n",
        "mean_success",
        "std_success",
        "task1_mean",
        "task2_mean",
        "task3_mean",
        "task4_mean",
        "task5_mean",
    ]
    if not path.exists():
        return fields, []

    rows = read_rows(path)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)

    out: list[dict[str, str]] = []
    for method in order:
        vals = grouped.get(method, [])
        if not vals:
            continue

        def avg_col(key: str) -> float:
            return mean(float(row[key]) for row in vals)

        row = {
            "method": METHOD_LABELS.get(method, method),
            "sweeps": str(int(float(vals[0]["iters"]))),
            "n": str(len(vals)),
            "mean_success": fmt(avg_col("overall_success")),
            "std_success": fmt(pstdev(float(v["overall_success"]) for v in vals)),
        }
        for task_i in range(1, 6):
            row[f"task{task_i}_mean"] = fmt(avg_col(f"task{task_i}_success"))
        out.append(row)

    return fields, out


def build_pointmaze_topology_summary() -> tuple[list[str], list[dict[str, str]]]:
    return build_pointmaze_topology_summary_for(
        POINTMAZE_TOPOLOGY,
        ["bellman_matched", "sto_trl_matched", "bellman_full"],
    )


def build_pointmaze_topology_stitch_summary() -> tuple[list[str], list[dict[str, str]]]:
    return build_pointmaze_topology_summary_for(
        POINTMAZE_TOPOLOGY_STITCH,
        ["bellman_matched", "sto_trl_matched", "bellman_full"],
    )


def build_pointmaze_topology_stitch_support_summary() -> tuple[list[str], list[dict[str, str]]]:
    return build_pointmaze_topology_summary_for(
        POINTMAZE_TOPOLOGY_STITCH_SUPPORT,
        ["bellman_matched", "support_trl_matched", "sto_trl_matched", "bellman_full"],
    )


def build_pointmaze_topology_paired_stats_for(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    rows = read_rows(path)
    by_method_seed = {
        (row["method"], int(float(row["seed"]))): float(row["overall_success"])
        for row in rows
    }

    def paired_row(left: str, right: str, label: str) -> dict[str, str]:
        seeds = sorted(
            seed
            for method, seed in by_method_seed
            if method == left and (right, seed) in by_method_seed
        )
        diffs = [by_method_seed[(left, seed)] - by_method_seed[(right, seed)] for seed in seeds]
        n = len(diffs)
        mu = mean(diffs)
        sd = sample_stdev(diffs)
        sem = sd / math.sqrt(n) if n else float("nan")
        half_width = t_critical_95(n) * sem if n else float("nan")
        right_mean = mean(by_method_seed[(right, seed)] for seed in seeds) if seeds else float("nan")
        return {
            "comparison": label,
            "n_seeds": str(n),
            "mean_diff": fmt(mu),
            "sample_sd_diff": fmt(sd),
            "sem_diff": fmt(sem),
            "ci95_low": fmt(mu - half_width),
            "ci95_high": fmt(mu + half_width),
            "min_seed_diff": fmt(min(diffs) if diffs else None),
            "max_seed_diff": fmt(max(diffs) if diffs else None),
            "all_seed_diffs_nonnegative": str(all(diff >= -1e-12 for diff in diffs)),
            "relative_gain_vs_right": fmt(mu / right_mean if right_mean else None),
        }

    fields = [
        "comparison",
        "n_seeds",
        "mean_diff",
        "sample_sd_diff",
        "sem_diff",
        "ci95_low",
        "ci95_high",
        "min_seed_diff",
        "max_seed_diff",
        "all_seed_diffs_nonnegative",
        "relative_gain_vs_right",
    ]
    return fields, [
        paired_row(
            "sto_trl_matched",
            "bellman_matched",
            "Stochastic TRL - Bellman matched",
        ),
        paired_row(
            "bellman_full",
            "sto_trl_matched",
            "Bellman full - Stochastic TRL",
        ),
    ]


def build_pointmaze_topology_paired_stats() -> tuple[list[str], list[dict[str, str]]]:
    return build_pointmaze_topology_paired_stats_for(POINTMAZE_TOPOLOGY)


def build_pointmaze_topology_stitch_paired_stats() -> tuple[list[str], list[dict[str, str]]]:
    if not POINTMAZE_TOPOLOGY_STITCH.exists():
        fields, _rows = build_pointmaze_topology_paired_stats_for(POINTMAZE_TOPOLOGY)
        return fields, []
    return build_pointmaze_topology_paired_stats_for(POINTMAZE_TOPOLOGY_STITCH)


def build_antmaze_bc_summary() -> tuple[list[str], list[dict[str, str]]]:
    if not ANTMAZE_BC_TOPOLOGY.exists():
        fields = [
            "method",
            "sweeps",
            "n",
            "mean_success",
            "task1_mean",
            "task2_mean",
            "task3_mean",
            "task4_mean",
            "task5_mean",
        ]
        return fields, []

    rows = read_rows(ANTMAZE_BC_TOPOLOGY)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)

    order = ["bellman_matched", "sto_trl_matched", "bellman_full"]
    out: list[dict[str, str]] = []
    for method in order:
        vals = grouped.get(method, [])
        if not vals:
            continue

        def avg_col(key: str) -> float:
            return mean(float(row[key]) for row in vals)

        row = {
            "method": METHOD_LABELS.get(method, method),
            "sweeps": str(int(float(vals[0]["iters"]))),
            "n": str(len(vals)),
            "mean_success": fmt(avg_col("overall_success")),
        }
        for task_i in range(1, 6):
            row[f"task{task_i}_mean"] = fmt(avg_col(f"task{task_i}_success"))
        out.append(row)

    fields = [
        "method",
        "sweeps",
        "n",
        "mean_success",
        "task1_mean",
        "task2_mean",
        "task3_mean",
        "task4_mean",
        "task5_mean",
    ]
    return fields, out


def build_antmaze_navigate_bodyk16_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "method",
        "sweeps",
        "n",
        "mean_success",
        "task1_mean",
        "task2_mean",
        "task3_mean",
        "task4_mean",
        "task5_mean",
    ]
    rows: list[dict[str, str]] = []
    for path in ANTMAZE_NAVIGATE_BODYK16_FILES:
        if path.exists():
            rows.extend(read_rows(path))
    if not rows:
        return fields, []

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)

    out: list[dict[str, str]] = []
    for method in ["bellman_matched", "sto_trl_matched", "bellman_full"]:
        vals = grouped.get(method, [])
        if not vals:
            continue

        def avg_col(key: str) -> float:
            return mean(float(row[key]) for row in vals)

        row = {
            "method": METHOD_LABELS.get(method, method),
            "sweeps": str(int(float(vals[0]["iters"]))),
            "n": str(len(vals)),
            "mean_success": fmt(avg_col("overall_success")),
        }
        for task_i in range(1, 6):
            row[f"task{task_i}_mean"] = fmt(avg_col(f"task{task_i}_success"))
        out.append(row)
    return fields, out


def build_antmaze_bodyk16_multiseed_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "episodes_per_task",
        "eval_seeds",
        "controller_steps",
        "method",
        "sweeps",
        "mean_success",
        "success_std",
        "task1_mean",
        "task2_mean",
        "task3_mean",
        "task4_mean",
        "task5_mean",
    ]
    rows: list[dict[str, str]] = []
    for path in ANTMAZE_BODYK16_MULTI_FILES:
        if path.exists():
            rows.extend(read_rows(path))
    if not rows:
        return fields, []

    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["env"], row["method"]), []).append(row)

    out: list[dict[str, str]] = []
    env_order = ["antmaze-teleport-navigate-v0", "antmaze-teleport-stitch-v0"]
    for env in env_order:
        for method in ["bellman_matched", "sto_trl_matched", "bellman_full"]:
            vals = grouped.get((env, method), [])
            if not vals:
                continue

            def avg_col(key: str) -> float:
                return mean(float(row[key]) for row in vals)

            success_vals = [float(row["overall_success"]) for row in vals]
            row = {
                "env": env,
                "episodes_per_task": str(int(float(vals[0]["episodes_per_task"]))),
                "eval_seeds": str(len(vals)),
                "controller_steps": str(int(float(vals[0]["bc_steps"]))),
                "method": METHOD_LABELS.get(method, method),
                "sweeps": str(int(float(vals[0]["iters"]))),
                "mean_success": fmt(mean(success_vals)),
                "success_std": fmt(pstdev(success_vals)),
            }
            for task_i in range(1, 6):
                row[f"task{task_i}_mean"] = fmt(avg_col(f"task{task_i}_success"))
            out.append(row)
    return fields, out


def build_antmaze_bodyk16_ep20_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "episodes_per_task",
        "eval_seed",
        "controller_steps",
        "method",
        "sweeps",
        "mean_success",
        "task1_mean",
        "task2_mean",
        "task3_mean",
        "task4_mean",
        "task5_mean",
    ]
    rows: list[dict[str, str]] = []
    for path in ANTMAZE_BODYK16_EP20_FILES:
        if path.exists():
            rows.extend(read_rows(path))
    if not rows:
        return fields, []

    env_order = {"antmaze-teleport-navigate-v0": 0, "antmaze-teleport-stitch-v0": 1}
    method_order = {"bellman_matched": 0, "sto_trl_matched": 1, "bellman_full": 2}
    controller_steps = {
        "antmaze-teleport-navigate-v0": 50000,
        "antmaze-teleport-stitch-v0": 20000,
    }
    out: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda r: (env_order.get(r["env"], 99), method_order.get(r["method"], 99))):
        out.append(
            {
                "env": row["env"],
                "episodes_per_task": str(int(float(row["episodes_per_task"]))),
                "eval_seed": str(int(float(row["seed"]))),
                "controller_steps": str(controller_steps.get(row["env"], int(float(row["bc_steps"])))),
                "method": METHOD_LABELS.get(row["method"], row["method"]),
                "sweeps": str(int(float(row["iters"]))),
                "mean_success": fmt(float(row["overall_success"])),
                "task1_mean": fmt(float(row["task1_success"])),
                "task2_mean": fmt(float(row["task2_success"])),
                "task3_mean": fmt(float(row["task3_success"])),
                "task4_mean": fmt(float(row["task4_success"])),
                "task5_mean": fmt(float(row["task5_success"])),
            }
        )
    return fields, out


def build_antmaze_bodyk16_ep20_paired_stats() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "n_eval_seeds",
        "sto_minus_matched_mean",
        "ci95_low",
        "ci95_high",
        "min_seed_diff",
        "max_seed_diff",
        "full_minus_sto_mean",
        "all_seed_diffs_positive",
    ]
    rows: list[dict[str, str]] = []
    for path in ANTMAZE_BODYK16_EP20_FILES:
        if path.exists():
            rows.extend(read_rows(path))
    if not rows:
        return fields, []

    by_env_method_seed = {
        (row["env"], row["method"], int(float(row["seed"]))): float(row["overall_success"])
        for row in rows
    }
    env_order = ["antmaze-teleport-navigate-v0", "antmaze-teleport-stitch-v0"]
    out: list[dict[str, str]] = []
    for env in env_order:
        seeds = sorted(
            seed
            for row_env, method, seed in by_env_method_seed
            if row_env == env
            and method == "sto_trl_matched"
            and (env, "bellman_matched", seed) in by_env_method_seed
            and (env, "bellman_full", seed) in by_env_method_seed
        )
        if not seeds:
            continue
        diffs = [
            by_env_method_seed[(env, "sto_trl_matched", seed)]
            - by_env_method_seed[(env, "bellman_matched", seed)]
            for seed in seeds
        ]
        full_diffs = [
            by_env_method_seed[(env, "bellman_full", seed)]
            - by_env_method_seed[(env, "sto_trl_matched", seed)]
            for seed in seeds
        ]
        n = len(diffs)
        mu = mean(diffs)
        sd = sample_stdev(diffs)
        sem = sd / math.sqrt(n) if n else float("nan")
        half_width = t_critical_95(n) * sem if n else float("nan")
        out.append(
            {
                "env": env,
                "n_eval_seeds": str(n),
                "sto_minus_matched_mean": fmt(mu),
                "ci95_low": fmt(mu - half_width),
                "ci95_high": fmt(mu + half_width),
                "min_seed_diff": fmt(min(diffs)),
                "max_seed_diff": fmt(max(diffs)),
                "full_minus_sto_mean": fmt(mean(full_diffs)),
                "all_seed_diffs_positive": str(all(diff > 0.0 for diff in diffs)),
            }
        )
    return fields, out


def build_antmaze_bcseed_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "controller_seed",
        "episodes_per_task",
        "eval_seeds",
        "controller_steps",
        "method",
        "sweeps",
        "mean_success",
        "success_std",
        "task1_mean",
        "task2_mean",
        "task3_mean",
        "task4_mean",
        "task5_mean",
    ]
    rows: list[dict[str, str]] = []
    for path in ANTMAZE_BCSEED1_FILES:
        if path.exists():
            rows.extend(read_rows(path))
    if not rows:
        return fields, []

    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["env"], row["method"]), []).append(row)

    out: list[dict[str, str]] = []
    env_order = ["antmaze-teleport-navigate-v0", "antmaze-teleport-stitch-v0"]
    for env in env_order:
        for method in ["bellman_matched", "sto_trl_matched", "bellman_full"]:
            vals = grouped.get((env, method), [])
            if not vals:
                continue
            success_vals = [float(row["overall_success"]) for row in vals]

            def avg_col(key: str) -> float:
                return mean(float(row[key]) for row in vals)

            out_row = {
                "env": vals[0]["env"],
                "controller_seed": str(int(float(vals[0].get("bc_seed", "1")))),
                "episodes_per_task": str(int(float(vals[0]["episodes_per_task"]))),
                "eval_seeds": str(len(vals)),
                "controller_steps": str(int(float(vals[0]["bc_steps"]))),
                "method": METHOD_LABELS.get(method, method),
                "sweeps": str(int(float(vals[0]["iters"]))),
                "mean_success": fmt(mean(success_vals)),
                "success_std": fmt(pstdev(success_vals)),
            }
            for task_i in range(1, 6):
                out_row[f"task{task_i}_mean"] = fmt(avg_col(f"task{task_i}_success"))
            out.append(out_row)
    return fields, out


def build_antmaze_stitch_controller_seed_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "controller_seed",
        "episodes_per_task",
        "eval_seeds",
        "controller_steps",
        "method",
        "sweeps",
        "mean_success",
        "success_std",
        "task1_mean",
        "task2_mean",
        "task3_mean",
        "task4_mean",
        "task5_mean",
    ]
    out: list[dict[str, str]] = []
    for controller_seed, path in ANTMAZE_STITCH_CONTROLLER_SEED_FILES:
        if not path.exists():
            continue
        rows = read_rows(path)
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row["method"], []).append(row)
        for method in ["bellman_matched", "sto_trl_matched", "bellman_full"]:
            vals = grouped.get(method, [])
            if not vals:
                continue
            success_vals = [float(row["overall_success"]) for row in vals]

            def avg_col(key: str) -> float:
                return mean(float(row[key]) for row in vals)

            out_row = {
                "env": vals[0]["env"],
                "controller_seed": controller_seed,
                "episodes_per_task": str(int(float(vals[0]["episodes_per_task"]))),
                "eval_seeds": str(len(vals)),
                "controller_steps": str(int(float(vals[0]["bc_steps"]))),
                "method": METHOD_LABELS.get(method, method),
                "sweeps": str(int(float(vals[0]["iters"]))),
                "mean_success": fmt(mean(success_vals)),
                "success_std": fmt(pstdev(success_vals)),
            }
            for task_i in range(1, 6):
                out_row[f"task{task_i}_mean"] = fmt(avg_col(f"task{task_i}_success"))
            out.append(out_row)
    return fields, out


def build_antmaze_navigate_controller_seed_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "controller_seed",
        "episodes_per_task",
        "eval_seeds",
        "controller_steps",
        "method",
        "sweeps",
        "mean_success",
        "success_std",
        "task1_mean",
        "task2_mean",
        "task3_mean",
        "task4_mean",
        "task5_mean",
    ]
    out: list[dict[str, str]] = []
    for controller_seed, path in ANTMAZE_NAVIGATE_CONTROLLER_SEED_FILES:
        if not path.exists():
            continue
        rows = read_rows(path)
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row["method"], []).append(row)
        for method in ["bellman_matched", "sto_trl_matched", "bellman_full"]:
            vals = grouped.get(method, [])
            if not vals:
                continue
            success_vals = [float(row["overall_success"]) for row in vals]

            def avg_col(key: str) -> float:
                return mean(float(row[key]) for row in vals)

            out_row = {
                "env": vals[0]["env"],
                "controller_seed": controller_seed,
                "episodes_per_task": str(int(float(vals[0]["episodes_per_task"]))),
                "eval_seeds": str(len(vals)),
                "controller_steps": str(int(float(vals[0]["bc_steps"]))),
                "method": METHOD_LABELS.get(method, method),
                "sweeps": str(int(float(vals[0]["iters"]))),
                "mean_success": fmt(mean(success_vals)),
                "success_std": fmt(pstdev(success_vals)),
            }
            for task_i in range(1, 6):
                out_row[f"task{task_i}_mean"] = fmt(avg_col(f"task{task_i}_success"))
            out.append(out_row)
    return fields, out


def build_antmaze_budget() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "planner",
        "sweeps",
        "mean_success",
        "task1_mean",
        "task2_mean",
        "task3_mean",
        "task4_mean",
        "task5_mean",
    ]
    rows: list[dict[str, str]] = []
    for path in ANTMAZE_BUDGET_FILES:
        if path.exists():
            rows.extend(read_rows(path))
    if not rows:
        return fields, []

    def planner_and_sweeps(method: str, row: dict[str, str]) -> tuple[str, int, int]:
        if method == "bellman_full":
            return "Bellman full", int(float(row["iters"])), 1000
        if method.startswith("sto_trl_") and method.endswith("_sweeps"):
            return "Stochastic TRL", int(float(row["iters"])), 1
        if method.startswith("bellman_") and method.endswith("_sweeps"):
            return "Bellman", int(float(row["iters"])), 0
        return METHOD_LABELS.get(method, method), int(float(row["iters"])), 500

    out: list[dict[str, str]] = []
    env_order = {"antmaze-teleport-navigate-v0": 0, "antmaze-teleport-stitch-v0": 1}
    for row in sorted(
        rows,
        key=lambda r: (
            env_order.get(r["env"], 99),
            planner_and_sweeps(r["method"], r)[1],
            planner_and_sweeps(r["method"], r)[2],
        ),
    ):
        planner, sweeps, _rank = planner_and_sweeps(row["method"], row)
        out.append(
            {
                "env": row["env"],
                "planner": planner,
                "sweeps": str(sweeps),
                "mean_success": fmt(float(row["overall_success"])),
                "task1_mean": fmt(float(row["task1_success"])),
                "task2_mean": fmt(float(row["task2_success"])),
                "task3_mean": fmt(float(row["task3_success"])),
                "task4_mean": fmt(float(row["task4_success"])),
                "task5_mean": fmt(float(row["task5_success"])),
            }
        )
    return fields, out


def build_antmaze_stitch_executor_ablation() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "executor",
        "method",
        "sweeps",
        "mean_success",
        "task1_mean",
        "task2_mean",
        "task3_mean",
        "task4_mean",
        "task5_mean",
    ]
    rows: list[dict[str, str]] = []
    for executor, path in ANTMAZE_STITCH_EXECUTOR_ABLATION_FILES:
        if not path.exists():
            continue
        for row in read_rows(path):
            rows.append({**row, "executor": executor})
    if not rows:
        return fields, []

    out: list[dict[str, str]] = []
    executor_order = {"nearest_xy": 0, "body_nearest_k16": 1}
    method_order = {"sto_trl_matched": 0, "bellman_full": 1}
    for row in sorted(rows, key=lambda r: (executor_order.get(r["executor"], 99), method_order.get(r["method"], 99))):
        out.append(
            {
                "executor": row["executor"],
                "method": METHOD_LABELS.get(row["method"], row["method"]),
                "sweeps": str(int(float(row["iters"]))),
                "mean_success": fmt(float(row["overall_success"])),
                "task1_mean": fmt(float(row["task1_success"])),
                "task2_mean": fmt(float(row["task2_success"])),
                "task3_mean": fmt(float(row["task3_success"])),
                "task4_mean": fmt(float(row["task4_success"])),
                "task5_mean": fmt(float(row["task5_success"])),
            }
        )
    return fields, out


def build_teleport_stitch_screen() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "executor",
        "episodes_per_task",
        "controller_steps",
        "method",
        "sweeps",
        "mean_success",
        "task1_mean",
        "task2_mean",
        "task3_mean",
        "task4_mean",
        "task5_mean",
    ]
    if not TELEPORT_STITCH_SCREEN.exists():
        return fields, []
    return fields, read_rows(TELEPORT_STITCH_SCREEN)


def build_main_hard_task_results() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "executor",
        "eval_setting",
        "controller_steps",
        "matched_sweeps",
        "full_sweeps",
        "bellman_matched",
        "stochastic_trl",
        "bellman_full",
        "sto_minus_matched",
        "sto_equals_full",
    ]

    def summarize(path: Path) -> dict[str, tuple[float, int]]:
        if not path.exists():
            return {}
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in read_rows(path):
            grouped.setdefault(row["method"], []).append(row)
        out: dict[str, tuple[float, int]] = {}
        for method, rows in grouped.items():
            out[method] = (
                mean(float(row["overall_success"]) for row in rows),
                int(float(rows[0]["iters"])),
            )
        return out

    def row_from_summary(
        env: str,
        executor: str,
        eval_setting: str,
        controller_steps: str,
        summary: dict[str, tuple[float, int]],
    ) -> dict[str, str] | None:
        if not {"bellman_matched", "sto_trl_matched", "bellman_full"}.issubset(summary):
            return None
        matched, matched_sweeps = summary["bellman_matched"]
        sto, _sto_sweeps = summary["sto_trl_matched"]
        full, full_sweeps = summary["bellman_full"]
        return {
            "env": env,
            "executor": executor,
            "eval_setting": eval_setting,
            "controller_steps": controller_steps,
            "matched_sweeps": str(matched_sweeps),
            "full_sweeps": str(full_sweeps),
            "bellman_matched": fmt(matched),
            "stochastic_trl": fmt(sto),
            "bellman_full": fmt(full),
            "sto_minus_matched": fmt(sto - matched),
            "sto_equals_full": str(abs(sto - full) <= 1e-12),
        }

    specs = [
        (
            "pointmaze-teleport-navigate-v0",
            "dataset topology scaffold",
            "5 eval seeds, 50 episodes/task",
            "0",
            POINTMAZE_TOPOLOGY,
        ),
        (
            "pointmaze-teleport-stitch-v0",
            "dataset topology scaffold",
            "5 eval seeds, 50 episodes/task",
            "0",
            POINTMAZE_TOPOLOGY_STITCH,
        ),
        (
            "antmaze-teleport-navigate-v0",
            "full-goal BC + body-nearest k16",
            "3 eval seeds, 20 episodes/task",
            "50000",
            RESULTS / "antmaze_bc_topology_navigate_fullgoal_50k_ep20_seed012_bodyk16_cpu.csv",
        ),
        (
            "antmaze-teleport-stitch-v0",
            "full-goal BC + body-nearest k16",
            "3 eval seeds, 20 episodes/task",
            "20000",
            RESULTS / "antmaze_bc_topology_stitch_fullgoal_20k_ep20_seed012_bodyk16_cpu.csv",
        ),
    ]
    rows = []
    for env, executor, eval_setting, controller_steps, path in specs:
        row = row_from_summary(env, executor, eval_setting, controller_steps, summarize(path))
        if row is not None:
            rows.append(row)
    return fields, rows


def build_hard_task_stress_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "task_scope",
        "eval_setting",
        "controller_steps",
        "bellman_matched",
        "support_trl",
        "stochastic_trl",
        "bellman_full",
        "sto_minus_matched",
        "source",
    ]
    out: list[dict[str, str]] = []
    for env, task_scope, eval_setting, controller_steps, path in HARD_TASK_STRESS_SPECS:
        if not path.exists():
            continue
        summary: dict[str, float] = {}
        for row in read_rows(path):
            summary[row["method"]] = float(row["overall_success"])
        if "bellman_matched" not in summary or "sto_trl_matched" not in summary:
            continue
        out.append(
            {
                "env": env,
                "task_scope": task_scope,
                "eval_setting": eval_setting,
                "controller_steps": controller_steps,
                "bellman_matched": fmt(summary.get("bellman_matched")),
                "support_trl": fmt(summary.get("support_trl_matched")),
                "stochastic_trl": fmt(summary.get("sto_trl_matched")),
                "bellman_full": fmt(summary.get("bellman_full")),
                "sto_minus_matched": fmt(summary["sto_trl_matched"] - summary["bellman_matched"]),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return fields, out


def build_pointmaze_single_task_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "task_id",
        "eval_setting",
        "matched_sweeps",
        "full_sweeps",
        "bellman_matched",
        "stochastic_trl",
        "bellman_full",
        "sto_minus_matched",
        "sto_equals_full",
        "source",
    ]
    if not POINTMAZE_STITCH_TASK5.exists():
        return fields, []
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in read_rows(POINTMAZE_STITCH_TASK5):
        grouped.setdefault(row["method"], []).append(row)
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    if not required.issubset(grouped):
        return fields, []

    def method_mean(method: str) -> float:
        return mean(float(row["overall_success"]) for row in grouped[method])

    matched = method_mean("bellman_matched")
    sto = method_mean("sto_trl_matched")
    full = method_mean("bellman_full")
    rows = [
        {
            "env": "pointmaze-teleport-stitch-v0",
            "task_id": "5",
            "eval_setting": "5 eval seeds, 100 episodes",
            "matched_sweeps": str(int(float(grouped["bellman_matched"][0]["iters"]))),
            "full_sweeps": str(int(float(grouped["bellman_full"][0]["iters"]))),
            "bellman_matched": fmt(matched),
            "stochastic_trl": fmt(sto),
            "bellman_full": fmt(full),
            "sto_minus_matched": fmt(sto - matched),
            "sto_equals_full": str(abs(sto - full) <= 1e-12),
            "source": str(POINTMAZE_STITCH_TASK5.relative_to(ROOT)),
        }
    ]
    return fields, rows


def build_pointmaze_learned_transition_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "transition_model",
        "eval_setting",
        "transition_seeds",
        "matched_sweeps",
        "full_sweeps",
        "bellman_matched",
        "stochastic_trl",
        "bellman_full",
        "sto_minus_matched",
        "oracle_l1_range",
        "oracle_top1_range",
        "sources",
    ]
    out: list[dict[str, str]] = []
    for spec in POINTMAZE_LEARNED_TRANSITION_FILES:
        env = str(spec["env"])
        transition_model = str(spec["transition_model"])
        paths = list(spec["paths"])
        summaries: list[dict[str, dict[str, str]]] = []
        for path in paths:
            if not path.exists():
                continue
            by_method = {row["method"]: row for row in read_rows(path)}
            required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
            if required.issubset(by_method):
                summaries.append(by_method)
        if len(summaries) != len(paths):
            continue

        def mean_method(method: str) -> float:
            return mean(float(summary[method]["overall_success"]) for summary in summaries)

        matched = mean_method("bellman_matched")
        sto = mean_method("sto_trl_matched")
        full = mean_method("bellman_full")
        oracle_l1 = [float(summary["sto_trl_matched"].get("transition_oracle_l1", 0.0)) for summary in summaries]
        oracle_top1 = [float(summary["sto_trl_matched"].get("transition_oracle_top1", 0.0)) for summary in summaries]
        first = summaries[0]
        transition_seeds = ",".join(str(int(float(summary["sto_trl_matched"]["transition_seed"]))) for summary in summaries)
        eval_setting = (
            "seed 0, 50 episodes/task"
            if len(summaries) == 1
            else f"seed 0, {len(summaries)} transition seeds, 50 episodes/task"
        )
        out.append(
            {
                "env": env,
                "transition_model": transition_model,
                "eval_setting": eval_setting,
                "transition_seeds": transition_seeds,
                "matched_sweeps": str(int(float(first["bellman_matched"]["iters"]))),
                "full_sweeps": str(int(float(first["bellman_full"]["iters"]))),
                "bellman_matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "bellman_full": fmt(full),
                "sto_minus_matched": fmt(sto - matched),
                "oracle_l1_range": f"{fmt(min(oracle_l1))}-{fmt(max(oracle_l1))}",
                "oracle_top1_range": f"{fmt(min(oracle_top1))}-{fmt(max(oracle_top1))}",
                "sources": ";".join(str(path.relative_to(ROOT)) for path in paths),
            }
        )
    return fields, out


def build_pointmaze_bc_controller_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "controller",
        "eval_setting",
        "eval_seeds",
        "controller_steps",
        "matched_sweeps",
        "full_sweeps",
        "bellman_matched",
        "stochastic_trl",
        "bellman_full",
        "sto_minus_matched",
        "sources",
    ]
    out: list[dict[str, str]] = []
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for spec in POINTMAZE_BC_CONTROLLER_FILES:
        env = str(spec["env"])
        controller = str(spec["controller"])
        paths = list(spec["paths"])
        rows: list[dict[str, str]] = []
        for path in paths:
            if not path.exists():
                continue
            rows.extend(read_rows(path))
        if not rows:
            continue
        by_method: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_method.setdefault(row["method"], []).append(row)
        if not required.issubset(by_method):
            continue

        def mean_method(method: str) -> float:
            return mean(float(row["overall_success"]) for row in by_method[method])

        matched = mean_method("bellman_matched")
        sto = mean_method("sto_trl_matched")
        full = mean_method("bellman_full")
        eval_seeds = ",".join(
            str(int(float(seed))) for seed in sorted({row["seed"] for row in rows}, key=lambda x: int(float(x)))
        )
        first_matched = by_method["bellman_matched"][0]
        first_full = by_method["bellman_full"][0]
        out.append(
            {
                "env": env,
                "controller": controller,
                "eval_setting": "3 eval seeds, 20 episodes/task",
                "eval_seeds": eval_seeds,
                "controller_steps": str(int(float(first_matched["bc_steps"]))),
                "matched_sweeps": str(int(float(first_matched["iters"]))),
                "full_sweeps": str(int(float(first_full["iters"]))),
                "bellman_matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "bellman_full": fmt(full),
                "sto_minus_matched": fmt(sto - matched),
                "sources": ";".join(str(path.relative_to(ROOT)) for path in paths),
            }
        )
    return fields, out


def build_controller_execution_isolation(
    pointmaze_bc_rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "execution_path",
        "env",
        "controller_or_actor",
        "eval_setting",
        "best_success",
        "final_success",
        "source",
    ]
    out: list[dict[str, str]] = []

    for spec in POINTMAZE_DIRECT_ACTOR_FILES:
        path = Path(spec["path"])
        if not path.exists():
            continue
        rows = read_rows(path)
        if not rows:
            continue
        values = [float(row["evaluation/overall_success"]) for row in rows]
        out.append(
            {
                "execution_path": str(spec["execution_path"]),
                "env": str(spec["env"]),
                "controller_or_actor": str(spec["controller_or_actor"]),
                "eval_setting": str(spec["eval_setting"]),
                "best_success": fmt(max(values)),
                "final_success": fmt(values[-1]),
                "source": str(path.relative_to(ROOT)),
            }
        )

    for row in pointmaze_bc_rows:
        out.append(
            {
                "execution_path": "Stochastic TRL + learned waypoint BC",
                "env": row["env"],
                "controller_or_actor": row["controller"],
                "eval_setting": row["eval_setting"],
                "best_success": row["stochastic_trl"],
                "final_success": row["stochastic_trl"],
                "source": row["sources"],
            }
        )
    return fields, out


def build_fast_eval_profile_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "screen",
        "role",
        "env",
        "task_ids",
        "seed",
        "episodes_per_task",
        "eval_action_repeat",
        "matched_success",
        "stochastic_trl",
        "sto_minus_matched",
        "setup_seconds",
        "matched_eval_seconds",
        "sto_eval_seconds",
        "sto_env_step_ms",
        "sto_action_ms_per_call",
        "source",
    ]
    out: list[dict[str, str]] = []
    for spec in FAST_EVAL_PROFILE_FILES:
        path = Path(spec["path"])
        if not path.exists():
            continue
        rows = read_rows(path)
        by_method = {row["method"]: row for row in rows}
        if "bellman_matched" not in by_method or "sto_trl_matched" not in by_method:
            continue
        matched = by_method["bellman_matched"]
        sto = by_method["sto_trl_matched"]
        matched_success = float(matched["overall_success"])
        sto_success = float(sto["overall_success"])
        out.append(
            {
                "screen": str(spec["screen"]),
                "role": str(spec["role"]),
                "env": sto["env"],
                "task_ids": sto["task_ids"],
                "seed": sto["seed"],
                "episodes_per_task": sto["episodes_per_task"],
                "eval_action_repeat": sto["eval_action_repeat"],
                "matched_success": fmt(matched_success),
                "stochastic_trl": fmt(sto_success),
                "sto_minus_matched": fmt(sto_success - matched_success),
                "setup_seconds": fmt(float(sto["setup_seconds"]), digits=2),
                "matched_eval_seconds": fmt(float(matched["eval_seconds"]), digits=2),
                "sto_eval_seconds": fmt(float(sto["eval_seconds"]), digits=2),
                "sto_env_step_ms": fmt(float(sto["profile_env_step_ms"]), digits=3),
                "sto_action_ms_per_call": fmt(float(sto["profile_action_ms_per_call"]), digits=3),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return fields, out


def build_antmaze_support_ablation_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "controller",
        "eval_setting",
        "matched_sweeps",
        "full_sweeps",
        "bellman_matched",
        "support_trl",
        "stochastic_trl",
        "bellman_full",
        "support_minus_matched",
        "sto_minus_support",
        "sto_eval_seconds",
        "source",
    ]
    out: list[dict[str, str]] = []
    required = {"bellman_matched", "support_trl_matched", "sto_trl_matched", "bellman_full"}
    for spec in ANTMAZE_SUPPORT_ABLATION_FILES:
        path = Path(spec["path"])
        if not path.exists():
            continue
        by_method = {row["method"]: row for row in read_rows(path)}
        if not required.issubset(by_method):
            continue
        matched = float(by_method["bellman_matched"]["overall_success"])
        support = float(by_method["support_trl_matched"]["overall_success"])
        sto = float(by_method["sto_trl_matched"]["overall_success"])
        full = float(by_method["bellman_full"]["overall_success"])
        sto_row = by_method["sto_trl_matched"]
        out.append(
            {
                "env": str(spec["env"]),
                "controller": str(spec["controller"]),
                "eval_setting": f"seed {sto_row['seed']}, tasks {sto_row['task_ids']}, "
                f"{sto_row['episodes_per_task']} episodes/task",
                "matched_sweeps": str(int(float(sto_row["iters"]))),
                "full_sweeps": str(int(float(by_method["bellman_full"]["iters"]))),
                "bellman_matched": fmt(matched),
                "support_trl": fmt(support),
                "stochastic_trl": fmt(sto),
                "bellman_full": fmt(full),
                "support_minus_matched": fmt(support - matched),
                "sto_minus_support": fmt(sto - support),
                "sto_eval_seconds": fmt(float(sto_row["eval_seconds"]), digits=2),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return fields, out


def build_pointmaze_tie_policy_head_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "transition_model",
        "control_head",
        "eval_setting",
        "eval_seed",
        "matched_sweeps",
        "full_sweeps",
        "bellman_matched",
        "stochastic_trl",
        "bellman_full",
        "sto_minus_matched",
        "transition_top1",
        "value_action_agreement",
        "source",
    ]
    out: list[dict[str, str]] = []
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for path in POINTMAZE_TIE_POLICY_HEAD_FILES:
        if not path.exists():
            continue
        by_method = {row["method"]: row for row in read_rows(path)}
        if not required.issubset(by_method):
            continue

        matched = float(by_method["bellman_matched"]["overall_success"])
        sto = float(by_method["sto_trl_matched"]["overall_success"])
        full = float(by_method["bellman_full"]["overall_success"])
        sto_row = by_method["sto_trl_matched"]
        out.append(
            {
                "env": sto_row["env"],
                "transition_model": "raw-observation MLP cell-change",
                "control_head": "raw-observation tie-policy MLP",
                "eval_setting": "seed 0, all tasks, 20 episodes/task",
                "eval_seed": str(int(float(sto_row["seed"]))),
                "matched_sweeps": str(int(float(by_method["bellman_matched"]["iters"]))),
                "full_sweeps": str(int(float(by_method["bellman_full"]["iters"]))),
                "bellman_matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "bellman_full": fmt(full),
                "sto_minus_matched": fmt(sto - matched),
                "transition_top1": fmt(float(sto_row["transition_top1"])),
                "value_action_agreement": fmt(float(sto_row["value_action_agreement"])),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return fields, out


def build_pointmaze_prev_policy_head_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "transition_model",
        "control_head",
        "eval_setting",
        "eval_seed",
        "value_steps",
        "matched_sweeps",
        "full_sweeps",
        "bellman_matched",
        "stochastic_trl",
        "bellman_full",
        "sto_minus_matched",
        "transition_top1",
        "value_action_agreement",
        "source",
    ]
    out: list[dict[str, str]] = []
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for path in POINTMAZE_PREV_POLICY_HEAD_FILES:
        if not path.exists():
            continue
        by_method = {row["method"]: row for row in read_rows(path)}
        if not required.issubset(by_method):
            continue

        matched = float(by_method["bellman_matched"]["overall_success"])
        sto = float(by_method["sto_trl_matched"]["overall_success"])
        full = float(by_method["bellman_full"]["overall_success"])
        sto_row = by_method["sto_trl_matched"]
        out.append(
            {
                "env": sto_row["env"],
                "transition_model": "raw-observation MLP cell-change",
                "control_head": "previous-action policy MLP",
                "eval_setting": "seed 0, all tasks, 20 episodes/task",
                "eval_seed": str(int(float(sto_row["seed"]))),
                "value_steps": str(int(float(sto_row["value_steps"]))),
                "matched_sweeps": str(int(float(by_method["bellman_matched"]["iters"]))),
                "full_sweeps": str(int(float(by_method["bellman_full"]["iters"]))),
                "bellman_matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "bellman_full": fmt(full),
                "sto_minus_matched": fmt(sto - matched),
                "transition_top1": fmt(float(sto_row["transition_top1"])),
                "value_action_agreement": fmt(float(sto_row["value_action_agreement"])),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return fields, out


def build_pointmaze_tie_policy_head_eval_seed_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "transition_model",
        "control_head",
        "eval_setting",
        "eval_seeds",
        "matched_sweeps",
        "full_sweeps",
        "bellman_matched",
        "stochastic_trl",
        "bellman_full",
        "sto_minus_matched",
        "transition_top1",
        "value_action_agreement_min",
        "sources",
    ]
    out: list[dict[str, str]] = []
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for spec in POINTMAZE_TIE_POLICY_HEAD_EVAL_SEED_FILES:
        env = str(spec["env"])
        paths = list(spec["paths"])
        rows: list[dict[str, str]] = []
        for path in paths:
            if path.exists():
                rows.extend(read_rows(path))
        if not rows:
            continue
        by_method: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_method.setdefault(row["method"], []).append(row)
        if not required.issubset(by_method):
            continue

        def mean_method(method: str) -> float:
            return mean(float(row["overall_success"]) for row in by_method[method])

        matched = mean_method("bellman_matched")
        sto = mean_method("sto_trl_matched")
        full = mean_method("bellman_full")
        sto_rows = by_method["sto_trl_matched"]
        eval_seeds = ",".join(
            str(int(float(seed))) for seed in sorted({row["seed"] for row in rows}, key=lambda x: int(float(x)))
        )
        out.append(
            {
                "env": env,
                "transition_model": "raw-observation MLP cell-change",
                "control_head": "raw-observation tie-policy MLP",
                "eval_setting": "3 eval seeds, all tasks, 20 episodes/task",
                "eval_seeds": eval_seeds,
                "matched_sweeps": str(int(float(by_method["bellman_matched"][0]["iters"]))),
                "full_sweeps": str(int(float(by_method["bellman_full"][0]["iters"]))),
                "bellman_matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "bellman_full": fmt(full),
                "sto_minus_matched": fmt(sto - matched),
                "transition_top1": fmt(min(float(row["transition_top1"]) for row in sto_rows)),
                "value_action_agreement_min": fmt(min(float(row["value_action_agreement"]) for row in sto_rows)),
                "sources": ";".join(str(path.relative_to(ROOT)) for path in paths),
            }
        )
    return fields, out


def build_pointmaze_tie_policy_head_transition_seed_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "transition_model",
        "control_head",
        "eval_setting",
        "eval_seed",
        "transition_seeds",
        "matched_sweeps",
        "full_sweeps",
        "bellman_matched",
        "stochastic_trl",
        "bellman_full",
        "sto_minus_matched",
        "transition_top1_min",
        "value_action_agreement_min",
        "sources",
    ]
    out: list[dict[str, str]] = []
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for spec in POINTMAZE_TIE_POLICY_HEAD_TRANSITION_SEED_FILES:
        env = str(spec["env"])
        paths = list(spec["paths"])
        summaries: list[dict[str, dict[str, str]]] = []
        for path in paths:
            if not path.exists():
                continue
            by_method = {row["method"]: row for row in read_rows(path)}
            if required.issubset(by_method):
                summaries.append(by_method)
        if len(summaries) != len(paths):
            continue

        def mean_method(method: str) -> float:
            return mean(float(summary[method]["overall_success"]) for summary in summaries)

        matched = mean_method("bellman_matched")
        sto = mean_method("sto_trl_matched")
        full = mean_method("bellman_full")
        sto_rows = [summary["sto_trl_matched"] for summary in summaries]
        transition_seeds = ",".join(
            str(int(float(row["transition_seed"])))
            for row in sorted(sto_rows, key=lambda row: int(float(row["transition_seed"])))
        )
        out.append(
            {
                "env": env,
                "transition_model": "raw-observation MLP cell-change",
                "control_head": "raw-observation tie-policy MLP",
                "eval_setting": "seed 0, 3 transition seeds, all tasks, 20 episodes/task",
                "eval_seed": str(int(float(sto_rows[0]["seed"]))),
                "transition_seeds": transition_seeds,
                "matched_sweeps": str(int(float(summaries[0]["bellman_matched"]["iters"]))),
                "full_sweeps": str(int(float(summaries[0]["bellman_full"]["iters"]))),
                "bellman_matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "bellman_full": fmt(full),
                "sto_minus_matched": fmt(sto - matched),
                "transition_top1_min": fmt(min(float(row["transition_top1"]) for row in sto_rows)),
                "value_action_agreement_min": fmt(min(float(row["value_action_agreement"]) for row in sto_rows)),
                "sources": ";".join(str(path.relative_to(ROOT)) for path in paths),
            }
        )
    return fields, out


def build_antmaze_tie_policy_head_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "controller",
        "control_head",
        "eval_setting",
        "eval_seed",
        "transition_seeds",
        "task_ids",
        "controller_steps",
        "matched_sweeps",
        "full_sweeps",
        "bellman_matched",
        "stochastic_trl",
        "bellman_full",
        "sto_minus_matched",
        "value_action_agreement",
        "value_steps",
        "source",
    ]
    out: list[dict[str, str]] = []
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for spec in ANTMAZE_TIE_POLICY_HEAD_FILES:
        path = Path(spec["path"])
        if not path.exists():
            continue
        by_method = {row["method"]: row for row in read_rows(path)}
        if not required.issubset(by_method):
            continue

        matched = float(by_method["bellman_matched"]["overall_success"])
        sto = float(by_method["sto_trl_matched"]["overall_success"])
        full = float(by_method["bellman_full"]["overall_success"])
        sto_row = by_method["sto_trl_matched"]
        out.append(
            {
                "env": str(spec["env"]),
                "controller": str(spec["controller"]),
                "control_head": "raw-observation tie-policy MLP",
                "eval_setting": "seed 0, tasks 4-5, 20 episodes/task",
                "eval_seed": str(int(float(sto_row["seed"]))),
                "task_ids": sto_row["task_ids"],
                "controller_steps": str(int(float(sto_row["bc_steps"]))),
                "matched_sweeps": str(int(float(by_method["bellman_matched"]["iters"]))),
                "full_sweeps": str(int(float(by_method["bellman_full"]["iters"]))),
                "bellman_matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "bellman_full": fmt(full),
                "sto_minus_matched": fmt(sto - matched),
                "value_action_agreement": fmt(float(sto_row["value_action_agreement"])),
                "value_steps": str(int(float(sto_row["value_steps"]))),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return fields, out


def build_antmaze_rawobs_tie_policy_head_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "controller",
        "transition_model",
        "control_head",
        "eval_setting",
        "eval_seed",
        "transition_seeds",
        "task_ids",
        "controller_steps",
        "matched_sweeps",
        "full_sweeps",
        "bellman_matched",
        "stochastic_trl",
        "bellman_full",
        "sto_minus_matched",
        "transition_oracle_l1_max",
        "transition_oracle_top1_min",
        "value_action_agreement_min",
        "sources",
    ]
    out: list[dict[str, str]] = []
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for spec in ANTMAZE_RAWOBS_TIE_POLICY_HEAD_FILES:
        paths = list(spec["paths"])
        summaries: list[dict[str, dict[str, str]]] = []
        for path in paths:
            if not path.exists():
                continue
            by_method = {row["method"]: row for row in read_rows(path)}
            if required.issubset(by_method):
                summaries.append(by_method)
        if len(summaries) != len(paths):
            continue

        def mean_method(method: str) -> float:
            return mean(float(summary[method]["overall_success"]) for summary in summaries)

        matched = mean_method("bellman_matched")
        sto = mean_method("sto_trl_matched")
        full = mean_method("bellman_full")
        first = summaries[0]
        transition_seeds = ",".join(
            str(int(float(summary["sto_trl_matched"]["transition_seed"]))) for summary in summaries
        )
        transition_oracle_l1 = [
            float(summary["sto_trl_matched"]["transition_oracle_l1"]) for summary in summaries
        ]
        transition_oracle_top1 = [
            float(summary["sto_trl_matched"]["transition_oracle_top1"]) for summary in summaries
        ]
        value_action_agreement = [
            float(summary["sto_trl_matched"]["value_action_agreement"]) for summary in summaries
        ]
        out.append(
            {
                "env": str(spec["env"]),
                "controller": str(spec["controller"]),
                "transition_model": "raw-observation MLP jump-change",
                "control_head": "raw-observation tie-policy MLP",
                "eval_setting": "seed 0, tasks 4-5, 10 episodes/task",
                "eval_seed": str(int(float(first["sto_trl_matched"]["seed"]))),
                "transition_seeds": transition_seeds,
                "task_ids": first["sto_trl_matched"]["task_ids"],
                "controller_steps": str(int(float(first["sto_trl_matched"]["bc_steps"]))),
                "matched_sweeps": str(int(float(first["bellman_matched"]["iters"]))),
                "full_sweeps": str(int(float(first["bellman_full"]["iters"]))),
                "bellman_matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "bellman_full": fmt(full),
                "sto_minus_matched": fmt(sto - matched),
                "transition_oracle_l1_max": fmt(max(transition_oracle_l1)),
                "transition_oracle_top1_min": fmt(min(transition_oracle_top1)),
                "value_action_agreement_min": fmt(min(value_action_agreement)),
                "sources": ";".join(str(path.relative_to(ROOT)) for path in paths),
            }
        )
    return fields, out


def build_antmaze_rawobs_tie_policy_eval_seed_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "controller",
        "transition_model",
        "control_head",
        "eval_setting",
        "eval_seeds",
        "transition_seed",
        "task_ids",
        "controller_steps",
        "matched_sweeps",
        "full_sweeps",
        "bellman_matched",
        "stochastic_trl",
        "bellman_full",
        "sto_minus_matched",
        "transition_oracle_top1",
        "value_action_agreement_min",
        "sources",
    ]
    out: list[dict[str, str]] = []
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for spec in ANTMAZE_RAWOBS_TIE_POLICY_EVAL_SEED_FILES:
        paths = list(spec["paths"])
        rows: list[dict[str, str]] = []
        for path in paths:
            if path.exists():
                rows.extend(read_rows(path))
        if not rows:
            continue
        by_method: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_method.setdefault(row["method"], []).append(row)
        if not required.issubset(by_method):
            continue

        def mean_method(method: str) -> float:
            return mean(float(row["overall_success"]) for row in by_method[method])

        matched = mean_method("bellman_matched")
        sto = mean_method("sto_trl_matched")
        full = mean_method("bellman_full")
        first = by_method["sto_trl_matched"][0]
        eval_seeds = ",".join(
            str(int(float(seed))) for seed in sorted({row["seed"] for row in rows}, key=lambda x: int(float(x)))
        )
        value_action_agreement = [float(row["value_action_agreement"]) for row in by_method["sto_trl_matched"]]
        out.append(
            {
                "env": str(spec["env"]),
                "controller": str(spec["controller"]),
                "transition_model": "raw-observation MLP jump-change",
                "control_head": "raw-observation tie-policy MLP",
                "eval_setting": "3 eval seeds, tasks 4-5, 10 episodes/task",
                "eval_seeds": eval_seeds,
                "transition_seed": str(int(float(first["transition_seed"]))),
                "task_ids": first["task_ids"],
                "controller_steps": str(int(float(first["bc_steps"]))),
                "matched_sweeps": str(int(float(by_method["bellman_matched"][0]["iters"]))),
                "full_sweeps": str(int(float(by_method["bellman_full"][0]["iters"]))),
                "bellman_matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "bellman_full": fmt(full),
                "sto_minus_matched": fmt(sto - matched),
                "transition_oracle_top1": fmt(float(first["transition_oracle_top1"])),
                "value_action_agreement_min": fmt(min(value_action_agreement)),
                "sources": ";".join(str(path.relative_to(ROOT)) for path in paths),
            }
        )
    return fields, out


def build_antmaze_rawobs_prev_policy_eval_seed_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "controller",
        "transition_model",
        "control_head",
        "eval_setting",
        "eval_seeds",
        "transition_seed",
        "task_ids",
        "controller_steps",
        "matched_sweeps",
        "full_sweeps",
        "bellman_matched",
        "stochastic_trl",
        "bellman_full",
        "sto_minus_matched",
        "transition_oracle_top1",
        "value_action_agreement_min",
        "sources",
    ]
    out: list[dict[str, str]] = []
    required = {"bellman_matched", "sto_trl_matched", "bellman_full"}
    for spec in ANTMAZE_RAWOBS_PREV_POLICY_EVAL_SEED_FILES:
        paths = list(spec["paths"])
        rows: list[dict[str, str]] = []
        for path in paths:
            if path.exists():
                rows.extend(read_rows(path))
        if not rows:
            continue
        by_method: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_method.setdefault(row["method"], []).append(row)
        if not required.issubset(by_method):
            continue

        def mean_method(method: str) -> float:
            return mean(float(row["overall_success"]) for row in by_method[method])

        matched = mean_method("bellman_matched")
        sto = mean_method("sto_trl_matched")
        full = mean_method("bellman_full")
        first = by_method["sto_trl_matched"][0]
        eval_seeds = ",".join(
            str(int(float(seed))) for seed in sorted({row["seed"] for row in rows}, key=lambda x: int(float(x)))
        )
        value_action_agreement = [float(row["value_action_agreement"]) for row in by_method["sto_trl_matched"]]
        out.append(
            {
                "env": str(spec["env"]),
                "controller": str(spec["controller"]),
                "transition_model": "raw-observation MLP jump-change",
                "control_head": "previous-action policy MLP",
                "eval_setting": f"3 eval seeds, tasks 4-5, {int(float(first['episodes_per_task']))} episodes/task",
                "eval_seeds": eval_seeds,
                "transition_seed": str(int(float(first["transition_seed"]))),
                "task_ids": first["task_ids"],
                "controller_steps": str(int(float(first["bc_steps"]))),
                "matched_sweeps": str(int(float(by_method["bellman_matched"][0]["iters"]))),
                "full_sweeps": str(int(float(by_method["bellman_full"][0]["iters"]))),
                "bellman_matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "bellman_full": fmt(full),
                "sto_minus_matched": fmt(sto - matched),
                "transition_oracle_top1": fmt(float(first["transition_oracle_top1"])),
                "value_action_agreement_min": fmt(min(value_action_agreement)),
                "sources": ";".join(str(path.relative_to(ROOT)) for path in paths),
            }
        )
    return fields, out


def build_antmaze_learned_transition_summary() -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "env",
        "controller",
        "transition_model",
        "eval_setting",
        "transition_seeds",
        "matched_sweeps",
        "full_sweeps",
        "bellman_matched",
        "stochastic_trl",
        "bellman_full",
        "sto_minus_matched",
        "oracle_l1_range",
        "oracle_top1_range",
        "sources",
    ]
    out: list[dict[str, str]] = []
    for spec in ANTMAZE_LEARNED_TRANSITION_FILES:
        env = str(spec["env"])
        controller = str(spec["controller"])
        transition_model = str(spec["transition_model"])
        paths = list(spec["paths"])
        summaries: list[dict[str, dict[str, str]]] = []
        for path in paths:
            if not path.exists():
                continue
            by_method = {row["method"]: row for row in read_rows(path)}
            if {"bellman_matched", "sto_trl_matched", "bellman_full"}.issubset(by_method):
                summaries.append(by_method)
        if len(summaries) != len(paths):
            continue

        def mean_method(method: str) -> float:
            return mean(float(summary[method]["overall_success"]) for summary in summaries)

        matched = mean_method("bellman_matched")
        sto = mean_method("sto_trl_matched")
        full = mean_method("bellman_full")
        oracle_l1 = [float(summary["sto_trl_matched"]["transition_oracle_l1"]) for summary in summaries]
        oracle_top1 = [float(summary["sto_trl_matched"]["transition_oracle_top1"]) for summary in summaries]
        first = summaries[0]
        transition_seeds = ",".join(str(int(float(summary["sto_trl_matched"]["transition_seed"]))) for summary in summaries)
        out.append(
            {
                "env": env,
                "controller": controller,
                "transition_model": transition_model,
                "eval_setting": "seed 0, tasks 4-5, 10 episodes/task",
                "transition_seeds": transition_seeds,
                "matched_sweeps": str(int(float(first["bellman_matched"]["iters"]))),
                "full_sweeps": str(int(float(first["bellman_full"]["iters"]))),
                "bellman_matched": fmt(matched),
                "stochastic_trl": fmt(sto),
                "bellman_full": fmt(full),
                "sto_minus_matched": fmt(sto - matched),
                "oracle_l1_range": f"{fmt(min(oracle_l1))}-{fmt(max(oracle_l1))}",
                "oracle_top1_range": f"{fmt(min(oracle_top1))}-{fmt(max(oracle_top1))}",
                "sources": ";".join(str(path.relative_to(ROOT)) for path in paths),
            }
        )
    return fields, out


def write_report(
    main_fields: list[str],
    main_rows: list[dict[str, str]],
    hard_stress_fields: list[str],
    hard_stress_rows: list[dict[str, str]],
    pointmaze_single_fields: list[str],
    pointmaze_single_rows: list[dict[str, str]],
    pointmaze_lt_fields: list[str],
    pointmaze_lt_rows: list[dict[str, str]],
    pointmaze_bc_fields: list[str],
    pointmaze_bc_rows: list[dict[str, str]],
    controller_iso_fields: list[str],
    controller_iso_rows: list[dict[str, str]],
    fast_eval_fields: list[str],
    fast_eval_rows: list[dict[str, str]],
    antmaze_support_fields: list[str],
    antmaze_support_rows: list[dict[str, str]],
    pointmaze_tie_fields: list[str],
    pointmaze_tie_rows: list[dict[str, str]],
    pointmaze_prev_fields: list[str],
    pointmaze_prev_rows: list[dict[str, str]],
    pointmaze_tie_eval_fields: list[str],
    pointmaze_tie_eval_rows: list[dict[str, str]],
    pointmaze_tie_transition_fields: list[str],
    pointmaze_tie_transition_rows: list[dict[str, str]],
    antmaze_tie_fields: list[str],
    antmaze_tie_rows: list[dict[str, str]],
    antmaze_rawobs_tie_fields: list[str],
    antmaze_rawobs_tie_rows: list[dict[str, str]],
    antmaze_rawobs_eval_fields: list[str],
    antmaze_rawobs_eval_rows: list[dict[str, str]],
    antmaze_rawobs_prev_eval_fields: list[str],
    antmaze_rawobs_prev_eval_rows: list[dict[str, str]],
    antmaze_lt_fields: list[str],
    antmaze_lt_rows: list[dict[str, str]],
    tabular_fields: list[str],
    tabular_rows: list[dict[str, str]],
    horizon_fields: list[str],
    horizon_rows: list[dict[str, str]],
    aggregate_fields: list[str],
    aggregate_rows: list[dict[str, str]],
    grid_fields: list[str],
    grid_rows: list[dict[str, str]],
    grid_realized_fields: list[str],
    grid_realized_rows: list[dict[str, str]],
    budget_fields: list[str],
    budget_rows: list[dict[str, str]],
    graph_fields: list[str],
    graph_rows: list[dict[str, str]],
    graph_stats_fields: list[str],
    graph_stats_rows: list[dict[str, str]],
    graph_task_fields: list[str],
    graph_task_rows: list[dict[str, str]],
    topo_fields: list[str],
    topo_rows: list[dict[str, str]],
    topo_stitch_fields: list[str],
    topo_stitch_rows: list[dict[str, str]],
    topo_stitch_support_fields: list[str],
    topo_stitch_support_rows: list[dict[str, str]],
    topo_stats_fields: list[str],
    topo_stats_rows: list[dict[str, str]],
    topo_stitch_stats_fields: list[str],
    topo_stitch_stats_rows: list[dict[str, str]],
    ant_fields: list[str],
    ant_rows: list[dict[str, str]],
    ant_nav_body_fields: list[str],
    ant_nav_body_rows: list[dict[str, str]],
    ant_multi_fields: list[str],
    ant_multi_rows: list[dict[str, str]],
    ant_bcseed_fields: list[str],
    ant_bcseed_rows: list[dict[str, str]],
    ant_stitch_seed_fields: list[str],
    ant_stitch_seed_rows: list[dict[str, str]],
    ant_nav_seed_fields: list[str],
    ant_nav_seed_rows: list[dict[str, str]],
    ant_ep20_fields: list[str],
    ant_ep20_rows: list[dict[str, str]],
    ant_ep20_stats_fields: list[str],
    ant_ep20_stats_rows: list[dict[str, str]],
    ant_budget_fields: list[str],
    ant_budget_rows: list[dict[str, str]],
    ant_exec_fields: list[str],
    ant_exec_rows: list[dict[str, str]],
    stitch_fields: list[str],
    stitch_rows: list[dict[str, str]],
) -> None:
    safe_tabular = [
        row
        for row in tabular_rows
        if row["p_success"] in {"0.02", "0.05"}
        and row["method"] in {"MC positive", "Bellman matched", "Stochastic TRL"}
    ]
    report = [
        "# Paper Artifacts",
        "",
        "Generated from the current stochastic TRL prototype results.",
        "",
        "## Artifact Files",
        "",
        "- `results/main_claim_verification.md`",
        "- `results/paper_tables/main_hard_task_results.csv`",
        "- `results/paper_tables/hard_task_stress_seed0.csv`",
        "- `results/paper_tables/pointmaze_stitch_task5_ep100_seed01234.csv`",
        "- `results/paper_tables/pointmaze_learned_transition.csv`",
        "- `results/paper_tables/pointmaze_learned_controller_ep20_seed012.csv`",
        "- `results/paper_tables/controller_execution_isolation.csv`",
        "- `results/paper_tables/fast_eval_profile.csv`",
        "- `results/paper_tables/antmaze_support_ablation_ep5_seed0_task45.csv`",
        "- `results/paper_tables/pointmaze_tie_policy_head_ep20_seed0.csv`",
        "- `results/paper_tables/pointmaze_rawobs_transition_prev_policy_head_ep20_seed0.csv`",
        "- `results/paper_tables/pointmaze_tie_policy_head_ep20_evalseed012.csv`",
        "- `results/paper_tables/pointmaze_tie_policy_head_ep20_tseed012.csv`",
        "- `results/paper_tables/antmaze_tie_policy_head_hard_tasks_ep20_seed0.csv`",
        "- `results/paper_tables/antmaze_rawobs_transition_tie_policy_head_ep10_tseed012.csv`",
        "- `results/paper_tables/antmaze_rawobs_transition_tie_policy_head_ep10_evalseed012.csv`",
        "- `results/paper_tables/antmaze_rawobs_transition_prev_policy_head_ep10_evalseed012.csv`",
        "- `results/paper_tables/antmaze_learned_transition_robustness.csv`",
        "- `results/paper_tables/tabular_l128.csv`",
        "- `results/paper_tables/tabular_safe_horizon.csv`",
        "- `results/paper_tables/tabular_risky_aggregate.csv`",
        "- `results/paper_tables/grid_shortcut_2d.csv`",
        "- `results/paper_tables/grid_realized_diagnostic.csv`",
        "- `results/paper_tables/grid_budget_curve.csv`",
        "- `results/paper_tables/pointmaze_graph_5seed.csv`",
        "- `results/paper_tables/pointmaze_graph_paired_stats.csv`",
        "- `results/paper_tables/pointmaze_graph_task_deltas.csv`",
        "- `results/paper_tables/pointmaze_topology_5seed.csv`",
        "- `results/paper_tables/pointmaze_topology_stitch_5seed.csv`",
        "- `results/paper_tables/pointmaze_topology_stitch_support_baseline_5seed.csv`",
        "- `results/paper_tables/pointmaze_topology_paired_stats.csv`",
        "- `results/paper_tables/pointmaze_topology_stitch_paired_stats.csv`",
        "- `results/paper_tables/antmaze_bc_topology_20k_ep3_seed0.csv`",
        "- `results/paper_tables/antmaze_navigate_50k_bodyk16_ep10_seed0.csv`",
        "- `results/paper_tables/antmaze_bodyk16_multiseed_ep5.csv`",
        "- `results/paper_tables/antmaze_bcseed1_ep20_seed012.csv`",
        "- `results/paper_tables/antmaze_stitch_controller_seeds_ep20_seed012.csv`",
        "- `results/paper_tables/antmaze_navigate_controller_seeds_ep20_seed012.csv`",
        "- `results/paper_tables/antmaze_bodyk16_ep20_seed012.csv`",
        "- `results/paper_tables/antmaze_bodyk16_ep20_seed012_paired_stats.csv`",
        "- `results/paper_tables/antmaze_budget_ep3_seed0.csv`",
        "- `results/paper_tables/antmaze_stitch_executor_ablation_ep3_seed0.csv`",
        "- `results/paper_tables/teleport_stitch_screen_seed0.csv`",
        "- `results/figures/tabular_l128_safe_success.svg`",
        "- `results/figures/tabular_horizon_success.svg`",
        "- `results/figures/grid_shortcut_success.svg`",
        "- `results/figures/grid_budget_curve.svg`",
        "- `results/figures/pointmaze_graph_success.svg`",
        "- `results/figures/antmaze_bodyk16_multiseed.svg`",
        "- `results/figures/antmaze_budget.svg`",
        "",
        "## Main Hard-Task Results",
        "",
        markdown_table(main_fields, main_rows),
        "",
        "Main signal: on both PointMaze and AntMaze teleport long-horizon tasks, stochastic TRL reaches the 180-sweep Bellman reference at the matched 6-sweep budget, while matched Bellman remains substantially lower. PointMaze rows use five evaluation seeds; AntMaze rows use the learned BC executor with three evaluation seeds and 20 episodes per task.",
        "",
        "## Hard-Task Stress Checks",
        "",
        markdown_table(hard_stress_fields, hard_stress_rows),
        "",
        "Stress-check signal: on a single eval seed focused on harder long-horizon task slices, stochastic TRL reaches 0.93 on PointMaze teleport stitch tasks 4-5, 0.95 on AntMaze navigate tasks 4-5, and 1.00 on AntMaze stitch tasks 4-5. These rows are not a replacement for the main multi-seed table; they are fast iteration evidence for the hardest task slices.",
        "",
        "## Focused PointMaze Single-Task Check",
        "",
        markdown_table(pointmaze_single_fields, pointmaze_single_rows),
        "",
        "Focused signal: on PointMaze teleport stitch task 5, stochastic TRL reaches 0.908 success over five evaluation seeds and 100 episodes per seed, matching the 180-sweep Bellman reference while matched Bellman reaches 0.380. This is appendix evidence, not a replacement for the all-task headline table.",
        "",
        "## PointMaze Learned-Transition Screen",
        "",
        markdown_table(pointmaze_lt_fields, pointmaze_lt_rows),
        "",
        "Learned-transition signal: fitting either a table-softmax transition model or a shared raw-observation MLP transition head from collapsed offline cell changes preserves the stochastic TRL gain on PointMaze teleport navigate and stitch. Stochastic TRL reaches 0.916 success with the 6-sweep matched budget in both cases; matched Bellman reaches 0.408 with the table-softmax model and 0.512/0.483 on navigate/stitch with the raw-observation MLP.",
        "",
        "## PointMaze Learned-Controller Screen",
        "",
        markdown_table(pointmaze_bc_fields, pointmaze_bc_rows),
        "",
        "Learned-controller signal: with saved 5k-step full-goal BC executors and body-nearest waypoint goals, stochastic TRL reaches 1.000 success on both PointMaze teleport navigate and stitch over three evaluation seeds, matching full Bellman while matched Bellman remains far lower. This is appendix evidence that the PointMaze result is not tied to the simple topology executor.",
        "",
        "### Controller Execution Isolation",
        "",
        markdown_table(controller_iso_fields, controller_iso_rows),
        "",
        "Controller-isolation signal: short direct final-goal actor screens in the official loop remain at or below 0.120 success on PointMaze teleport navigate, while the learned waypoint BC executor reaches 1.000 when driven by stochastic TRL waypoints on both navigate and stitch. This isolates the current direct-actor bottleneck from the high-level stochastic value-propagation result.",
        "",
        "### Fast Evaluation Profile",
        "",
        markdown_table(fast_eval_fields, fast_eval_rows),
        "",
        "Fast-eval signal: cached topology and saved BC policies make single-seed hard-slice AntMaze screening cheap. With `--task-ids 4 5`, `--episodes 5`, and `--methods bellman_matched sto_trl_matched`, navigate reaches 0.900 stochastic TRL success and stitch reaches 1.000 in roughly five seconds per stochastic evaluation row. Increasing `--eval-action-repeat` to 2 is not claim-safe in the current stitch screen: it slightly reduces policy calls but drops stochastic TRL success from 1.000 to 0.750.",
        "",
        "Empirical graph speed note: `scripts/run_pointmaze_graph_planner.py` caches solved graph Q tables and can profile rollouts with `--profile-eval`. On `pointmaze-teleport-stitch-v0` task 4, the first 220-sweep Bellman graph solve took `11.09s`, while a cache hit took `0.03s`; a failed 1000-step rollout took `0.39s`. The learned BC graph executor was faster per step than `transition_value`, but still failed stitch task 4 in the 10-episode seed-0 screen; see `results/pointmaze_graph_summary.md`.",
        "",
        "### AntMaze Deterministic-Support Ablation",
        "",
        markdown_table(antmaze_support_fields, antmaze_support_rows),
        "",
        "Ablation signal: on AntMaze hard tasks, optimistic support-only TRL matches the low matched-Bellman success in these fast slices, while stochastic TRL reaches 0.900 on both navigate and stitch. This isolates the AntMaze gain from mere transitive reachability over observed teleport support.",
        "",
        "## PointMaze Tie-Preserving Policy-Head Screen",
        "",
        markdown_table(pointmaze_tie_fields, pointmaze_tie_rows),
        "",
        "Neural-head signal: on PointMaze teleport navigate and stitch, a raw-observation MLP transition model plus a tie-preserving raw-observation policy head recovers 0.980 stochastic TRL success and matches the full-Bellman reference, while matched Bellman reaches 0.530. This is a positive value/control-head diagnostic over the cell abstraction, not a complete end-to-end neural TRL result.",
        "",
        "### PointMaze Previous-Action Policy-Head Screen",
        "",
        markdown_table(pointmaze_prev_fields, pointmaze_prev_rows),
        "",
        "Previous-action signal: making the previous high-level action explicit lets a conventional single-label raw-observation policy head reproduce the successful PointMaze table policy on both teleport variants. With the raw-observation MLP transition head and 20 episodes per task, stochastic TRL reaches 0.980 on navigate and stitch, matching full Bellman while matched Bellman reaches 0.530. The shared evaluator now applies the same sticky previous-action tie-break to 4D previous-action policy scores; without that tie-break, the learned head could drift out of the goal cell after arrival.",
        "",
        "### PointMaze Tie-Preserving Policy-Head Eval-Seed Screen",
        "",
        markdown_table(pointmaze_tie_eval_fields, pointmaze_tie_eval_rows),
        "",
        "Eval-seed signal: with transition seed 0 fixed, the same PointMaze raw-observation transition plus tie-policy head reaches 0.927 stochastic TRL success on both navigate and stitch across three evaluation seeds, matching full Bellman while matched Bellman reaches 0.417.",
        "",
        "### PointMaze Tie-Preserving Policy-Head Transition-Seed Screen",
        "",
        markdown_table(pointmaze_tie_transition_fields, pointmaze_tie_transition_rows),
        "",
        "Transition-seed signal: with evaluation seed 0 fixed, the PointMaze raw-observation transition plus tie-policy head reaches 0.980 stochastic TRL success on both navigate and stitch across three transition seeds, matching full Bellman while matched Bellman reaches 0.530.",
        "",
        "## AntMaze Tie-Preserving Policy-Head Screen",
        "",
        markdown_table(antmaze_tie_fields, antmaze_tie_rows),
        "",
        "Neural-head signal: on AntMaze hard tasks with the learned BC executor and topology transition model, a raw-observation tie-preserving high-level policy head keeps stochastic TRL high at 0.925 on navigate and 0.950 on stitch, while matched Bellman reaches 0.350 and 0.400. This is a single-seed hard-task diagnostic, not a replacement for the multi-seed AntMaze topology table.",
        "",
        "## AntMaze Raw-Observation Transition And Policy-Head Screen",
        "",
        markdown_table(antmaze_rawobs_tie_fields, antmaze_rawobs_tie_rows),
        "",
        "Combined learned-module signal: on AntMaze hard tasks with a learned BC executor, a raw-observation MLP jump-change transition head plus a raw-observation tie-policy head reaches 0.950 stochastic TRL success on navigate and 1.000 on stitch across three transition seeds, matching the full-Bellman reference while matched Bellman reaches 0.300 on both tasks. This is still a high-level cell-abstraction diagnostic, but it removes the table transition and table policy from the AntMaze screen.",
        "",
        "### AntMaze Combined Learned-Module Eval-Seed Screen",
        "",
        markdown_table(antmaze_rawobs_eval_fields, antmaze_rawobs_eval_rows),
        "",
        "Eval-seed signal: with transition seed 0 fixed, the same raw-observation transition plus tie-policy head reaches 0.933 stochastic TRL success on navigate and 0.967 on stitch across three evaluation seeds, matching full Bellman while matched Bellman remains below 0.300.",
        "",
        "### AntMaze Previous-Action Policy-Head Eval-Seed Screen",
        "",
        markdown_table(antmaze_rawobs_prev_eval_fields, antmaze_rawobs_prev_eval_rows),
        "",
        "Previous-action signal: making the previous high-level action explicit lets a single-label raw-observation policy head preserve the stochastic TRL advantage across three evaluation seeds on AntMaze hard tasks. With a raw-observation jump-change transition head and 10 episodes per hard task, stochastic TRL reaches 0.933 on navigate and 0.967 on stitch, within 0.02 of the full-Bellman reference while matched Bellman remains at or below 0.350.",
        "",
        "## AntMaze Learned-Transition Screens",
        "",
        markdown_table(antmaze_lt_fields, antmaze_lt_rows),
        "",
        "Learned-transition signal: on AntMaze hard tasks with a learned BC executor, three independent table-softmax transition-model seeds keep stochastic TRL at 0.950 success on navigate and 1.000 on stitch. Three raw-observation MLP jump-change transition seeds reach the same 0.950 and 1.000 success. These are learned high-level transition screens, not a replacement for the main multi-evaluation-seed topology table.",
        "",
        "## Long-Horizon Tabular Safe-Optimal Cases",
        "",
        markdown_table(tabular_fields, safe_tabular),
        "",
        "Main signal: when the risky shortcut is suboptimal, stochastic TRL reaches success 1.0 with the matched transitive sweep budget, while matched Bellman and realized MC-positive baselines choose the risky shortcut.",
        "",
        "## Horizon Scaling",
        "",
        markdown_table(horizon_fields, horizon_rows),
        "",
        "Main signal: across safe path lengths 16, 32, 64, and 128, stochastic TRL keeps success 1.0 with only logarithmic matched sweeps, while matched Bellman and MC-positive baselines keep choosing the risky shortcut.",
        "",
        "## Risky-Shortcut Aggregate",
        "",
        markdown_table(aggregate_fields, aggregate_rows),
        "",
        "Main signal: stochastic TRL matches the full Bellman solution on the stochastic shortcut family while realized-trajectory variants overestimate lucky risky outcomes.",
        "",
        "## 2D Stochastic Grid Shortcut",
        "",
        markdown_table(grid_fields, grid_rows),
        "",
        "Main signal: on 2D snake-corridor grids with safe path lengths 31, 63, and 127, stochastic TRL reaches success 1.0 with 6, 7, and 8 matched sweeps. Matched Bellman and MC-positive choose the risky portal and succeed only at the portal rate, while full Bellman also reaches success 1.0.",
        "",
        "### Realized-TRL Diagnostic",
        "",
        markdown_table(grid_realized_fields, grid_realized_rows),
        "",
        "Diagnostic signal: on the reduced 2D grid diagnostic with path lengths 31 and 63, raw/log realized TRL choose the risky portal and succeed at the portal rate, while stochastic TRL again matches full Bellman.",
        "",
        "### Planning-Budget Curve",
        "",
        markdown_table(budget_fields, budget_rows),
        "",
        "Budget signal: on the hardest 16x8 grid with safe path length 127, stochastic TRL reaches success 1.0 after 7 sweeps. Bellman remains at the portal success rate through 120 sweeps and first reaches success 1.0 at 126 sweeps.",
        "",
        "## PointMaze Empirical Graph Planner",
        "",
        "### Topology-Level Planner",
        "",
        markdown_table(topo_fields, topo_rows),
        "",
        markdown_table(topo_stats_fields, topo_stats_rows),
        "",
        "High-success signal: with a coarse topology inferred from the offline dataset as the low-level routing scaffold, stochastic TRL reaches mean success 0.901 with the 6-sweep matched budget and matches the 180-sweep Bellman reference. Matched Bellman with the same 6 sweeps reaches 0.343.",
        "",
        "### Topology-Level Teleport Stitch Planner",
        "",
        markdown_table(topo_stitch_fields, topo_stitch_rows),
        "",
        markdown_table(topo_stitch_stats_fields, topo_stitch_stats_rows),
        "",
        "Stitch signal: on `pointmaze-teleport-stitch-v0`, stochastic TRL again reaches mean success 0.901 with the 6-sweep matched budget and matches the 180-sweep Bellman reference, while matched Bellman reaches 0.343.",
        "",
        "### Topology-Level Stitch Deterministic-Support Ablation",
        "",
        markdown_table(topo_stitch_support_fields, topo_stitch_support_rows),
        "",
        "Ablation signal: optimistic support TRL treats every observed stochastic teleport outcome as a reliable transition. It improves over matched Bellman but reaches only 0.449 mean success, far below calibrated stochastic TRL at 0.901. This supports the claim that stochastic Bellman calibration, not transitive composition alone, is responsible for the high-success stitch result.",
        "",
        "### Empirical Graph Planner",
        "",
        markdown_table(graph_fields, graph_rows),
        "",
        "Main signal: after fixing the evaluation seed schedule so the five seeds use disjoint rollout randomness, stochastic TRL reaches mean success 0.355 versus 0.216 for matched Bellman with the same 11-sweep planning budget. It approaches the 220-sweep full Bellman reference at 0.402.",
        "",
        "### PointMaze Paired Statistics",
        "",
        markdown_table(graph_stats_fields, graph_stats_rows),
        "",
        "The paired seed-level improvement over matched Bellman is positive for all five seeds. Stochastic TRL recovers about 75% of the gap between matched Bellman and the 220-sweep Bellman reference.",
        "",
        "### PointMaze Task Deltas",
        "",
        markdown_table(graph_task_fields, graph_task_rows),
        "",
        "Task-level means show that the matched-budget improvement comes mainly from tasks 1 and 2, where matched Bellman gets zero success, while tasks 3 and 4 are already reachable for matched Bellman.",
        "",
        "## AntMaze Learned-Controller Diagnostic",
        "",
        markdown_table(ant_fields, ant_rows),
        "",
        "Preliminary signal: on `antmaze-teleport-navigate-v0`, stochastic TRL reaches mean success 0.933 with the matched 6-sweep inferred-topology budget and a shared 20k-step full-observation BC controller, while matched Bellman reaches 0.333. This is a seed-0, 3-episode-per-task smoke, not yet the final multi-seed AntMaze table.",
        "",
        "### Navigate Body-Candidate Check",
        "",
        markdown_table(ant_nav_body_fields, ant_nav_body_rows),
        "",
        "Executor signal: with a saved 50k-step full-observation BC controller and 16 body-nearest candidate waypoint goals, stochastic TRL matches the 180-sweep Bellman reference at 0.940 success over 10 episodes per task while matched Bellman remains at 0.380.",
        "",
        "### AntMaze Body-Candidate Multi-Seed Screen",
        "",
        markdown_table(ant_multi_fields, ant_multi_rows),
        "",
        "Multi-seed signal: across three evaluation seeds and five episodes per task, stochastic TRL matches the 180-sweep Bellman reference on both AntMaze navigate and stitch, while matched Bellman remains near 0.34 success.",
        "",
        "### AntMaze Independent Controller-Seed Screen",
        "",
        markdown_table(ant_bcseed_fields, ant_bcseed_rows),
        "",
        "Controller-seed signal: with independently trained `bc_seed=1` AntMaze controllers, stochastic TRL again matches the 180-sweep Bellman reference over three evaluation seeds and 20 episodes per task. It reaches 0.933 on navigate and 0.950 on stitch, while matched Bellman remains near 0.32 on both tasks. This strengthens the AntMaze claim beyond evaluation-seed robustness for a single saved controller.",
        "",
        "### AntMaze Stitch Controller-Seed Aggregate",
        "",
        markdown_table(ant_stitch_seed_fields, ant_stitch_seed_rows),
        "",
        "Stitch robustness signal: across controller seeds 0, 1, and 2, stochastic TRL stays in the 0.950-0.963 success range under the 6-sweep matched budget, while matched Bellman remains near 0.32. The seed-2 run is a near-match to the 180-sweep Bellman reference, trailing it by one successful episode out of 300 aggregate rollouts.",
        "",
        "### AntMaze Navigate Controller-Seed Aggregate",
        "",
        markdown_table(ant_nav_seed_fields, ant_nav_seed_rows),
        "",
        "Navigate robustness signal: across controller seeds 0, 1, and 2, stochastic TRL stays in the 0.933-0.947 success range under the 6-sweep matched budget and matches the 180-sweep Bellman reference, while matched Bellman remains near 0.31.",
        "",
        "### AntMaze 20-Episode Three-Seed Hard-Task Check",
        "",
        markdown_table(ant_ep20_fields, ant_ep20_rows),
        "",
        markdown_table(ant_ep20_stats_fields, ant_ep20_stats_rows),
        "",
        "High-episode signal: with the same body-nearest executor and full-horizon evaluation, stochastic TRL matches the 180-sweep Bellman reference on both AntMaze navigate and stitch over three evaluation seeds and 20 episodes per task, while matched Bellman remains near 0.31 success.",
        "",
        "### AntMaze Planning-Budget Screens",
        "",
        markdown_table(ant_budget_fields, ant_budget_rows),
        "",
        "Budget signal: on both AntMaze navigate and stitch seed-0 screens, stochastic TRL reaches 1.000 success with 6 sweeps. Ordinary Bellman is 0.333 at 6 sweeps and first reaches 1.000 at 45 sweeps in these screens.",
        "",
        "### AntMaze Stitch Executor Ablation",
        "",
        markdown_table(ant_exec_fields, ant_exec_rows),
        "",
        "Executor signal: on the AntMaze stitch seed-0 screen, replacing a single nearest-xy waypoint observation with 16 body-nearest waypoint candidates improves both stochastic TRL and full Bellman from 0.867 to 1.000. This isolates the remaining failures as low-level executor sensitivity, not value-propagation errors.",
        "",
        "## Teleport Stitch Screens",
        "",
        markdown_table(stitch_fields, stitch_rows),
        "",
        "Stitch signal: stochastic TRL matches the long-sweep Bellman reference on PointMaze and nearly matches it on AntMaze under the matched 6-sweep budget. These are seed-0 fast screens, but they are harder long-horizon variants than the navigate-only task.",
        "",
    ]
    (RESULTS / "paper_artifacts.md").write_text("\n".join(report))


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    tabular_fields, tabular_rows, safe_success_values = build_tabular_l128()
    horizon_fields, horizon_rows = build_tabular_horizon()
    aggregate_fields, aggregate_rows = build_tabular_risky_aggregate()
    grid_fields, grid_rows = build_grid_shortcut()
    grid_realized_fields, grid_realized_rows = build_grid_realized_diagnostic()
    budget_fields, budget_rows = build_grid_budget_curve()
    graph_fields, graph_rows, graph_bars = build_graph_summary()
    graph_stats_fields, graph_stats_rows = build_graph_paired_stats()
    graph_task_fields, graph_task_rows = build_graph_task_deltas()
    main_fields, main_rows = build_main_hard_task_results()
    hard_stress_fields, hard_stress_rows = build_hard_task_stress_summary()
    pointmaze_single_fields, pointmaze_single_rows = build_pointmaze_single_task_summary()
    pointmaze_lt_fields, pointmaze_lt_rows = build_pointmaze_learned_transition_summary()
    pointmaze_bc_fields, pointmaze_bc_rows = build_pointmaze_bc_controller_summary()
    controller_iso_fields, controller_iso_rows = build_controller_execution_isolation(pointmaze_bc_rows)
    fast_eval_fields, fast_eval_rows = build_fast_eval_profile_summary()
    antmaze_support_fields, antmaze_support_rows = build_antmaze_support_ablation_summary()
    pointmaze_tie_fields, pointmaze_tie_rows = build_pointmaze_tie_policy_head_summary()
    pointmaze_prev_fields, pointmaze_prev_rows = build_pointmaze_prev_policy_head_summary()
    pointmaze_tie_eval_fields, pointmaze_tie_eval_rows = build_pointmaze_tie_policy_head_eval_seed_summary()
    pointmaze_tie_transition_fields, pointmaze_tie_transition_rows = (
        build_pointmaze_tie_policy_head_transition_seed_summary()
    )
    antmaze_tie_fields, antmaze_tie_rows = build_antmaze_tie_policy_head_summary()
    antmaze_rawobs_tie_fields, antmaze_rawobs_tie_rows = build_antmaze_rawobs_tie_policy_head_summary()
    antmaze_rawobs_eval_fields, antmaze_rawobs_eval_rows = build_antmaze_rawobs_tie_policy_eval_seed_summary()
    antmaze_rawobs_prev_eval_fields, antmaze_rawobs_prev_eval_rows = (
        build_antmaze_rawobs_prev_policy_eval_seed_summary()
    )
    antmaze_lt_fields, antmaze_lt_rows = build_antmaze_learned_transition_summary()
    topo_fields, topo_rows = build_pointmaze_topology_summary()
    topo_stitch_fields, topo_stitch_rows = build_pointmaze_topology_stitch_summary()
    topo_stitch_support_fields, topo_stitch_support_rows = build_pointmaze_topology_stitch_support_summary()
    topo_stats_fields, topo_stats_rows = build_pointmaze_topology_paired_stats()
    topo_stitch_stats_fields, topo_stitch_stats_rows = build_pointmaze_topology_stitch_paired_stats()
    ant_fields, ant_rows = build_antmaze_bc_summary()
    ant_nav_body_fields, ant_nav_body_rows = build_antmaze_navigate_bodyk16_summary()
    ant_multi_fields, ant_multi_rows = build_antmaze_bodyk16_multiseed_summary()
    ant_bcseed_fields, ant_bcseed_rows = build_antmaze_bcseed_summary()
    ant_stitch_seed_fields, ant_stitch_seed_rows = build_antmaze_stitch_controller_seed_summary()
    ant_nav_seed_fields, ant_nav_seed_rows = build_antmaze_navigate_controller_seed_summary()
    ant_ep20_fields, ant_ep20_rows = build_antmaze_bodyk16_ep20_summary()
    ant_ep20_stats_fields, ant_ep20_stats_rows = build_antmaze_bodyk16_ep20_paired_stats()
    ant_budget_fields, ant_budget_rows = build_antmaze_budget()
    ant_exec_fields, ant_exec_rows = build_antmaze_stitch_executor_ablation()
    stitch_fields, stitch_rows = build_teleport_stitch_screen()

    write_csv(TABLE_DIR / "tabular_l128.csv", tabular_fields, tabular_rows)
    write_csv(TABLE_DIR / "tabular_safe_horizon.csv", horizon_fields, horizon_rows)
    write_csv(TABLE_DIR / "tabular_risky_aggregate.csv", aggregate_fields, aggregate_rows)
    write_csv(TABLE_DIR / "grid_shortcut_2d.csv", grid_fields, grid_rows)
    write_csv(TABLE_DIR / "grid_realized_diagnostic.csv", grid_realized_fields, grid_realized_rows)
    write_csv(TABLE_DIR / "grid_budget_curve.csv", budget_fields, budget_rows)
    write_csv(TABLE_DIR / "pointmaze_graph_5seed.csv", graph_fields, graph_rows)
    write_csv(TABLE_DIR / "pointmaze_graph_paired_stats.csv", graph_stats_fields, graph_stats_rows)
    write_csv(TABLE_DIR / "pointmaze_graph_task_deltas.csv", graph_task_fields, graph_task_rows)
    write_csv(TABLE_DIR / "main_hard_task_results.csv", main_fields, main_rows)
    write_csv(TABLE_DIR / "hard_task_stress_seed0.csv", hard_stress_fields, hard_stress_rows)
    write_csv(
        TABLE_DIR / "pointmaze_stitch_task5_ep100_seed01234.csv",
        pointmaze_single_fields,
        pointmaze_single_rows,
    )
    write_csv(TABLE_DIR / "pointmaze_learned_transition.csv", pointmaze_lt_fields, pointmaze_lt_rows)
    write_csv(
        TABLE_DIR / "pointmaze_learned_controller_ep20_seed012.csv",
        pointmaze_bc_fields,
        pointmaze_bc_rows,
    )
    write_csv(
        TABLE_DIR / "controller_execution_isolation.csv",
        controller_iso_fields,
        controller_iso_rows,
    )
    write_csv(TABLE_DIR / "fast_eval_profile.csv", fast_eval_fields, fast_eval_rows)
    write_csv(
        TABLE_DIR / "antmaze_support_ablation_ep5_seed0_task45.csv",
        antmaze_support_fields,
        antmaze_support_rows,
    )
    write_csv(
        TABLE_DIR / "pointmaze_tie_policy_head_ep20_seed0.csv",
        pointmaze_tie_fields,
        pointmaze_tie_rows,
    )
    write_csv(
        TABLE_DIR / "pointmaze_rawobs_transition_prev_policy_head_ep20_seed0.csv",
        pointmaze_prev_fields,
        pointmaze_prev_rows,
    )
    write_csv(
        TABLE_DIR / "pointmaze_tie_policy_head_ep20_evalseed012.csv",
        pointmaze_tie_eval_fields,
        pointmaze_tie_eval_rows,
    )
    write_csv(
        TABLE_DIR / "pointmaze_tie_policy_head_ep20_tseed012.csv",
        pointmaze_tie_transition_fields,
        pointmaze_tie_transition_rows,
    )
    write_csv(
        TABLE_DIR / "antmaze_tie_policy_head_hard_tasks_ep20_seed0.csv",
        antmaze_tie_fields,
        antmaze_tie_rows,
    )
    write_csv(
        TABLE_DIR / "antmaze_rawobs_transition_tie_policy_head_ep10_tseed012.csv",
        antmaze_rawobs_tie_fields,
        antmaze_rawobs_tie_rows,
    )
    write_csv(
        TABLE_DIR / "antmaze_rawobs_transition_tie_policy_head_ep10_evalseed012.csv",
        antmaze_rawobs_eval_fields,
        antmaze_rawobs_eval_rows,
    )
    write_csv(
        TABLE_DIR / "antmaze_rawobs_transition_prev_policy_head_ep10_evalseed012.csv",
        antmaze_rawobs_prev_eval_fields,
        antmaze_rawobs_prev_eval_rows,
    )
    write_csv(
        TABLE_DIR / "antmaze_learned_transition_robustness.csv",
        antmaze_lt_fields,
        antmaze_lt_rows,
    )
    write_csv(TABLE_DIR / "pointmaze_topology_5seed.csv", topo_fields, topo_rows)
    write_csv(TABLE_DIR / "pointmaze_topology_stitch_5seed.csv", topo_stitch_fields, topo_stitch_rows)
    write_csv(
        TABLE_DIR / "pointmaze_topology_stitch_support_baseline_5seed.csv",
        topo_stitch_support_fields,
        topo_stitch_support_rows,
    )
    write_csv(TABLE_DIR / "pointmaze_topology_paired_stats.csv", topo_stats_fields, topo_stats_rows)
    write_csv(
        TABLE_DIR / "pointmaze_topology_stitch_paired_stats.csv",
        topo_stitch_stats_fields,
        topo_stitch_stats_rows,
    )
    write_csv(TABLE_DIR / "antmaze_bc_topology_20k_ep3_seed0.csv", ant_fields, ant_rows)
    write_csv(
        TABLE_DIR / "antmaze_navigate_50k_bodyk16_ep10_seed0.csv",
        ant_nav_body_fields,
        ant_nav_body_rows,
    )
    write_csv(TABLE_DIR / "antmaze_bodyk16_multiseed_ep5.csv", ant_multi_fields, ant_multi_rows)
    write_csv(TABLE_DIR / "antmaze_bcseed1_ep20_seed012.csv", ant_bcseed_fields, ant_bcseed_rows)
    write_csv(
        TABLE_DIR / "antmaze_stitch_controller_seeds_ep20_seed012.csv",
        ant_stitch_seed_fields,
        ant_stitch_seed_rows,
    )
    write_csv(
        TABLE_DIR / "antmaze_navigate_controller_seeds_ep20_seed012.csv",
        ant_nav_seed_fields,
        ant_nav_seed_rows,
    )
    write_csv(TABLE_DIR / "antmaze_bodyk16_ep20_seed012.csv", ant_ep20_fields, ant_ep20_rows)
    write_csv(
        TABLE_DIR / "antmaze_bodyk16_ep20_seed012_paired_stats.csv",
        ant_ep20_stats_fields,
        ant_ep20_stats_rows,
    )
    write_csv(TABLE_DIR / "antmaze_budget_ep3_seed0.csv", ant_budget_fields, ant_budget_rows)
    write_csv(
        TABLE_DIR / "antmaze_stitch_executor_ablation_ep3_seed0.csv",
        ant_exec_fields,
        ant_exec_rows,
    )

    write_grouped_svg(
        FIG_DIR / "tabular_l128_safe_success.svg",
        "Long-horizon safe-optimal stochastic shortcut",
        ["p=0.02", "p=0.05"],
        ["mc_positive", "bellman_matched", "sto_trl_matched"],
        safe_success_values,
    )
    write_horizon_svg(
        FIG_DIR / "tabular_horizon_success.svg",
        "Safe-optimal stochastic shortcut scaling",
        horizon_rows,
        ["0.02", "0.05"],
        ["mc_positive", "bellman_matched", "sto_trl_matched", "bellman_full"],
    )
    write_horizon_svg(
        FIG_DIR / "grid_shortcut_success.svg",
        "2D stochastic shortcut grid scaling",
        grid_rows,
        ["0.02", "0.05"],
        ["mc_positive", "bellman_matched", "sto_trl_matched", "bellman_full"],
    )
    write_budget_curve_svg(
        FIG_DIR / "grid_budget_curve.svg",
        "Planning budget on 16x8 stochastic grid",
        budget_rows,
    )
    write_bar_svg(
        FIG_DIR / "pointmaze_graph_success.svg",
        "PointMaze seeded matched-budget graph planner",
        graph_bars,
    )
    write_antmaze_multiseed_svg(FIG_DIR / "antmaze_bodyk16_multiseed.svg", ant_multi_rows)
    write_antmaze_budget_svg(FIG_DIR / "antmaze_budget.svg", ant_budget_rows)
    write_report(
        main_fields,
        main_rows,
        hard_stress_fields,
        hard_stress_rows,
        pointmaze_single_fields,
        pointmaze_single_rows,
        pointmaze_lt_fields,
        pointmaze_lt_rows,
        pointmaze_bc_fields,
        pointmaze_bc_rows,
        controller_iso_fields,
        controller_iso_rows,
        fast_eval_fields,
        fast_eval_rows,
        antmaze_support_fields,
        antmaze_support_rows,
        pointmaze_tie_fields,
        pointmaze_tie_rows,
        pointmaze_prev_fields,
        pointmaze_prev_rows,
        pointmaze_tie_eval_fields,
        pointmaze_tie_eval_rows,
        pointmaze_tie_transition_fields,
        pointmaze_tie_transition_rows,
        antmaze_tie_fields,
        antmaze_tie_rows,
        antmaze_rawobs_tie_fields,
        antmaze_rawobs_tie_rows,
        antmaze_rawobs_eval_fields,
        antmaze_rawobs_eval_rows,
        antmaze_rawobs_prev_eval_fields,
        antmaze_rawobs_prev_eval_rows,
        antmaze_lt_fields,
        antmaze_lt_rows,
        tabular_fields,
        tabular_rows,
        horizon_fields,
        horizon_rows,
        aggregate_fields,
        aggregate_rows,
        grid_fields,
        grid_rows,
        grid_realized_fields,
        grid_realized_rows,
        budget_fields,
        budget_rows,
        graph_fields,
        graph_rows,
        graph_stats_fields,
        graph_stats_rows,
        graph_task_fields,
        graph_task_rows,
        topo_fields,
        topo_rows,
        topo_stitch_fields,
        topo_stitch_rows,
        topo_stitch_support_fields,
        topo_stitch_support_rows,
        topo_stats_fields,
        topo_stats_rows,
        topo_stitch_stats_fields,
        topo_stitch_stats_rows,
        ant_fields,
        ant_rows,
        ant_nav_body_fields,
        ant_nav_body_rows,
        ant_multi_fields,
        ant_multi_rows,
        ant_bcseed_fields,
        ant_bcseed_rows,
        ant_stitch_seed_fields,
        ant_stitch_seed_rows,
        ant_nav_seed_fields,
        ant_nav_seed_rows,
        ant_ep20_fields,
        ant_ep20_rows,
        ant_ep20_stats_fields,
        ant_ep20_stats_rows,
        ant_budget_fields,
        ant_budget_rows,
        ant_exec_fields,
        ant_exec_rows,
        stitch_fields,
        stitch_rows,
    )

    print("wrote results/paper_artifacts.md")
    print("wrote results/paper_tables/*.csv")
    print("wrote results/figures/*.svg")


if __name__ == "__main__":
    main()
