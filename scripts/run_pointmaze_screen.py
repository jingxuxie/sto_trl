from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRL_DIR = ROOT / "external" / "trl"
DEFAULT_SAVE_DIR = ROOT / "results" / "ogbench_screen_exp"


AGENT_FILES = {
    "gcfbc": "agents/gcfbc.py",
    "msebc": "agents/msebc.py",
    "trl": "agents/trl.py",
    "trl_log": "agents/trl_log.py",
    "trl_log_mc": "agents/trl_log_mc.py",
    "trl_log_relax_mc": "agents/trl_log_relax_mc.py",
    "trl_log_relax_neg": "agents/trl_log_relax_neg.py",
    "trl_log_relax_td": "agents/trl_log_relax_td.py",
    "trl_log_relax_tdmax": "agents/trl_log_relax_tdmax.py",
}


def build_command(args: argparse.Namespace, agent: str, seed: int) -> list[str]:
    agent_file = AGENT_FILES.get(agent, agent)
    agent_stem = Path(agent_file).stem
    actor_only_agent = agent_stem in {"gcfbc", "msebc"}
    cmd = [
        sys.executable,
        "main.py",
        "--wandb_mode=disabled",
        f"--run_group={args.run_group}",
        f"--seed={seed}",
        f"--env_name={args.env_name}",
        f"--agent={agent_file}",
        f"--offline_steps={args.steps}",
        f"--log_interval={args.log_interval}",
        f"--eval_interval={args.eval_interval}",
        f"--eval_episodes={args.eval_episodes}",
        f"--eval_at_start={str(args.eval_at_start)}",
        "--video_episodes=0",
        "--save_interval=1000000000",
        f"--save_dir={args.save_dir}",
        f"--agent.batch_size={args.batch_size}",
        f"--agent.actor_hidden_dims={args.hidden_dims}",
        f"--agent.layer_norm={str(args.layer_norm)}",
    ]
    if not actor_only_agent:
        cmd.extend(
            [
                f"--agent.pe_type={args.pe_type}",
                f"--agent.value_hidden_dims={args.hidden_dims}",
            ]
        )
    if not actor_only_agent and args.pe_type == "rpg":
        cmd.extend(
            [
                f"--agent.rpg.alpha={args.rpg_alpha}",
                "--agent.rpg.const_std=True",
            ]
        )
    if not actor_only_agent and agent in {
        "trl_log_mc",
        "trl_log_relax_mc",
        "trl_log_relax_neg",
        "trl_log_relax_td",
        "trl_log_relax_tdmax",
    } and args.mc_alpha is not None:
        cmd.append(f"--agent.mc_alpha={args.mc_alpha}")
    if not actor_only_agent and agent in {"trl_log_relax_mc", "trl_log_relax_neg", "trl_log_relax_td", "trl_log_relax_tdmax"} and args.tr_alpha is not None:
        cmd.append(f"--agent.tr_alpha={args.tr_alpha}")
    if not actor_only_agent and agent == "trl_log_relax_neg" and args.neg_alpha is not None:
        cmd.append(f"--agent.neg_alpha={args.neg_alpha}")
    if not actor_only_agent and agent in {"trl_log_relax_td", "trl_log_relax_tdmax"} and args.td_alpha is not None:
        cmd.append(f"--agent.td_alpha={args.td_alpha}")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="pointmaze-teleport-navigate-v0")
    parser.add_argument("--agents", nargs="+", default=["trl", "trl_log", "trl_log_mc"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-at-start", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dims", default="(128, 128)")
    parser.add_argument("--layer-norm", action="store_true")
    parser.add_argument("--pe-type", default="rpg", choices=["rpg", "frs", "discrete"])
    parser.add_argument("--rpg-alpha", type=float, default=0.03)
    parser.add_argument("--mc-alpha", type=float, default=None)
    parser.add_argument("--tr-alpha", type=float, default=None)
    parser.add_argument("--neg-alpha", type=float, default=None)
    parser.add_argument("--td-alpha", type=float, default=None)
    parser.add_argument("--run-group", default="pointmaze_teleport_fast_screen")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.save_dir = args.save_dir.resolve()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        env["JAX_PLATFORMS"] = "cpu"

    for seed in args.seeds:
        for agent in args.agents:
            cmd = build_command(args, agent, seed)
            print("\n[run]", " ".join(cmd), flush=True)
            if args.dry_run:
                continue
            subprocess.run(cmd, cwd=TRL_DIR, env=env, check=True)


if __name__ == "__main__":
    main()
