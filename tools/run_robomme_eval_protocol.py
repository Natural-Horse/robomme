#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from vla_scratch.robomme_eval.eval_client_template import MINI_CLIENT
from vla_scratch.robomme_eval.methods import METHODS, PAPER_PRIMARY_METHODS, TASKS_3
from vla_scratch.robomme_eval.protocol import EvalProtocol, build_run_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_ROOT = Path("/home/airhust/mtr/res/robomme_official_eval")
DEFAULT_LOG_DIR = DEFAULT_OFFICIAL_ROOT / "logs"
DEFAULT_DOCKER_IMAGE = "robomme-mme-vla:cuda122-local"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or print the unified RoboMME test protocol for VLA-scratch methods."
    )
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument(
        "--list-paper-methods",
        action="store_true",
        help="List the primary paper-aligned methods, excluding scratch fusion ablations.",
    )
    parser.add_argument("--method", choices=sorted(METHODS), default="baseline_prompt")
    parser.add_argument("--task", choices=TASKS_3, default="PatternLock")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--port", type=int, default=8150)
    parser.add_argument("--episode-limit", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1300)
    parser.add_argument("--obs-horizon", type=int, default=16)
    parser.add_argument("--action-chunk-size", type=int, default=None)
    parser.add_argument("--inference-steps", type=int, default=1)
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--run", action="store_true", help="Actually run server + Docker client.")
    return parser.parse_args()


def list_methods(names=None) -> None:
    names = names or METHODS.keys()
    for name in names:
        method = METHODS[name]
        print(
            json.dumps(
                {
                    "name": name,
                    "policy": method.policy,
                    "data_family": method.data_family,
                    "subgoal_type": method.subgoal_type,
                    "memory_kind": method.memory_kind,
                    "description": method.description,
                },
                ensure_ascii=False,
            )
        )


def wait_health(port: int, proc: subprocess.Popen, timeout_s: int = 180) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/healthz"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(url, timeout=1).read()
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"server did not become healthy: {url}")


def build_server_command(args: argparse.Namespace, spec) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "serve_robomme_policy.py"),
        f"policy={spec.policy}",
        f"data={spec.data}",
        f"host=127.0.0.1",
        f"port={args.port}",
        f"inference_steps={args.inference_steps}",
        f"use_bf16={'true' if args.use_bf16 else 'false'}",
    ]
    if spec.action_chunk_size is not None:
        command.append(f"chunk_size={spec.action_chunk_size}")
    if spec.checkpoint_path:
        command.append(f"checkpoint_path={spec.checkpoint_path}")
    return command


def build_docker_command(args: argparse.Namespace, client_path: Path) -> list[str]:
    official_root = args.official_root
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--network",
        "host",
        "-v",
        f"{official_root}:/workspace/robomme_official_eval",
        "-v",
        f"{client_path}:/tmp/robomme_eval_client.py:ro",
        "-w",
        "/workspace/robomme_official_eval/repos/robomme_policy_learning",
        args.docker_image,
        "bash",
        "-lc",
        "micromamba run -n robomme python /tmp/robomme_eval_client.py",
    ]


def print_plan(args: argparse.Namespace, spec, server_command: list[str], docker_command: list[str]) -> None:
    plan = {
        "protocol": {
            "split": spec.split,
            "task": spec.task,
            "episode_limit": spec.episode_limit,
            "action_space": spec.action_space,
            "max_steps": spec.max_steps,
            "obs_horizon": spec.obs_horizon,
            "action_chunk_size": spec.action_chunk_size,
            "success_status": "success",
        },
        "method": spec.method,
        "policy": spec.policy,
        "data": spec.data,
        "subgoal_type": spec.subgoal_type,
        "checkpoint_path": spec.checkpoint_path,
        "server_command": server_command,
        "docker_command": docker_command,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.list_methods:
        list_methods()
        return
    if args.list_paper_methods:
        list_methods(PAPER_PRIMARY_METHODS)
        return

    protocol = EvalProtocol(
        max_steps=args.max_steps,
        obs_horizon=args.obs_horizon,
        action_chunk_size=args.action_chunk_size,
        episode_limit=args.episode_limit,
    )
    spec = build_run_spec(
        method_name=args.method,
        task=args.task,
        checkpoint_path=args.checkpoint_path,
        protocol=protocol,
    )
    server_command = build_server_command(args, spec)

    client_code = MINI_CLIENT.format(
        port=args.port,
        task=spec.task,
        max_steps=spec.max_steps,
        episode_limit=spec.episode_limit,
        subgoal_type=spec.subgoal_type,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        client_path = Path(fh.name)
        fh.write(client_code)

    docker_command = build_docker_command(args, client_path)
    if not args.run:
        print_plan(args, spec, server_command, docker_command)
        client_path.unlink(missing_ok=True)
        return

    args.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_base = f"robomme_eval_{args.method}_{args.task}_ep{args.episode_limit}_{stamp}"
    server_log = args.log_dir / f"{log_base}_server.log"
    client_log = args.log_dir / f"{log_base}_client.log"

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    server_proc = None
    try:
        with server_log.open("w", encoding="utf-8") as server_fh:
            server_proc = subprocess.Popen(
                server_command,
                cwd=REPO_ROOT,
                env=env,
                stdout=server_fh,
                stderr=subprocess.STDOUT,
                text=True,
            )
            wait_health(args.port, server_proc)
            print(f"[eval] server healthy on port {args.port}; log={server_log}")
            with client_log.open("w", encoding="utf-8") as client_fh:
                result = subprocess.run(
                    docker_command,
                    cwd=REPO_ROOT,
                    stdout=client_fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
        print(json.dumps({"server_log": str(server_log), "client_log": str(client_log), "returncode": result.returncode}, indent=2))
        raise SystemExit(result.returncode)
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=15)
        client_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
