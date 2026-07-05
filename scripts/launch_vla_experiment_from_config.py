#!/usr/bin/env python3
"""Launch a VLA-scratch training/eval queue from one explicit JSON config."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def override_items(values: dict[str, Any]) -> list[str]:
    return [f"{key}={hydra_value(value)}" for key, value in values.items()]


def print_command(cmd: list[str]) -> None:
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)


def run_command(
    cmd: list[str],
    log_path: Path | None,
    dry_run: bool,
    env: dict[str, str],
    append: bool = False,
) -> None:
    print_command(cmd)
    if dry_run:
        return
    if log_path is None:
        subprocess.run(cmd, check=True, env=env)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        if proc.wait() != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def read_eval_info(eval_dir: Path) -> dict[str, Any] | None:
    info_path = eval_dir / "eval_info.json"
    if not info_path.exists():
        return None
    return json.loads(info_path.read_text(encoding="utf-8"))


def metric_value(info: dict[str, Any] | None, metric: str) -> float | None:
    if info is None:
        return None
    cur: Any = info
    for key in metric.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def checkpoint_epochs(cfg: dict[str, Any]) -> list[int]:
    eval_cfg = cfg.get("eval", {})
    if eval_cfg.get("checkpoint_epochs"):
        return [int(epoch) for epoch in eval_cfg["checkpoint_epochs"]]

    train = cfg["train"]
    epochs = int(train["epochs"])
    save_interval = int(train["save_interval"])
    epochs_to_eval = list(range(save_interval, epochs + 1, save_interval))
    if epochs not in epochs_to_eval:
        epochs_to_eval.append(epochs)
    return epochs_to_eval


def checkpoint_dir(cfg: dict[str, Any], method_name: str, suite: dict[str, Any], epoch: int) -> Path:
    run_dir = Path(cfg["output_root"]) / cfg["run_id"] / method_name / suite["name"]
    return run_dir / f"checkpoint_{epoch}"


def validate_config(cfg: dict[str, Any]) -> None:
    common = cfg.get("common_overrides", {})
    methods = cfg.get("methods", {})
    required = cfg.get("required_safety_overrides", {})

    missing: list[str] = []
    wrong: list[str] = []
    for method_name, method_cfg in methods.items():
        merged = {**common, **method_cfg.get("overrides", {})}
        for key, expected in required.items():
            actual = merged.get(key, None)
            if actual is None:
                missing.append(f"{method_name}.{key}")
            elif actual != expected:
                wrong.append(f"{method_name}.{key}: expected {expected!r}, got {actual!r}")

    if missing or wrong:
        lines = ["Config failed safety validation."]
        if missing:
            lines.append("Missing required keys: " + ", ".join(missing))
        if wrong:
            lines.append("Wrong values: " + "; ".join(wrong))
        raise ValueError("\n".join(lines))


def start_tensorboard(cfg: dict[str, Any], env: dict[str, str], dry_run: bool) -> None:
    tb = cfg.get("tensorboard", {})
    if not tb.get("enabled", False):
        return
    py = cfg["python"]
    output_root = cfg["output_root"]
    run_id = cfg["run_id"]
    log_dir = Path(cfg["log_dir"])
    port = str(tb.get("port", 6007))
    host = str(tb.get("host", "127.0.0.1"))
    log_path = log_dir / f"tensorboard_{run_id}_{port}.log"
    cmd = [
        py,
        "-m",
        "tensorboard.main",
        "--logdir",
        f"{output_root}/{run_id}",
        "--host",
        host,
        "--port",
        port,
    ]
    print("[tensorboard]", flush=True)
    print_command(cmd)
    if dry_run:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    (log_dir / f"tensorboard_{run_id}_{port}.pid").write_text(str(proc.pid), encoding="utf-8")


def train_one(
    cfg: dict[str, Any],
    method_name: str,
    suite: dict[str, Any],
    env: dict[str, str],
    dry_run: bool,
) -> None:
    method = cfg["methods"][method_name]
    train = cfg["train"]
    run_id = cfg["run_id"]
    epochs = int(train["epochs"])
    batch_size = int(train["batch_size"])
    run_dir = Path(cfg["output_root"]) / run_id / method_name / suite["name"]
    final_checkpoint = run_dir / f"checkpoint_{epochs}" / "model.pt"
    if final_checkpoint.exists() and not dry_run:
        print(f"[skip-train] {method_name}/{suite['name']} existing={final_checkpoint}", flush=True)
        return

    merged = {
        **cfg["common_overrides"],
        **method.get("overrides", {}),
        "batch_size": batch_size,
        "num_workers": train["num_workers"],
        "prefetch_factor": train["prefetch_factor"],
        "use_ddp": train["use_ddp"],
        "low_mem": train["low_mem"],
        "epochs": epochs,
        "log_interval": train["log_interval"],
        "eval_interval": train["eval_interval"],
        "save_interval": train["save_interval"],
        "wandb.mode": train["wandb_mode"],
        "exp_name": f"{run_id}_{method_name}_{suite['name']}_b{batch_size}_e{epochs}",
        "hydra.run.dir": str(run_dir),
    }
    cmd = [
        cfg["python"],
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={train['nproc_per_node']}",
        train["script"],
        f"policy={method['policy']}",
        f"data={suite['data']}",
        *override_items(merged),
    ]
    local_env = env.copy()
    local_env["MASTER_ADDR"] = "127.0.0.1"
    local_env["MASTER_PORT"] = str(suite["master_port"])
    print(f"[train] {method_name}/{suite['name']}", flush=True)
    run_command(cmd, Path(cfg["log_dir"]) / f"train_{method_name}_{suite['name']}.log", dry_run, local_env)


def eval_one(
    cfg: dict[str, Any],
    method_name: str,
    suite: dict[str, Any],
    env: dict[str, str],
    dry_run: bool,
) -> dict[str, Any] | None:
    eval_cfg = cfg.get("eval", {})
    if not eval_cfg.get("enabled", False):
        return None
    method = cfg["methods"][method_name]
    epochs = [int(cfg["train"]["epochs"])]
    if eval_cfg.get("every_checkpoint", False):
        epochs = checkpoint_epochs(cfg)
    results: list[dict[str, Any]] = []
    metric = eval_cfg.get("best_metric", "overall.pc_success")

    for epoch in epochs:
        ckpt_dir = checkpoint_dir(cfg, method_name, suite, epoch)
        if not (ckpt_dir / "model.pt").exists() and not dry_run:
            print(f"[skip-eval] {method_name}/{suite['name']} missing={ckpt_dir / 'model.pt'}", flush=True)
            continue

        result = eval_checkpoint(cfg, method_name, method, suite, epoch, ckpt_dir, env, dry_run)
        if result is not None:
            result["metric"] = metric_value(result.get("eval_info"), metric)
            results.append(result)

    if not results:
        return None

    best = max(
        results,
        key=lambda item: (
            float("-inf") if item.get("metric") is None else item["metric"],
            int(item["epoch"]),
        ),
    )
    write_best_summary(cfg, method_name, suite, metric, results, best, dry_run)
    if eval_cfg.get("keep_only_best", False):
        prune_non_best_results(results, best, dry_run)
    return best


def eval_checkpoint(
    cfg: dict[str, Any],
    method_name: str,
    method: dict[str, Any],
    suite: dict[str, Any],
    epoch: int,
    ckpt_dir: Path,
    env: dict[str, str],
    dry_run: bool,
) -> dict[str, Any] | None:
    eval_cfg = cfg.get("eval", {})
    eval_dir = Path(cfg["eval_root"]) / f"{method_name}_{suite['name']}_e{epoch}_1"
    if (eval_dir / "eval_info.json").exists() and not dry_run:
        print(f"[skip-eval] {method_name}/{suite['name']} existing={eval_dir / 'eval_info.json'}", flush=True)
        return {
            "epoch": epoch,
            "checkpoint": str(ckpt_dir),
            "eval_dir": str(eval_dir),
            "eval_info": read_eval_info(eval_dir),
        }

    cmd = [
        cfg["python"],
        eval_cfg["script"],
        f"policy={method['policy']}",
        f"data={suite['data']}",
        "data.video_backend=pyav",
        f"checkpoint_path={ckpt_dir}",
        f"merge_policy_cfg={hydra_value(eval_cfg['merge_policy_cfg'])}",
        f"env_task={suite['env_task']}",
        f"eval_n_episodes={eval_cfg['eval_n_episodes']}",
        f"eval_batch_size={eval_cfg['eval_batch_size']}",
        f"max_episodes_rendered={eval_cfg['max_episodes_rendered']}",
        f"output_dir={eval_dir}",
    ]
    log_path = Path(cfg["log_dir"]) / f"eval_{method_name}_{suite['name']}_e{epoch}.log"
    print(f"[eval] {method_name}/{suite['name']} checkpoint_{epoch}", flush=True)
    run_command(cmd, log_path, dry_run, env)

    rename_script = eval_cfg.get("rename_script")
    if rename_script:
        rename_cmd = [cfg["python"], rename_script, str(eval_dir), "--suite", suite["env_task"]]
        print(f"[rename-eval-videos] {method_name}/{suite['name']} checkpoint_{epoch}", flush=True)
        run_command(rename_cmd, None if dry_run else log_path, dry_run, env, append=True)
    return {
        "epoch": epoch,
        "checkpoint": str(ckpt_dir),
        "eval_dir": str(eval_dir),
        "eval_info": None if dry_run else read_eval_info(eval_dir),
    }


def write_best_summary(
    cfg: dict[str, Any],
    method_name: str,
    suite: dict[str, Any],
    metric: str,
    results: list[dict[str, Any]],
    best: dict[str, Any],
    dry_run: bool,
) -> None:
    summary = {
        "method": method_name,
        "suite": suite["name"],
        "metric": metric,
        "best_epoch": best["epoch"],
        "best_metric": best.get("metric"),
        "best_checkpoint": best["checkpoint"],
        "best_eval_dir": best["eval_dir"],
        "results": [
            {
                "epoch": item["epoch"],
                "metric": item.get("metric"),
                "checkpoint": item["checkpoint"],
                "eval_dir": item["eval_dir"],
            }
            for item in results
        ],
    }
    summary_path = Path(cfg["eval_root"]) / f"best_{method_name}_{suite['name']}.json"
    print(
        f"[best-eval] {method_name}/{suite['name']} epoch={best['epoch']} {metric}={best.get('metric')}",
        flush=True,
    )
    if dry_run:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prune_non_best_results(results: list[dict[str, Any]], best: dict[str, Any], dry_run: bool) -> None:
    best_dir = Path(best["eval_dir"]).resolve()
    for item in results:
        eval_dir = Path(item["eval_dir"])
        if eval_dir.resolve() == best_dir:
            continue
        print(f"[prune-eval] remove non-best {eval_dir}", flush=True)
        if not dry_run and eval_dir.exists():
            shutil.rmtree(eval_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON experiment config.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    validate_config(cfg)

    repo_root = Path(cfg["repo_root"])
    os.chdir(repo_root)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    for key, value in cfg.get("env", {}).items():
        env[key] = str(value)

    print(f"[launch] config={cfg_path}", flush=True)
    print(f"[launch] run_id={cfg['run_id']}", flush=True)
    print(f"[launch] output={cfg['output_root']}/{cfg['run_id']}", flush=True)
    start_tensorboard(cfg, env, args.dry_run)

    for method_name in cfg["queue"]["methods"]:
        if method_name not in cfg["methods"]:
            raise KeyError(f"Unknown method in queue: {method_name}")
        for suite in cfg["queue"]["suites"]:
            train_one(cfg, method_name, suite, env, args.dry_run)
            eval_one(cfg, method_name, suite, env, args.dry_run)


if __name__ == "__main__":
    main()
