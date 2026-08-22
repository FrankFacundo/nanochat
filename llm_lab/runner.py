"""Isolated, reproducible experiment execution for the Nanochat Laboratory."""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_lab.catalog import Catalog, Experiment
from llm_lab.metrics import parse_metrics, summarize_metrics


REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_ALLOWED_ARGS = {
    "depth", "aspect_ratio", "head_dim", "max_seq_len", "window_pattern",
    "num_iterations", "target_flops", "target_param_data_ratio", "device_batch_size",
    "total_batch_size", "embedding_lr", "unembedding_lr", "weight_decay", "matrix_lr",
    "scalar_lr", "warmup_steps", "warmdown_ratio", "final_lr_frac", "eval_every",
    "eval_tokens", "core_metric_every", "core_metric_max_per_task", "sample_every",
    "save_every", "device_type",
}
SFT_ALLOWED_ARGS = {
    "num_iterations", "max_seq_len", "device_batch_size", "total_batch_size",
    "embedding_lr", "unembedding_lr", "matrix_lr", "init_lr_frac", "warmup_ratio",
    "warmdown_ratio", "final_lr_frac", "eval_every", "eval_tokens", "chatcore_every",
    "chatcore_max_cat", "chatcore_max_sample", "mmlu_epochs", "gsm8k_epochs",
    "load_optimizer", "device_type",
}
RL_ALLOWED_ARGS = {
    "num_epochs", "device_batch_size", "examples_per_step", "num_samples",
    "max_new_tokens", "temperature", "top_k", "embedding_lr", "unembedding_lr",
    "matrix_lr", "weight_decay", "init_lr_frac", "eval_every", "eval_examples",
    "save_every", "device_type",
}
TOKENIZER_ALLOWED_ARGS = {"max_chars", "doc_cap", "vocab_size"}
EVAL_ALLOWED_ARGS = {"device_batch_size", "split_tokens", "max_per_task", "eval", "device_type"}


class ExperimentError(RuntimeError):
    pass


def default_shared_dir() -> Path:
    return Path(os.environ.get("NANOCHAT_BASE_DIR", Path.home() / ".cache" / "nanochat")).expanduser()


def default_runs_dir() -> Path:
    return Path(os.environ.get("NANOCHAT_LAB_DIR", default_shared_dir() / "lab_runs")).expanduser()


def list_runs(runs_dir: Path | None = None) -> list[dict[str, Any]]:
    root = runs_dir or default_runs_dir()
    if not root.exists():
        return []
    records = []
    for summary_path in root.glob("*/summary.json"):
        try:
            records.append(json.loads(summary_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(records, key=lambda item: item.get("started_at", ""), reverse=True)


def resolve_run(run_id: str, runs_dir: Path | None = None) -> Path:
    root = (runs_dir or default_runs_dir()).resolve()
    exact = root / run_id
    if exact.is_dir():
        return exact
    matches = [path for path in root.glob(f"{run_id}*") if path.is_dir()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ExperimentError(f"No run matches '{run_id}' under {root}")
    raise ExperimentError(f"Run prefix '{run_id}' is ambiguous: {', '.join(path.name for path in matches)}")


def build_commands(experiment: Experiment, defaults: dict[str, Any], wandb: bool) -> list[list[str]]:
    run_name = experiment.id if wandb else "dummy"
    commands: list[list[str]] = []
    if experiment.tokenizer_args:
        _validate_args(experiment.tokenizer_args, TOKENIZER_ALLOWED_ARGS, experiment.id)
        commands.append([sys.executable, "-m", "scripts.tok_train", *_as_cli(experiment.tokenizer_args)])

    if experiment.stage == "base":
        args = {**defaults.get("base", {}), **experiment.args}
        _validate_args(args, BASE_ALLOWED_ARGS, experiment.id)
        commands.append(
            [sys.executable, "-m", "scripts.base_train", *_as_cli(args), "--model-tag", "lab", "--run", run_name]
        )
        if experiment.eval_args:
            _validate_args(experiment.eval_args, EVAL_ALLOWED_ARGS, experiment.id)
            commands.append(
                [sys.executable, "-m", "scripts.base_eval", "--model-tag", "lab", *_as_cli(experiment.eval_args)]
            )
    elif experiment.stage == "sft":
        args = {**defaults.get("sft", {}), **experiment.args}
        _validate_args(args, SFT_ALLOWED_ARGS, experiment.id)
        commands.append(
            [sys.executable, "-m", "scripts.chat_sft", *_as_cli(args), "--model-tag", "lab", "--run", run_name]
        )
    elif experiment.stage == "rl":
        args = {**defaults.get("rl", {}), **experiment.args}
        _validate_args(args, RL_ALLOWED_ARGS, experiment.id)
        commands.append(
            [sys.executable, "-m", "scripts.chat_rl", *_as_cli(args), "--model-tag", "lab", "--run", run_name]
        )
    return commands


def run_experiment(
    experiment: Experiment,
    catalog: Catalog,
    *,
    parent_run: str | None = None,
    wandb: bool = False,
    dry_run: bool = False,
    allow_expensive: bool = False,
) -> Path | None:
    if not experiment.runnable:
        contract = json.dumps(experiment.implementation_contract, indent=2)
        raise ExperimentError(
            f"{experiment.id} is an implementation assignment, not runnable yet.\nContract:\n{contract}"
        )
    if experiment.estimated_minutes > 10 and not allow_expensive and not dry_run:
        raise ExperimentError(
            f"{experiment.id} is estimated at {experiment.estimated_minutes} minutes. "
            "Re-run with --yes after completing its pre-registration."
        )
    if experiment.stage in {"sft", "rl"} and not parent_run:
        raise ExperimentError(f"{experiment.stage} experiments require --parent RUN_ID")

    commands = build_commands(experiment, catalog.defaults, wandb)
    if dry_run:
        for command in commands:
            print(shlex.join(command))
        return None

    active_pids = _active_training_pids()
    if active_pids:
        raise ExperimentError(
            "Another nanochat training process is active "
            f"(PID{'s' if len(active_pids) > 1 else ''}: {', '.join(active_pids)}). "
            "Concurrent runs invalidate timing comparisons."
        )

    shared_dir = default_shared_dir().resolve()
    runs_dir = default_runs_dir().resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    execution_id = f"{experiment.id}--{timestamp}"
    run_dir = runs_dir / execution_id
    workspace = run_dir / "workspace"
    run_dir.mkdir(parents=True, exist_ok=False)
    workspace.mkdir()

    parent_path = resolve_run(parent_run, runs_dir) if parent_run else None
    _prepare_workspace(experiment, shared_dir, workspace, parent_path)

    manifest = {
        "schema_version": 1,
        "execution_id": execution_id,
        "experiment_id": experiment.id,
        "title": experiment.title,
        "topic": experiment.topic,
        "stage": experiment.stage,
        "tier": experiment.tier,
        "question": experiment.question,
        "hypothesis_prompt": experiment.hypothesis_prompt,
        "args": experiment.args,
        "tokenizer_args": experiment.tokenizer_args,
        "parent_run": parent_path.name if parent_path else None,
        "wandb_enabled": wandb,
        "commands": commands,
        "repo_commit": _git_output("rev-parse", "HEAD"),
        "repo_dirty": bool(_git_output("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "started_at": _now(),
    }
    _write_json(run_dir / "manifest.json", manifest)

    env = os.environ.copy()
    env["NANOCHAT_BASE_DIR"] = str(workspace)
    # Training output is piped through this process. Force line buffering so the
    # terminal, stdout.log, W&B-adjacent monitoring, and wall-time annotations
    # update during the run instead of arriving in one block at process exit.
    env["PYTHONUNBUFFERED"] = "1"
    if wandb:
        env["WANDB_MODE"] = "online"
    all_metrics: list[dict[str, Any]] = []
    exit_code = 0
    started = time.monotonic()
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as log_file, (run_dir / "metrics.jsonl").open(
        "w", encoding="utf-8"
    ) as metrics_file:
        for command_index, command in enumerate(commands):
            header = f"\n$ {shlex.join(command)}\n"
            print(header, end="")
            log_file.write(header)
            log_file.flush()
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                for parsed in parse_metrics(line):
                    metric = {
                        **parsed.as_dict(),
                        "command_index": command_index,
                        "wall_time_seconds": round(time.monotonic() - started, 3),
                    }
                    all_metrics.append(metric)
                    metrics_file.write(json.dumps(metric, sort_keys=True) + "\n")
                    metrics_file.flush()
            exit_code = process.wait()
            if exit_code:
                break

    summary = {
        **manifest,
        "status": "completed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "finished_at": _now(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "metrics": summarize_metrics(all_metrics),
        "run_dir": str(run_dir),
    }
    _write_json(run_dir / "summary.json", summary)
    if exit_code:
        raise ExperimentError(f"Experiment failed with exit code {exit_code}. See {run_dir / 'stdout.log'}")
    return run_dir


def _prepare_workspace(experiment: Experiment, shared: Path, workspace: Path, parent: Path | None) -> None:
    source_workspace = parent / "workspace" if parent else shared
    data_source = shared / "base_data_climbmix"
    if not data_source.exists():
        raise ExperimentError(f"Missing dataset at {data_source}. Run: python -m nanochat.dataset -n 8")
    _symlink(data_source, workspace / "base_data_climbmix")

    tokenizer_source = source_workspace / "tokenizer"
    if not experiment.tokenizer_args:
        if not tokenizer_source.exists():
            raise ExperimentError(f"Missing tokenizer at {tokenizer_source}. Run: python -m scripts.tok_train")
        shutil.copytree(tokenizer_source, workspace / "tokenizer")

    eval_bundle = shared / "eval_bundle"
    if eval_bundle.exists():
        _symlink(eval_bundle, workspace / "eval_bundle")

    if parent:
        required_checkpoint = "base_checkpoints" if experiment.stage == "sft" else "chatsft_checkpoints"
        checkpoint_source = source_workspace / required_checkpoint
        if not checkpoint_source.exists():
            raise ExperimentError(f"Parent {parent.name} has no {required_checkpoint}")
        _symlink(checkpoint_source, workspace / required_checkpoint)
        base_checkpoints = source_workspace / "base_checkpoints"
        if base_checkpoints.exists() and required_checkpoint != "base_checkpoints":
            _symlink(base_checkpoints, workspace / "base_checkpoints")


def _validate_args(args: dict[str, Any], allowed: set[str], experiment_id: str) -> None:
    unknown = sorted(set(args) - allowed)
    if unknown:
        raise ExperimentError(f"Unsupported arguments for {experiment_id}: {', '.join(unknown)}")


def _as_cli(args: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(flag)
        else:
            result.extend([flag, str(value)])
    return result


def _symlink(source: Path, destination: Path) -> None:
    destination.symlink_to(source, target_is_directory=True)


def _git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip()


def _active_training_pids() -> list[str]:
    pattern = r"python.*-m scripts\.(base_train|chat_sft|chat_rl)"
    try:
        result = subprocess.run(["pgrep", "-f", pattern], text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
