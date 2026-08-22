"""Run comparison and formative grading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from llm_lab.runner import default_runs_dir, resolve_run


def compare_runs(run_ids: list[str], output: Path | None = None) -> str:
    summaries = [_load_summary(resolve_run(run_id)) for run_id in run_ids]
    metric_names = sorted({name for summary in summaries for name in summary.get("metrics", {})})
    preferred = [name for name in ("val_bpb", "train_loss", "core", "chatcore", "reward", "tokens_per_second") if name in metric_names]
    lines = [
        "# Experiment comparison",
        "",
        "| Run | Experiment | Status | Duration (min) | " + " | ".join(preferred) + " |",
        "|---|---|---:|---:|" + "---:|" * len(preferred),
    ]
    for summary in summaries:
        cells = [
            summary["execution_id"],
            summary["experiment_id"],
            summary["status"],
            f"{summary.get('duration_seconds', 0) / 60:.1f}",
        ]
        for metric in preferred:
            value = summary.get("metrics", {}).get(metric, {}).get("last")
            cells.append("—" if value is None else f"{value:.6g}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Required interpretation",
            "",
            "- Prediction made before training:",
            "- Primary metric and why it answers the question:",
            "- Observed effect size, not just the winner:",
            "- Compute or parameter-count confound:",
            "- Plausible mechanism:",
            "- What evidence would change your conclusion:",
            "- Next experiment:",
            "",
        ]
    )
    text = "\n".join(lines)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return text


def grade_progress(course_dir: Path) -> tuple[int, list[str]]:
    runs = []
    root = default_runs_dir()
    if root.exists():
        runs = [_load_summary(path.parent) for path in root.glob("*/summary.json")]
    completed_topics = {run.get("topic") for run in runs if run.get("status") == "completed"}
    completed_stages = {run.get("stage") for run in runs if run.get("status") == "completed"}
    notes: list[str] = []
    score = 0

    baseline_done = any(run.get("experiment_id") == "w1-baseline" and run.get("status") == "completed" for run in runs)
    if baseline_done:
        score += 10
    else:
        notes.append("Missing completed w1-baseline run (10 points).")

    ablation_topics = {
        "learning-rate", "warmup", "schedule", "optimizer", "weight-decay", "gradient-clipping",
        "batch-size", "training-tokens", "depth-width", "attention-heads", "mlp-ratio", "context-length",
        "tokenizer-vocab", "tied-embeddings", "norm-placement", "activation", "rope", "data-mixture",
    }
    topic_points = round(45 * len(completed_topics & ablation_topics) / len(ablation_topics))
    score += topic_points
    notes.append(f"Ablation coverage: {len(completed_topics & ablation_topics)}/18 topics ({topic_points}/45 points).")

    post_stages = len(completed_stages & {"sft", "rl"})
    score += post_stages * 5
    if post_stages < 2:
        notes.append("Complete both an SFT and RL run for 10 post-training points.")

    for filename, points in (("predictions.json", 10), ("conclusions.json", 15), ("final-report.md", 10)):
        path = course_dir / "student" / filename
        if path.exists() and "TODO" not in path.read_text(encoding="utf-8"):
            score += points
        else:
            notes.append(f"Complete student/{filename} ({points} points).")
    return min(score, 100), notes


def aggregate_metric(summaries: list[dict[str, Any]], metric: str) -> float | None:
    values = [summary.get("metrics", {}).get(metric, {}).get("last") for summary in summaries]
    numeric = [float(value) for value in values if value is not None]
    return mean(numeric) if numeric else None


def _load_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
