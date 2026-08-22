"""Command-line interface for the Nanochat Laboratory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_lab.catalog import CATALOG_PATH, Catalog
from llm_lab.dashboard import serve
from llm_lab.report import compare_runs, grade_progress
from llm_lab.runner import ExperimentError, default_runs_dir, list_runs, run_experiment


COURSE_DIR = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ExperimentError, KeyError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m llm_lab", description="Nanochat Laboratory course tools")
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH, help="experiment catalog JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list course experiments")
    list_parser.add_argument("--week", type=int)
    list_parser.set_defaults(func=_list_experiments)

    show_parser = subparsers.add_parser("show", help="show one experiment and its hypothesis prompt")
    show_parser.add_argument("experiment_id")
    show_parser.set_defaults(func=_show_experiment)

    run_parser = subparsers.add_parser("run", help="run an isolated experiment")
    run_parser.add_argument("experiment_id")
    run_parser.add_argument("--parent", help="parent execution id for SFT or RL")
    run_parser.add_argument("--wandb", action="store_true", help="log training metrics online to W&B")
    run_parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    run_parser.add_argument("--yes", action="store_true", help="confirm an experiment estimated above 10 minutes")
    run_parser.set_defaults(func=_run)

    runs_parser = subparsers.add_parser("runs", help="list captured runs")
    runs_parser.set_defaults(func=_runs)

    compare_parser = subparsers.add_parser("compare", help="produce a Markdown comparison of runs")
    compare_parser.add_argument("run_ids", nargs="+")
    compare_parser.add_argument("--output", type=Path)
    compare_parser.set_defaults(func=_compare)

    dashboard_parser = subparsers.add_parser("dashboard", help="start the interactive local comparison dashboard")
    dashboard_parser.add_argument("--host", default="127.0.0.1")
    dashboard_parser.add_argument("--port", type=int, default=8765)
    dashboard_parser.add_argument("--open", action="store_true", dest="open_browser")
    dashboard_parser.set_defaults(func=_dashboard)

    grade_parser = subparsers.add_parser("grade", help="run the formative completion audit")
    grade_parser.set_defaults(func=_grade)
    return parser


def _list_experiments(args: argparse.Namespace) -> None:
    catalog = Catalog(args.catalog)
    print("ID                              WEEK  STAGE   TIER       MIN  TOPIC")
    for experiment in catalog.all():
        if args.week is not None and experiment.week != args.week:
            continue
        marker = "" if experiment.runnable else "*"
        print(
            f"{experiment.id + marker:<31} {experiment.week:>4}  {experiment.stage:<7} "
            f"{experiment.tier:<9} {experiment.estimated_minutes:>4}  {experiment.topic}"
        )
    print("\n* implementation assignment; complete its contract before running")


def _show_experiment(args: argparse.Namespace) -> None:
    experiment = Catalog(args.catalog).get(args.experiment_id)
    print(json.dumps(experiment.__dict__, indent=2, default=list))


def _run(args: argparse.Namespace) -> None:
    catalog = Catalog(args.catalog)
    experiment = catalog.get(args.experiment_id)
    result = run_experiment(
        experiment,
        catalog,
        parent_run=args.parent,
        wandb=args.wandb,
        dry_run=args.dry_run,
        allow_expensive=args.yes,
    )
    if result:
        print(f"\nRun captured at: {result}")


def _runs(args: argparse.Namespace) -> None:
    records = list_runs()
    if not records:
        print(f"No runs found under {default_runs_dir()}")
        return
    print("EXECUTION ID                                      STATUS      MIN  FINAL VAL BPB")
    for record in records:
        val = record.get("metrics", {}).get("val_bpb", {}).get("last")
        val_text = "—" if val is None else f"{val:.5f}"
        print(
            f"{record.get('execution_id', '?'):<49} {record.get('status', '?'):<10} "
            f"{record.get('duration_seconds', 0) / 60:>5.1f}  {val_text}"
        )


def _compare(args: argparse.Namespace) -> None:
    print(compare_runs(args.run_ids, args.output))
    if args.output:
        print(f"Written to {args.output}", file=sys.stderr)


def _dashboard(args: argparse.Namespace) -> None:
    serve(args.host, args.port, args.open_browser)


def _grade(args: argparse.Namespace) -> None:
    score, notes = grade_progress(COURSE_DIR)
    print(f"Formative completion score: {score}/100")
    for note in notes:
        print(f"- {note}")
