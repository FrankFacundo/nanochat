"""Experiment catalog loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CATALOG_PATH = Path(__file__).with_name("configs") / "catalog.json"
VALID_STAGES = {"base", "sft", "rl", "design"}


@dataclass(frozen=True)
class Experiment:
    id: str
    title: str
    week: int
    topic: str
    stage: str
    tier: str
    estimated_minutes: int
    question: str
    hypothesis_prompt: str
    args: dict[str, Any]
    tokenizer_args: dict[str, Any]
    eval_args: dict[str, Any]
    requires: tuple[str, ...]
    implementation_contract: dict[str, Any]

    @property
    def runnable(self) -> bool:
        return self.stage != "design"


class Catalog:
    def __init__(self, path: Path = CATALOG_PATH):
        self.path = path
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("version") != 1:
            raise ValueError(f"Unsupported catalog version in {path}")
        self.defaults = raw.get("defaults", {})
        self._experiments: dict[str, Experiment] = {}
        for item in raw.get("experiments", []):
            experiment = _parse_experiment(item)
            if experiment.id in self._experiments:
                raise ValueError(f"Duplicate experiment id: {experiment.id}")
            self._experiments[experiment.id] = experiment

    def all(self) -> list[Experiment]:
        return sorted(self._experiments.values(), key=lambda exp: (exp.week, exp.id))

    def get(self, experiment_id: str) -> Experiment:
        try:
            return self._experiments[experiment_id]
        except KeyError as exc:
            choices = ", ".join(self._experiments)
            raise KeyError(f"Unknown experiment '{experiment_id}'. Available: {choices}") from exc


def _parse_experiment(item: dict[str, Any]) -> Experiment:
    required = {"id", "title", "week", "topic", "stage", "tier", "estimated_minutes", "question"}
    missing = sorted(required - item.keys())
    if missing:
        raise ValueError(f"Experiment is missing fields {missing}: {item}")
    stage = item["stage"]
    if stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage '{stage}' for {item['id']}")
    return Experiment(
        id=str(item["id"]),
        title=str(item["title"]),
        week=int(item["week"]),
        topic=str(item["topic"]),
        stage=stage,
        tier=str(item["tier"]),
        estimated_minutes=int(item["estimated_minutes"]),
        question=str(item["question"]),
        hypothesis_prompt=str(item.get("hypothesis_prompt", "")),
        args=dict(item.get("args", {})),
        tokenizer_args=dict(item.get("tokenizer_args", {})),
        eval_args=dict(item.get("eval_args", {})),
        requires=tuple(item.get("requires", [])),
        implementation_contract=dict(item.get("implementation_contract", {})),
    )
