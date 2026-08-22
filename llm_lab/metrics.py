"""Parse nanochat console output into stable JSONL metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
VAL_RE = re.compile(r"Step\s+(?P<step>\d+)\s+\|\s+Validation bpb:\s+(?P<value>[-+0-9.eE]+)")
CORE_RE = re.compile(r"Step\s+(?P<step>\d+)\s+\|\s+CORE metric:\s+(?P<value>[-+0-9.eE]+)")
CHATCORE_RE = re.compile(
    r"Step\s+(?P<step>\d+)\s+\|\s+ChatCORE:\s+(?P<value>[-+0-9.eE]+)"
    r"\s+\|\s+ChatCORE_cat:\s+(?P<cat>[-+0-9.eE]+)"
)
TRAIN_RE = re.compile(
    r"^step\s+(?P<step>\d+)(?:/\d+)?(?:\s+\([^)]*\))?\s+\|\s+"
    r"loss:\s+(?P<loss>[-+0-9.eE]+)\s+\|\s+lrm:\s+(?P<lrm>[-+0-9.eE]+)"
    r"\s+\|\s+dt:\s+(?P<dt>[-+0-9.eE]+)ms\s+\|\s+tok/sec:\s+(?P<tps>[0-9,]+)"
)
RL_RE = re.compile(
    r"Step\s+(?P<step>\d+)/\d+\s+\|\s+Average reward:\s+(?P<reward>[-+0-9.eE]+)"
    r"\s+\|\s+Average sequence length:\s+(?P<length>[-+0-9.eE]+)"
)
PASS_RE = re.compile(r"Pass@(?P<k>\d+):\s+(?P<value>[-+0-9.eE]+)")


@dataclass(frozen=True)
class ParsedMetric:
    name: str
    step: int
    value: float
    extra: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "step": self.step, "value": self.value, **self.extra}


def parse_metrics(line: str) -> list[ParsedMetric]:
    clean = ANSI_RE.sub("", line).strip()
    parsed: list[ParsedMetric] = []

    if match := VAL_RE.search(clean):
        parsed.append(_metric("val_bpb", match))
    if match := CORE_RE.search(clean):
        parsed.append(_metric("core", match))
    if match := CHATCORE_RE.search(clean):
        step = int(match.group("step"))
        parsed.append(ParsedMetric("chatcore", step, float(match.group("value")), {}))
        parsed.append(ParsedMetric("chatcore_categorical", step, float(match.group("cat")), {}))
    if match := TRAIN_RE.search(clean):
        step = int(match.group("step"))
        parsed.extend(
            [
                ParsedMetric("train_loss", step, float(match.group("loss")), {}),
                ParsedMetric("lr_multiplier", step, float(match.group("lrm")), {}),
                ParsedMetric("step_ms", step, float(match.group("dt")), {}),
                ParsedMetric("tokens_per_second", step, float(match.group("tps").replace(",", "")), {}),
            ]
        )
    if match := RL_RE.search(clean):
        step = int(match.group("step"))
        parsed.append(ParsedMetric("reward", step, float(match.group("reward")), {}))
        parsed.append(ParsedMetric("sequence_length", step, float(match.group("length")), {}))
    if "Pass@" in clean:
        step_match = re.search(r"Step\s+(\d+)", clean)
        step = int(step_match.group(1)) if step_match else 0
        for match in PASS_RE.finditer(clean):
            parsed.append(ParsedMetric(f"pass_at_{match.group('k')}", step, float(match.group("value")), {}))
    return parsed


def summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        grouped.setdefault(str(metric["name"]), []).append(metric)

    summary: dict[str, dict[str, float | int]] = {}
    for name, values in grouped.items():
        ordered = sorted(values, key=lambda value: int(value["step"]))
        numeric = [float(value["value"]) for value in ordered]
        summary[name] = {
            "first": numeric[0],
            "last": numeric[-1],
            "min": min(numeric),
            "max": max(numeric),
            "last_step": int(ordered[-1]["step"]),
            "count": len(ordered),
        }
    return summary


def _metric(name: str, match: re.Match[str]) -> ParsedMetric:
    return ParsedMetric(name, int(match.group("step")), float(match.group("value")), {})
