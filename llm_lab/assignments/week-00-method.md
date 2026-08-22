# Week 0 assignment — Measurement before modeling

## Objectives

Build a trustworthy experimental loop and demonstrate that you can distinguish observation, inference, and conclusion.

## Pre-lab

Without running code, answer in your lab notebook:

1. Why does one lower training-loss point not prove a model is better?
2. Name five variables that may change MPS throughput while model code is unchanged.
3. Define independent variable, dependent variable, control, nuisance variable, and confound for an LR ablation.
4. Why must a hypothesis include a result that would falsify it?
5. When should you compare fixed steps, fixed tokens, fixed FLOPs, and fixed wall time?

Create your first entry in `student/predictions.json` before continuing.

## Laboratory

```bash
python -m llm_lab show w0-smoke
python -m llm_lab run w0-smoke --dry-run
python -m llm_lab run w0-smoke
python -m llm_lab runs
```

Inspect all five evidence sources:

1. `manifest.json`: provenance and intended intervention.
2. `stdout.log`: unstructured ground truth and warnings.
3. `metrics.jsonl`: the lossless parsed time series.
4. `summary.json`: convenient aggregates, not a substitute for curves.
5. `workspace/`: exact tokenizer and checkpoint namespace.

Start the dashboard and verify the smoke curve is visible:

```bash
python -m llm_lab dashboard --open
```

## Audit exercise

Invent three invalid comparisons and explain the failure. Include at least one from each category:

- Measurement failure, such as comparing tokenizer perplexities.
- Control failure, such as changing batch and silently allowing automatic LR scaling.
- Selection failure, such as choosing the best checkpoint on the final test set.

Then write a stop rule for divergence. It should contain a measurable threshold and an observation window. “Stop if it looks bad” earns no credit.

## Deliverable

Submit a two-page protocol containing:

- Machine and software fingerprint.
- Fixed thermal/power/sleep conditions.
- Naming and artifact rules.
- Pre-registration template.
- Divergence and interruption rules.
- Validation/test policy.
- Definition of pilot and confirmatory evidence.

## Self-defense questions

You pass only if you can answer without notes:

1. Which run file is immutable provenance and which is derived convenience?
2. Can a smoke run support an architectural claim? Why?
3. What information is lost if only final validation BPB is retained?
4. Why might two identical commands have different wall time on a laptop?
5. What does it mean for a result to be practically significant but statistically uncertain?

## Rubric — 10 points

- 3: Complete, testable protocol.
- 2: Correct metric/control definitions.
- 2: Valid failure and test-set rules.
- 2: Artifact audit.
- 1: Three convincing invalid-comparison examples.
