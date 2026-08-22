# Week 3 assignment — Supervised fine-tuning

## Objective

Measure how supervised instruction data changes behavior, formatting, and retained base capability.

## Pre-lab

Read `scripts/chat_sft.py` and trace:

1. How a conversation becomes token IDs.
2. Which positions receive loss and which are masked.
3. How examples are packed.
4. Which pretraining hyperparameters and optimizer state are inherited.
5. How the SFT schedule differs from base pretraining.

Explain why “the assistant sounds better” is inadequate evaluation.

## Evaluation matrix

Freeze a held-out suite before SFT:

| Dimension | Base checkpoint | SFT baseline | Math-heavy SFT |
|---|---:|---:|---:|
| Validation BPB | | | |
| ARC-Easy | | | |
| ARC-Challenge | | | |
| MMLU | | | |
| GSM8K | | | |
| HumanEval | | | |
| Format compliance | | | |
| Refusal/uncertainty behavior | | | |

Use the same base parent for both SFT runs:

```bash
python -m llm_lab run w3-sft-baseline --parent BASE_RUN_ID --wandb
python -m llm_lab run w3-sft-math-heavy --parent BASE_RUN_ID --wandb
```

## Required analyses

1. Plot validation BPB and ChatCORE by SFT step.
2. Measure capability gain and forgetting separately.
3. Inspect at least 30 fixed prompts, categorized before seeing outputs.
4. Report exact-format accuracy rather than choosing only favorable examples.
5. Explain whether optimizer warm-start is an intervention. Design a fresh-optimizer comparison.
6. Compare data-mixture changes at fixed examples, fixed tokens, and fixed steps conceptually.

## Data-card exercise

For SmolTalk, MMLU, and GSM8K mixture components, record:

- Intended behavior.
- Sampling/epoch weight.
- Potential contamination.
- Loss mask behavior.
- Known failure mode.
- Held-out metric.

## Deliverable

A four-page SFT memo with the evaluation matrix, learning curves, ten representative failures, and a conclusion distinguishing instruction-following gains from newly acquired knowledge.

## Rubric — 10 points

- 3: Controlled same-parent comparison.
- 3: Quantitative capability and forgetting measures.
- 2: Data/masking understanding.
- 2: Fixed qualitative suite and failure taxonomy.
