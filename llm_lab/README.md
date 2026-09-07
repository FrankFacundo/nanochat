# Nanochat Laboratory

Nanochat Laboratory is a six-week, experiment-driven course in language-model pretraining and post-training. It is designed around one machine—the 128 GB M3 Max—and one rule: a claim is not accepted until it is supported by a controlled run, an appropriate metric, and a written account of uncertainty.

This is not a recipe for producing the largest model that fits in memory. It is a laboratory for learning how optimizers, training budgets, architecture, tokenization, data, SFT, preference optimization, and policy-gradient training change a model.

## What you will produce

By the end you will have:

- A reproducible d6 pretraining baseline.
- Structured evidence for all 18 requested ablation topics.
- Implementations of several missing experimental controls in nanochat.
- A base → SFT → preference/RL comparison.
- An interactive record of learning curves and efficiency.
- A 30-minute M3 Max “Parameter Golf” capstone.
- A final report containing positive and negative results.

The course deliberately distinguishes three levels of evidence:

1. **Smoke run:** proves the instrumentation works; supports no model-quality conclusion.
2. **Pilot:** identifies plausible effects cheaply; generates a confirmatory hypothesis.
3. **Confirmatory run:** uses a frozen protocol and sufficient budget to support a conclusion.

## Why this format

The course uses a mix of formats:

- Markdown for lectures, derivations, pre-registrations, and conclusions.
- A Python CLI for isolated and reproducible experiments.
- JSONL for lossless, tool-independent metric storage.
- W&B optionally for remote monitoring and curve inspection.
- A local interactive dashboard for direct comparisons.

A notebook is intentionally not the primary runner. Hidden notebook state is convenient for exploration but weak for reproducibility. You may use notebooks for secondary analysis after the immutable run artifacts exist.

## Quick start

From the repository root:

```bash
./entrypoint.sh course
```

This starts the course dashboard at http://127.0.0.1:8765 and opens it in your
browser. It uses the repository's `.venv` when available. Stop it with `Ctrl+C`.
Set `ANTHROPIC_API_KEY` before launching to enable AI grading.

To run experiments from the command line:

```bash
python -m llm_lab list
python -m llm_lab show w0-smoke
python -m llm_lab run w0-smoke --dry-run
python -m llm_lab run w0-smoke
python -m llm_lab runs
```

Before every non-smoke experiment:

1. Add your prediction and falsifier to `student/predictions.json`.
2. Inspect the exact command with `--dry-run`.
3. Confirm that only the intended independent variable changed.
4. Launch the run. Experiments over ten estimated minutes require `--yes`.

To log training online:

```bash
uv run wandb login
python -m llm_lab run w1-baseline --wandb --yes
```

The base and SFT stages appear in the `nanochat` and `nanochat-sft` W&B projects. RL appears in `nanochat-rl`. Local records remain authoritative even when W&B is enabled.

Compare captured runs:

```bash
python -m llm_lab compare RUN_ID_A RUN_ID_B --output llm_lab/student/comparison.md
python -m llm_lab dashboard --open
```

## Adaptive LLM assessment

The dashboard is also the primary learning interface. It selects the next unlocked
challenge, accepts a written defense plus explicit workspace paths and run IDs,
and asks Anthropic to grade the submission against the course rubric. The API key
stays in the Python server process and is never sent to the browser.

```bash
export ANTHROPIC_API_KEY=your_key
python -m llm_lab dashboard --open
```

By default the evaluator uses `claude-opus-5` with high effort. You may override
those choices without changing course code:

```bash
export LLM_LAB_GRADER_MODEL=claude-opus-5
export LLM_LAB_GRADER_EFFORT=high
```

The mastery path contains 15 gated assessments over 18 skills. A score of 85
passes an activity; skill mastery uses repeated per-skill evidence, so one lucky
answer is not enough. Every attempt is appended to
`~/.cache/nanochat/lab_learning/attempts.jsonl`, including rubric scores,
feedback, evidence hashes, model metadata, and token usage. Set
`NANOCHAT_LEARNING_DIR` to change that location.

The dashboard includes four surfaces:

- **Learn:** answer the adaptive challenge and attach code, reports, and runs.
- **Progress:** inspect weighted course grades, week gates, and skill mastery.
- **Runs:** compare captured learning curves and compute metrics.
- **History:** audit every answer, grade, feedback record, and evaluator call.

`python -m llm_lab grade` prints the LLM-assessed transcript. The older
file/run checklist remains available as `python -m llm_lab audit`; it is not a
substantive grade.

Run identifiers may be shortened to any unique prefix.

For post-training, pass the exact parent run:

```bash
python -m llm_lab run w3-sft-baseline --parent W1_BASELINE_RUN_ID --wandb
python -m llm_lab run w5-rl-baseline --parent W3_SFT_RUN_ID --wandb --yes
```

## Where results live

By default, run artifacts are stored outside Git:

```text
~/.cache/nanochat/lab_runs/<experiment-id>--<timestamp>/
├── manifest.json       exact command, configuration, commit, parent, and machine
├── metrics.jsonl       parsed time series
├── stdout.log          complete console output
├── summary.json        outcome and summary statistics
└── workspace/          isolated tokenizer and checkpoints
```

Set `NANOCHAT_LAB_DIR` to change the run location. Dataset shards are shared through a symlink; tokenizers and checkpoint namespaces are isolated so an ablation cannot overwrite your baseline.

## The scientific protocol

Every conclusion must answer all seven questions:

1. What changed?
2. What was held fixed?
3. What was predicted before observing the result?
4. What primary metric decides the hypothesis?
5. What is the effect size and compute cost?
6. What confound or uncertainty remains?
7. What is the cheapest experiment that could falsify the interpretation?

Use validation bits-per-byte for comparisons involving different tokenizers. Token cross-entropy and perplexity are not comparable across vocabularies.

Never call a fixed-step architecture comparison “compute matched.” A wider, deeper, longer-context, or gated model changes FLOPs per token. First measure the raw intervention; then perform a fixed-token or fixed-FLOP confirmation appropriate to the claim.

## Required sequence

Read [SYLLABUS.md](SYLLABUS.md), then complete assignments in order. Week 2 is a menu: run all cheap native ablations, implement at least four architecture/optimization contracts, and select at least three effects for longer confirmation. Weeks 3–5 use the strongest justified checkpoint—not automatically the lowest pilot BPB.

The manual standard is defined in [RUBRIC.md](RUBRIC.md). The LLM evaluator uses
that standard for every submission. The CLI completion audit is only a checklist:

```bash
python -m llm_lab audit
```

## Guardrails

- Do not run two training jobs simultaneously; thermal and memory contention invalidate throughput comparisons.
- Keep the Mac plugged in and prevent sleep during timed experiments.
- Record macOS, PyTorch, nanochat commit, dtype, and thermal conditions.
- Never tune on the final test set.
- Do not compare runs with different validation samples unless that limitation is explicit.
- Do not delete negative results. They are part of the capstone appendix.
- Stop a clearly divergent run, but retain its log and apply the pre-registered failure rule.

## Course map

- [Week 0: Experimental method](assignments/week-00-method.md)
- [Week 1: Tokenization and pretraining](assignments/week-01-pretraining.md)
- [Week 2: Scaling, optimization, and architecture](assignments/week-02-ablations.md)
- [Week 3: Supervised fine-tuning](assignments/week-03-sft.md)
- [Week 4: Preference optimization](assignments/week-04-dpo.md)
- [Week 5: Verifiable-reward RL](assignments/week-05-rl.md)
- [Week 6: M3 Max Parameter Golf](assignments/week-06-capstone.md)

The complete intervention/control map is in [ABLATION_MATRIX.md](ABLATION_MATRIX.md).
