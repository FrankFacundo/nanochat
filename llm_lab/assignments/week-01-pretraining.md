# Week 1 assignment — Tokenization and pretraining

## Objectives

Understand the transformation from bytes to tokens to next-token loss, establish a full d6 baseline, and learn why bits-per-byte is the primary cross-tokenizer metric.

## Part A — Tokenizer anatomy

Read `scripts/tok_train.py`, `nanochat/tokenizer.py`, and `nanochat/loss_eval.py`.

Answer before running:

1. What are the initial symbols from which BPE merges are learned?
2. Why can a larger vocabulary lower tokens/document while increasing model parameters?
3. Which model tensors scale directly with vocabulary in this nanochat architecture?
4. Explain why perplexity 20 under tokenizer A and perplexity 18 under tokenizer B do not establish that B is better.
5. Derive BPB from total negative log likelihood in nats and the number of non-special bytes.

Run:

```bash
python -m llm_lab run w1-vocab-8192
python -m llm_lab run w1-vocab-16384
python -m llm_lab run w1-vocab-32768
python -m llm_lab run w1-pilot
```

Use the three `w1-vocab-*` runs for the strict comparison: each tokenizer sees the same `max_chars=500000000`. `w1-pilot` uses the shared 2B-character tokenizer and is therefore a useful training control, not the strict 32K tokenizer control.

Report:

- Bytes/token on news, code, math, science, train, and validation samples.
- Tokenizer training time.
- Model parameter categories.
- Tokens/s and estimated bytes/s.
- Final validation BPB with learning curves.
- At least five segmentation examples where vocabularies behave differently.

Do not declare a vocabulary winner from one 600-step token-matched run. Explain whether byte-matched or FLOP-matched training would change the interpretation.

## Part B — Forward and backward pass

Trace one batch through `nanochat/gpt.py` and annotate shapes for:

- Token IDs and embeddings.
- Q, K, and V before attention.
- Attention output before and after head reassembly.
- MLP expansion and projection.
- Logits and shifted targets.

For the d6 baseline, calculate before checking the log:

- `n_embd` and number of query heads.
- Tokens/micro-batch.
- Gradient-accumulation steps.
- Total tokens for 5,000 iterations.
- Approximate embedding, unembedding, value-embedding, and transformer-matrix parameters.

Reconcile your calculation with the printed parameter report. A mismatch must be explained, not silently corrected.

## Part C — Full baseline

Pre-register expected final BPB, runtime, and qualitative abilities, then run:

```bash
python -m llm_lab run w1-baseline --wandb --yes
```

Diagnose:

1. Warmup behavior.
2. The constant-LR region and linear warmdown in the current code.
3. Training-loss versus validation-BPB trend.
4. Throughput drift and evaluation pauses.
5. Sample evolution at steps 500–5,000.
6. Underfitting versus overfitting evidence.

## Deliverables

- Tokenizer comparison memo.
- Annotated forward-pass shape table.
- Parameter/token/FLOP calculation sheet.
- Baseline model card using `templates/model-card.md`.
- One dashboard screenshot or exported plot plus the execution ID.

## Rubric — 15 points

- 4: Correct tokenizer reasoning and BPB use.
- 3: Correct tensor shapes and objective.
- 3: Parameter/token accounting.
- 3: Evidence-based baseline diagnosis.
- 2: Reproducible model card and failure examples.
