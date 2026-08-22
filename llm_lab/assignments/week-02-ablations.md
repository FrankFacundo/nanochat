# Week 2 assignment — Optimization, scaling, and architecture

## Objective

Learn to make causal claims from controlled interventions. The goal is not to crown eighteen winners; it is to understand mechanisms, interactions, and the limits of each comparison.

Read [../ABLATION_MATRIX.md](../ABLATION_MATRIX.md) before launching anything.

## Phase 1 — Optimization pilots

Use `w1-pilot` as the control. Pre-register and run:

```bash
python -m llm_lab run w2-lr-half
python -m llm_lab run w2-lr-double
python -m llm_lab run w2-warmup-zero
python -m llm_lab run w2-warmup-long
python -m llm_lab run w2-wd-zero
python -m llm_lab run w2-wd-double
python -m llm_lab run w2-batch-8192
python -m llm_lab run w2-batch-32768 --yes
```

For each family, plot validation BPB against:

- Optimization step.
- Tokens processed.
- Wall-clock minutes.

Required reasoning:

1. LR: distinguish convergence speed from final quality.
2. Warmup: quantify the budget consumed before peak LR.
3. Decay: inspect train/validation movement separately.
4. Batch: state that nanochat scales LR as a function of total batch. Design a follow-up that isolates batch noise from this policy.

## Phase 2 — Scaling pilots

Run the token-horizon, allocation, head, and context studies:

```bash
python -m llm_lab run w2-tokens-1250 --yes
python -m llm_lab run w2-depth8-fixed-width
python -m llm_lab run w2-width512-fixed-depth
python -m llm_lab run w2-heads-4
python -m llm_lab run w2-heads-3
python -m llm_lab run w2-context-256
python -m llm_lab run w2-context-1024
```

Build a resource table containing actual parameters, FLOPs/token, total tokens, wall time, and median tokens/s. Answer:

1. Which comparisons are fixed-step, fixed-token, fixed-width, or approximately fixed-compute?
2. Why does d8 at width 384 still change more than depth alone?
3. Why does changing head dimension affect RoPE as well as head count?
4. What portion of context-length slowdown is predicted by attention's quadratic term?
5. Which result would you expect to reverse at a longer horizon?

## Phase 3 — Implementation contracts

Complete at least four starred entries from the catalog:

- Schedule family.
- AdamW versus Muon.
- Gradient clipping.
- MLP ratio.
- Tied embeddings.
- RMSNorm placement.
- ReLU²/GELU/SwiGLU.
- RoPE configuration.
- Data manifest/mixture.

For each implementation:

1. Write a unit test before the training run.
2. Preserve the existing configuration as the default.
3. Persist the option in checkpoint metadata.
4. Update parameter and FLOP accounting.
5. Prove every trainable parameter is optimized exactly once.
6. Re-run `w0-smoke` and show that baseline behavior remains compatible.

Do not compare SwiGLU and GELU with identical expansion dimensions and call it parameter matched. Do both a raw implementation comparison and a width-adjusted comparison.

## Phase 4 — Confirmation

Select three pilot effects:

- At least one optimization effect.
- At least one architecture, tokenizer, or data effect.
- At least one result that surprised you or appeared null.

Before confirmation, freeze:

- Exact code commit.
- Larger training budget.
- Primary metric.
- Early-stop rule.
- Expected direction and minimum meaningful effect.

If only the “winner” is confirmed, selection bias remains. Confirm the control and intervention under identical conditions.

## Portfolio format

For each of at least 12 topics, submit a one-page experiment card:

1. Question and mechanism.
2. Prediction and falsifier.
3. Independent variable and controls.
4. Execution IDs.
5. Curves and resource table.
6. Effect size.
7. Confounds and uncertainty.
8. Conclusion with strength label: observation, tentative, or confirmed.
9. Cheapest decisive follow-up.

Include at least two null or negative results.

## Oral self-defense

1. Why can changing total batch alter both gradient noise and LR in nanochat?
2. Why is final BPB insufficient for comparing schedules?
3. How would you fairly tune AdamW versus Muon without an unlimited search budget?
4. What does “parameter matched” fail to control?
5. When can a data-mixture gain be contamination rather than generalization?

## Rubric — 30 points

- 8: Breadth and correct controls.
- 6: Tested implementation contracts.
- 6: Confirmatory discipline.
- 5: Resource/effect-size analysis.
- 5: Calibrated conclusions and negative results.
