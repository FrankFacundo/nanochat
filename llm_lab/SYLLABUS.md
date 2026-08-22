# Syllabus

## Course objective

Develop the ability to design, execute, diagnose, and communicate language-model training experiments. Successful completion means you can explain not only which configuration won, but why the comparison was valid, what it cost, how uncertain it is, and which next run has the highest information value.

Expected effort is 8–12 hours per week, including unattended training. The M3 Max is treated as a fixed laboratory instrument. Unified memory is ample; compute throughput and experimental discipline are scarce.

## Learning outcomes

You will be able to:

1. Explain BPE training and use BPB to compare tokenizers.
2. Derive parameter, token, and approximate FLOP budgets for a decoder Transformer.
3. Diagnose healthy, underfit, unstable, and overfit learning curves.
4. Separate learning-rate, batch, optimizer, regularization, and schedule effects.
5. Design depth/width/head/MLP/context comparisons with explicit compute controls.
6. Implement and test architecture switches without changing the baseline accidentally.
7. Quantify SFT capability gains and forgetting.
8. Derive DPO and audit preference data.
9. Explain nanochat's policy-gradient objective and evaluate pass@k, reward, KL proxies, and response length.
10. Produce a reproducible technical report with calibrated conclusions.

## Weekly schedule

### Week 0 — Measurement before modeling

Topics: hypotheses, falsifiers, independent variables, nuisance variables, pilot versus confirmation, deterministic artifacts, failure rules.

Laboratory: `w0-smoke`; inspect every captured artifact; deliberately identify a comparison the smoke run cannot support.

Assessment: experimental protocol memo and ten concept checks.

### Week 1 — Tokens, objectives, and a baseline

Topics: Unicode bytes, BPE, vocabulary/capacity tradeoffs, causal language modeling, cross-entropy, perplexity, BPB, train/validation separation, sampling as weak evidence.

Laboratory: `w1-pilot`, 8K/16K/32K tokenizer study, then `w1-baseline`.

Assessment: tokenizer report, parameter accounting, baseline model card.

### Week 2 — Optimization, scaling, and architecture

Topics: LR and warmup, schedule shape, AdamW and Muon, decay, gradient clipping, gradient noise, batch policies, token horizons, depth/width allocation, heads, MLP expansion, context cost, embeddings, RMSNorm, activations, RoPE, data mixtures.

Laboratory: all inexpensive native pilots; implement at least four starred contracts; confirm three effects at a larger budget.

Assessment: ablation portfolio containing at least 12 topics, including four implementation studies and two null/negative results.

### Week 3 — Supervised fine-tuning

Topics: conversation serialization, loss masks, task mixtures, optimizer warm starts, instruction following, format learning, catastrophic forgetting, base versus instruct evaluation.

Laboratory: `w3-sft-baseline` and `w3-sft-math-heavy` from the same base parent.

Assessment: capability/forgetting matrix and SFT data card.

### Week 4 — Preference optimization

Topics: preference-pair construction, Bradley–Terry models, reference policies, DPO derivation, beta, chosen/rejected quality, label noise, held-out preference accuracy, KL drift.

Laboratory: implement `scripts/chat_dpo.py`; hand-verify its loss; run beta and data-quality ablations.

Assessment: derivation, unit test, preference-data audit, DPO comparison memo.

### Week 5 — Policy gradients with verifiable rewards

Topics: rollouts, sequence/token credit assignment, baselines and advantages, REINFORCE, GRPO differences, pass@k, reward hacking, entropy, response length, sample-group size, on-policy cost.

Laboratory: `w5-rl-baseline` and `w5-rl-samples8` from the same SFT parent.

Assessment: reward audit, learning-dynamics plot, failure-case taxonomy.

### Week 6 — M3 Max Parameter Golf

Topics: constrained optimization, staged ablations, interaction effects, reproducibility, final-test discipline, communicating negative evidence.

Laboratory: fixed 30-minute training budget, fixed dataset and validation, frozen timer definition. Combine only interventions supported by prior evidence.

Assessment: final report, reproducible entrypoint, model card, oral self-defense questions.

## Reading sequence

Read source code first; use papers to generalize what you observed.

1. `nanochat/tokenizer.py`, `scripts/tok_train.py`, `nanochat/loss_eval.py`.
2. `nanochat/gpt.py`, especially attention, MLP, residual path, initialization, and FLOP accounting.
3. `scripts/base_train.py`, especially optimizer groups and schedules.
4. *Attention Is All You Need*; *RoFormer*; *GLU Variants Improve Transformer*.
5. *AdamW*; the Muon implementation and comments in `nanochat/optim.py`.
6. *Scaling Laws for Neural Language Models* and *Training Compute-Optimal Large Language Models*.
7. `scripts/chat_sft.py`; *Training Language Models to Follow Instructions with Human Feedback*.
8. *Direct Preference Optimization*.
9. `scripts/chat_rl.py`; the policy-gradient and GRPO sections of *DeepSeekMath*.

## Assessment structure

- 10% experimental method and reproducibility
- 15% tokenizer/pretraining baseline
- 30% ablation portfolio
- 10% SFT study
- 10% DPO implementation and study
- 10% RL study
- 15% capstone

See [RUBRIC.md](RUBRIC.md) for quality standards. Completing commands without defensible analysis earns at most half credit.

## Advancement gates

You may advance only when:

- Week 0: the smoke manifest and metric JSONL are understood.
- Week 1: the baseline has a complete model card and untouched test set.
- Week 2: at least one claimed improvement survives a larger-budget confirmation.
- Week 3: capability gain and forgetting are both measured.
- Week 4: DPO loss matches the hand calculation.
- Week 5: reward improvement is checked against pass@1 and response length.
- Week 6: the protocol is frozen before final training.
