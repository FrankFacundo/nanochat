# Concept-check guide

Open this only after writing your own answers. These are grading criteria, not scripts to memorize.

## Week 0

- A lower training loss may reflect overfitting, changed tokenization, data leakage, more compute, or a metric mismatch. A valid quality claim needs a held-out metric and controlled budget.
- Fixed steps controls update count; fixed tokens controls data exposure; fixed FLOPs controls approximate compute; fixed wall time controls practical hardware value. None universally replaces the others.
- A falsifier names an observation inconsistent with the proposed mechanism. “The metric may be worse” is too vague without a threshold and budget.

## Week 1

- Perplexity is exponential mean token NLL and inherits the tokenizer's segmentation. BPB normalizes total NLL by underlying bytes, permitting cross-tokenizer comparison.
- In nanochat, vocabulary affects input embedding, output head, and alternating value embeddings. Consequently vocabulary is also an architecture/capacity intervention.
- For d6/aspect 64/head dimension 64, width is 384 and query-head count is six. The 32×512 micro-batch contains 16,384 tokens, so the standard total batch needs no accumulation.
- Samples are useful for diagnosing collapse, formatting, and gross capability, but a few prompts are selection-prone and cannot replace held-out evaluation.

## Week 2

- Nanochat has multiple parameter-group learning rates. A “global LR” ablation should scale all groups or explicitly study one group.
- The current base schedule is linear warmup, constant plateau, then linear warmdown—not cosine.
- Total-batch changes invoke automatic square-root LR scaling. This is a joint batch-plus-policy intervention unless scaling is disabled or compensated.
- Fixed-width depth studies still change layer count, residual transformations, value embeddings, parameters, and FLOPs. Fixed-step outcomes do not establish compute efficiency.
- Head-count changes through `head_dim`; because RoPE frequencies live in head channels, this also changes positional-frequency discretization.
- SwiGLU with the same nominal expansion adds a second input projection. Parameter-matched and raw comparisons answer different questions.
- Gradient clipping belongs after accumulation and, for fp16, after unscaling. Log how often it activates; otherwise “enabled” has no interpretable dose.
- Tied weights must be one shared Parameter and appear once in optimizer groups. Copying values is not tying.

## Week 3

- SFT loss must target assistant response tokens while excluding prompts, padding, and tool outputs as defined by the objective.
- Better format compliance can coexist with worse base BPB or knowledge metrics. Measure capability acquisition and forgetting separately.
- Reusing optimizer state can affect early dynamics, so it is part of the training policy and deserves a control when conclusions depend on it.

## Week 4

- DPO increases the policy's chosen-versus-rejected log-ratio relative to the frozen reference. Beta controls the strength/sensitivity of this preference update.
- Response-only log-probability sums prevent prompt likelihood from entering the preference comparison. Length remains a concern because sums scale with response length.
- Preference accuracy can improve through stylistic shortcuts; retention and qualitative audits are required.

## Week 5

- Subtracting the group mean is a variance-reducing baseline. If every reward is identical, every advantage is zero and that group gives no policy-gradient signal.
- Nanochat's method is on-policy and omits reference KL, PPO ratios/clipping, and standard sequence-level z-score GRPO normalization.
- Pass@k can rise while pass@1 stays flat because diversity increases. Reward can rise through parser exploitation. Report reward, pass@1, pass@k, length, and audited outputs together.

## Week 6

- Selecting and evaluating on the same validation repeatedly creates adaptive overfitting. The sealed test is opened only after the final candidate and protocol are locked.
- Combining individually useful changes can fail because optimizer, batch, schedule, capacity, and data interact. An ablation ladder and interaction tests are needed.
- Exact reproduction is evidence; a command that merely resembles the final run is not.
