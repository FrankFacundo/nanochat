# Week 4 assignment — Direct preference optimization

## Objective

Implement DPO, verify it independently of training, and study how preference quality and beta change behavior.

Nanochat does not currently provide DPO. This is an implementation assignment, not a command-following exercise.

## Derivation

For prompt `x`, chosen response `y_w`, rejected response `y_l`, policy `πθ`, frozen reference `πref`, and temperature-like coefficient `β`, derive and implement:

```text
Δθ   = log πθ(y_w|x)   − log πθ(y_l|x)
Δref = log πref(y_w|x) − log πref(y_l|x)
L    = −log σ(β(Δθ − Δref))
```

Your derivation must explain:

1. Why response log probabilities are sums over response tokens only.
2. Why prompt and padding tokens are masked.
3. Why the reference model is frozen.
4. What happens as beta approaches zero or becomes large.
5. Why chosen/rejected length can create a bias.

## Implementation gate

Complete the contract shown by:

```bash
python -m llm_lab show w4-dpo
```

Required tests:

- A hand-calculated two-pair batch matches the code loss.
- Swapping chosen and rejected reverses the update preference.
- Identical chosen/rejected responses yield the expected neutral loss.
- Reference parameters receive no gradients.
- Prompt and padding logits do not change the response score.

## Experimental design

Use one SFT parent and freeze a train/validation preference split. Study:

- Beta: low, medium, high.
- Clean versus deliberately corrupted preference labels.
- Easy versus hard rejected responses.
- Dataset size.

Primary measures:

- Held-out preference accuracy.
- Reward-model or verifier score, if valid.
- KL proxy to the SFT policy.
- Capability-retention suite from Week 3.
- Response length and format.

## Failure audit

Find examples of:

- Verbosity preference.
- Stylistic shortcuts.
- Overconfidence.
- Capability regression.
- Preference label ambiguity.
- Reference-policy exploitation.

## Deliverable

Submit the derivation, test evidence, data card, beta curves, held-out preference results, retention matrix, and ten failure cases.

## Rubric — 10 points

- 3: Correct derivation and tests.
- 2: Auditable preference data.
- 3: Controlled beta/quality study.
- 2: Drift and failure analysis.
