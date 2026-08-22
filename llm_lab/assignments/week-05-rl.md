# Week 5 assignment — Verifiable-reward policy gradients

## Objective

Understand on-policy rollout training through direct measurement of reward, pass@k, output length, and failure modes.

## Code reading

Read `scripts/chat_rl.py`. Explain why its header puts “GRPO” in quotation marks. Specifically identify the absence or modification of:

- A frozen reference/KL trust region.
- PPO importance ratios and clipping.
- Sequence-level z-score normalization.
- Off-policy reuse.

Trace one GSM8K question through prompt rendering, multiple rollouts, reward extraction, group-mean advantage, masked log probabilities, gradient accumulation, and optimizer update.

## Pre-lab derivation

For sampled response tokens `y₁…yT` and scalar advantage `A`, derive the loss implemented by:

```text
− A × Σ log πθ(yt | x, y<t)
```

Explain why subtracting the group mean is a baseline, what happens when every reward in a group is equal, and how longer responses affect token-normalized updates.

## Laboratory

From the same SFT parent:

```bash
python -m llm_lab run w5-rl-baseline --parent SFT_RUN_ID --wandb --yes
python -m llm_lab run w5-rl-samples8 --parent SFT_RUN_ID --wandb --yes
```

Freeze an evaluation set and plot:

- Mean reward.
- Pass@1 and pass@k.
- Mean response length.
- Wall-clock minutes per update.
- Fraction of groups with nonzero advantage variance.

Do not claim improvement if reward rises but pass@1 does not. Do not claim reasoning improvement until answers are checked for reward-parser exploits.

## Group-size analysis

For group sizes 2, 4, 8, and optionally 16, distinguish:

- Information per question.
- Rollout compute.
- Reward variance.
- Probability that a group contains both success and failure.
- Number of optimizer updates achievable in a fixed wall-clock budget.

## Reward-hacking audit

Inspect at least 100 fixed outputs and classify:

- Correct reasoning and correct answer.
- Incorrect reasoning with correct extracted answer.
- Correct reasoning with parser failure.
- Format exploitation.
- Repetition or length pathologies.
- Memorized-looking response.

Change the verifier only after freezing and reporting the original results.

## Deliverable

A reproducible RL memo with derivation, group-size comparison, reward/pass@k/length curves, wall-clock analysis, and failure taxonomy.

## Rubric — 10 points

- 3: Correct objective and code interpretation.
- 2: Same-parent group-size control.
- 3: Joint reward/capability/length analysis.
- 2: Verifier and reward-hacking audit.
