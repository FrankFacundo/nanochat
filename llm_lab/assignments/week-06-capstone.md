# Week 6 assignment — M3 Max Parameter Golf

## Objective

Produce the best defensible held-out BPB under a fixed local compute budget while demonstrating reproducibility and honest experimental reasoning.

## Frozen rules

Before any capstone run, write and commit:

- Hardware: this M3 Max, no external compute.
- Training budget: 30 minutes of measured training-loop time.
- Data manifest and tokenizer training corpus.
- Validation split and primary metric.
- Final test set, sealed until selection ends.
- Timer start/stop definition.
- Maximum checkpoint/artifact size, if used.
- Allowed precomputation.
- Failure/retry policy.

Changing a rule after observing a result creates a new competition version.

## Required baselines

1. Official d6 policy adjusted to the timer.
2. Best single intervention from Week 2.
3. A deliberately simple alternative, such as smaller depth with more tokens.

## Search strategy

Use staged evidence:

1. Screen cheap interventions with short pilots.
2. Confirm main effects individually.
3. Test interactions before combining them.
4. Lock a candidate.
5. Run at least one exact reproduction.
6. Evaluate the sealed test once.

An additive assumption is a hypothesis. LR, optimizer, batch, schedule, depth, and activation commonly interact.

## Required analysis

- Validation BPB versus time.
- Validation BPB versus tokens.
- Parameters and FLOPs/token.
- Median and tail throughput.
- Ablation ladder from baseline to final candidate.
- Reproduction difference.
- At least five discarded ideas with evidence.
- Test-versus-validation gap.

## Reproducible entrypoint

Create one executable script that:

1. Verifies the expected commit and prerequisites.
2. Uses a unique run/checkpoint namespace.
3. Captures stdout and structured metrics.
4. Enables optional W&B logging.
5. Produces the final evaluation without manual intervention.

## Final report structure

1. Abstract: one claim with measured effect and budget.
2. Rules and environment.
3. Baselines.
4. Search methodology.
5. Main ablations.
6. Final configuration.
7. Reproduction and final test.
8. Failure analysis.
9. Limitations.
10. Negative-results appendix.
11. Model/data card.

## Oral self-defense

1. Which intervention has the strongest causal evidence?
2. Which apparent improvement is most likely noise or selection bias?
3. What changed quality per token but hurt quality per minute?
4. What would you test with ten times more compute?
5. Which conclusion probably does not scale beyond d6?
6. What did the final test reveal that validation did not?

## Rubric — 15 points

- 4: Frozen, fair protocol.
- 4: Evidence-driven model development.
- 3: Exact reproducibility.
- 2: Test discipline and model card.
- 2: Clear report and negative evidence.
