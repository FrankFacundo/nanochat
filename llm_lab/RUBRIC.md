# Manual rubric

The CLI grade is a completion audit. Use this rubric for the substantive grade.

## 1. Experimental method — 10 points

- 4: Hypotheses, primary metrics, and falsifiers were written before runs.
- 3: Controls and nuisance variables are explicit.
- 2: Stop/failure rules and test-set policy are frozen.
- 1: Run artifacts and environment are reproducible.

## 2. Tokenizer and baseline — 15 points

- 4: Correct use of bytes/token and BPB across vocabularies.
- 4: Correct parameter/token/FLOP accounting.
- 4: Baseline curve diagnosis supported by evidence.
- 3: Complete model/data card and qualitative failure examples.

## 3. Ablation portfolio — 30 points

- 8: Coverage and correct execution of at least 12 topics.
- 6: At least four implementation contracts include tests and baseline compatibility.
- 6: At least three pilot effects receive confirmatory runs.
- 5: Compute, wall-clock, parameter, and token confounds are handled correctly.
- 5: Null/negative results and uncertainty are reported honestly.

## 4. SFT — 10 points

- 3: Same base parent and controlled mixtures.
- 3: Capability gain measured on held-out tasks.
- 2: Forgetting or base-loss regression measured.
- 2: Formatting examples and failure analysis included.

## 5. DPO — 10 points

- 3: Correct derivation and hand-verified implementation.
- 2: Preference data audit and held-out split.
- 3: Beta/data-quality ablation with preference accuracy and drift.
- 2: Limitations and label-noise sensitivity.

## 6. RL — 10 points

- 3: Reward, pass@1/pass@k, length, and wall time jointly analyzed.
- 2: Group-size intervention is controlled.
- 3: Reward-hacking or degeneracy audit.
- 2: Correct explanation of how nanochat differs from full GRPO/PPO.

## 7. Capstone — 15 points

- 4: Frozen and auditable 30-minute protocol.
- 4: Interventions selected from prior evidence, with interactions tested.
- 3: Final result reproducible from one command.
- 2: Clear model card and test-set discipline.
- 2: Concise report with a negative-results appendix.

## Interpretation bands

- 90–100: Research-ready experimental reasoning.
- 80–89: Strong; a few controls or interpretations need refinement.
- 70–79: Competent execution but conclusions overreach the evidence.
- 60–69: Runs completed; experimental design remains unreliable.
- Below 60: Revisit Weeks 0–2 before spending more compute.

## Automatic deductions

- −10: Test set used to choose hyperparameters.
- −8: Claimed tokenizer improvement using incomparable perplexities.
- −8: Claimed architecture efficiency from fixed-step results without FLOP accounting.
- −5: Missing failed/negative runs.
- −5: Unrecoverable command or missing manifest.
- −5: Concurrent jobs used for throughput comparisons.
