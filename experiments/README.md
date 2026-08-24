# Nanochat Mastery Lab — evidence packs

One runnable pack per course activity. Every number in every pack is produced by the
scripts here against this repo's real artifacts: the 171 ClimbMix shards, the trained
tokenizer, the `d6` base checkpoint (step 5000), the `d6` SFT checkpoint (step 1499),
and training runs executed on this machine.

```bash
source .venv/bin/activate
python -m experiments.<week>.<script>          # each pack is self-contained
python -m pytest experiments/w2_ablations/test_tied_embeddings.py experiments/w4_dpo/test_dpo.py
```

Attach the `.py` files plus `logs/` as workspace evidence when submitting.

---

## The number everything else depends on

**σ = 0.003651 bpb** — the noise floor, measured from 3 seeds of the pilot control
(`w2_ablations/run_noise_floor.sh`, final val_bpb 1.526232 / 1.523440 / 1.530679).
Decision rule used throughout: an effect is REAL only above 3σ = 0.011 bpb.

Caveat stated in every pack that uses it: nanochat seeds only weight init, not data
order, so this understates true run-to-run variance and every effect size below is an
optimistic significance estimate.

---

## Week 0 · Experimental method — [w0_method/](w0_method/)

| activity | script | what it proves |
|---|---|---|
| `w0-method-defense` | [01_variables_and_controls.py](w0_method/01_variables_and_controls.py) | IV/DV/controls read by diffing the CLI commands llm_lab actually builds (13 controls verified identical); batch→LR coupling computed from `base_train.py:299`; the four budget axes priced in real FLOPs |
| `w0-artifact-audit` | [02_artifact_audit.py](w0_method/02_artifact_audit.py) | `summary.json` recomputed from `metrics.jsonl` with **0 mismatches** → it is derived, not a record; `repo_dirty=true` found in the real w0-smoke manifest; three invalid comparisons quantified |

Key finding: the w0-smoke run pins `repo_commit` but the tree was dirty, so the commit
does not identify the code that ran.

## Week 1 · Tokenization & pretraining — [w1_bpb/](w1_bpb/)

| activity | script | what it proves |
|---|---|---|
| `w1-tokenizer-defense` | [01](w1_bpb/01_bpe_mechanism.py) · [02](w1_bpb/02_bpb_derivation.py) · [03](w1_bpb/03_vocab_capacity.py) · [04](w1_bpb/04_fair_comparison.py) | see below |
| `w1-baseline-diagnosis` | [05_baseline_diagnosis.py](w1_bpb/05_baseline_diagnosis.py) | shapes traced with live forward hooks; the run's own printed accounting recomputed and matched; verdict **underfitting** (3.53 tokens/param vs Chinchilla ~20 = 5.7× short) |

- **BPE**: four real tokenizers trained; ranks 0–255 proven to be all 256 bytes; merges
  reconstructed in learned order; compression 2.021 → 4.683 bytes/token at V=512→32768.
- **BPB**: `sum(token_bytes[ids])` = 3,024,593 = exact UTF-8 byte count; `evaluate_bpb`
  reproduced by hand on `d6` to 5.4e-08.
- **Cross-tokenizer PPL is invalid** — proven, not asserted: four real unigram LMs, same
  bytes. Token-PPL ranks them `[512, 2048, 8192, 32768]`; BPB ranks them **exactly reversed**.
- **Capacity**: three vocab-dependent tensor groups, not two — `wte`, `lm_head`, and
  **3× `value_embeds`**. 85.6% of d6's parameters are vocab-dependent.

## Week 2 · Optimization & architecture — [w2_ablations/](w2_ablations/)

| activity | script | what it proves |
|---|---|---|
| `w2-control-design` | [01_control_design.py](w2_ablations/01_control_design.py) | `--weight-decay` reaches only **14.4%** of parameters (Muon groups); the other 85.6% carry hardcoded AdamW decays the flag cannot touch. Fixed-step depth/width arms span 1.00–1.52× FLOPs |
| `w2-implementation-review` | [test_tied_embeddings.py](w2_ablations/test_tied_embeddings.py) (10 tests) · [03_implementation_review.py](w2_ablations/03_implementation_review.py) | full tied-embeddings contract in `gpt.py`/`checkpoint_manager.py`/`base_train.py`, default-off, baseline reproduced bit-identically |
| `w2-portfolio-defense` | [run_noise_floor.sh](w2_ablations/run_noise_floor.sh) · [run_portfolio.sh](w2_ablations/run_portfolio.sh) · [04_portfolio.py](w2_ablations/04_portfolio.py) | 4 real arms vs a 3-seed control, effect sizes in σ, winner's-curse Monte Carlo |

**Measured portfolio** (600 steps each, σ = 0.003651):

| arm | val_bpb | Δ vs control | σ | verdict |
|---|---|---|---|---|
| lr-double (all LRs ×2) | 1.451009 | **−0.075775** | −20.8 | REAL better |
| control (3 seeds) | 1.526784 | — | — | baseline |
| wd-zero | 1.530254 | +0.003470 | +1.0 | **NULL** |
| tied embeddings | 1.559273 | +0.032489 | +8.9 | REAL worse (raw) |
| lr-half (all LRs ×0.5) | 1.612853 | +0.086069 | +23.6 | REAL worse |

The LR response is **monotone across 0.5× / 1× / 2×**, which is far harder to explain by
chance than a single two-arm gap. The winner's-curse Monte Carlo on the measured σ shows an
8-arm all-null portfolio produces a winner that looks 1.4σ better by selection alone, and
clears 2σ **16.4%** of the time — so a 2σ result from a screen is not a result, while these
9–24σ effects are far outside the selection band.

**A real bug the contract caught**: tying only in `__init__` passed all 8 original unit
tests but `to_empty()` silently breaks the alias on the meta-device path `base_train`
uses. The smoke run reported the *untied* parameter count while metadata claimed
`tie_embeddings=True`. Fixed with `GPT.tie_weights()`; regression tests 9 and 10 pin
both the `to_empty` and the `load_state_dict(assign=True)` paths.

## Week 3 · Supervised fine-tuning — [w3_sft/](w3_sft/)

| activity | script | what it proves |
|---|---|---|
| `w3-sft-mechanism` | [01_sft_mechanism.py](w3_sft/01_sft_mechanism.py) | real GSM8K conversation traced token by token; tool CALLS get mask=1, tool OUTPUTS mask=0; a real 32-row packed batch is **67.9% supervised**, with 4/32 rows supervising nothing |
| `w3-sft-evaluation` | [02_sft_evaluation.py](w3_sft/02_sft_evaluation.py) | same-parent base vs SFT on identical bytes |

- **Forgetting: +0.5216 bpb (+45.6%)** on pretraining val — 143σ. Invisible to any chat metric.
- **Format gain**: greedy scores 0/8 for *both* models. A temperature sweep separates the
  causes — SFT terminates **17/32** at T=1.0 vs base **2/32**. The greedy 0/8 is a decoding
  failure, not a missing capability, and reporting it alone would have hidden the whole effect.

## Week 4 · DPO — [w4_dpo/](w4_dpo/)

DPO does not exist in nanochat; implemented here as [`nanochat/dpo.py`](../nanochat/dpo.py).

| activity | script | what it proves |
|---|---|---|
| `w4-dpo-derivation` | [01_derivation.py](w4_dpo/01_derivation.py) | derivation from the Gibbs solution through the partition-term cancellation; worked example hand-computed then matched to the implementation at 1.1e-16; gradient formula verified to 1e-12 |
| `w4-dpo-implementation` | [test_dpo.py](w4_dpo/test_dpo.py) (8 tests) · [02_data_and_drift_audit.py](w4_dpo/02_data_and_drift_audit.py) | hand-calc match, swap test, neutral pair (= log 2), frozen reference, masking, β limits |

Two preference datasets built from GSM8K and audited **before** any DPO run:
D2 (model-vs-reference) is **85% solvable by response length alone** and has
corr(length diff, log-prob diff) = **−0.902**; D1 (corrupted-answer) is length-controlled
but π_ref already ranks it 40/40, so it would saturate immediately.

## Week 5 · Verifiable-reward RL — [w5_rl/](w5_rl/)

| activity | script | what it proves |
|---|---|---|
| `w5-rl-objective` | [01_rl_objective.py](w5_rl/01_rl_objective.py) | one GSM8K item → 8 rollouts → group-mean advantage → `pg_obj` recomputed two ways, matching to 7.5e-09 |
| `w5-rl-audit` | [02_rl_audit.py](w5_rl/02_rl_audit.py) | the verifier attacked with 13 adversarial strings |

- **RL at d6 is an empty intervention**: 0 of 12 items had non-unanimous rewards
  (0 successes in 96 samples) → **100% of rollout compute yields exactly zero gradient.**
- **Verifier failures**: `'Reasoning is wrong but #### 16'` scores **1.0** — reasoning is
  never checked. `'#### 16.0'` and `'####16'` score **0.0** while being right, so part of
  what RL learns is literal marker formatting.

## Week 6 · Capstone — [w6_capstone/](w6_capstone/)

| activity | script | what it does |
|---|---|---|
| `w6-capstone-protocol` | [01_freeze_protocol.py](w6_capstone/01_freeze_protocol.py) · [entrypoint.sh](w6_capstone/entrypoint.sh) · [03_sealed_test.py](w6_capstone/03_sealed_test.py) | writes hashed [protocol.json](w6_capstone/protocol.json); re-running **verifies** the freeze and reports drift as a violation |
| `w6-capstone-defense` | [02_capstone_defense.py](w6_capstone/02_capstone_defense.py) | the claim, the ladder, per-token vs per-minute, five discards, the scaling caveat |

The seal is **enforced in code**: `NANOCHAT_SEALED_SHARDS` (added to `nanochat/dataset.py`)
removes shards 168/169 from train *and* val. `entrypoint.sh` verifies the freeze, times
itself, VOIDs overruns, and refuses to open the sealed test twice.

---

## Changes made to nanochat itself

All additive and default-off; the untied 30-step smoke run reproduces the pre-existing
`w0-smoke` val_bpb exactly (3.201590 / 2.994110 / 2.336091).

| file | change | default |
|---|---|---|
| `nanochat/gpt.py` | `GPTConfig.tie_embeddings`, `GPT.tie_weights()`, optimizer-coverage asserts, tying-aware `num_scaling_params` | `False` |
| `nanochat/checkpoint_manager.py` | re-tie after `load_state_dict(assign=True)` | no-op when untied |
| `scripts/base_train.py` | `--tie-embeddings` | `0` |
| `nanochat/common.py` | `NANOCHAT_SEED` env override | `42` |
| `nanochat/dataset.py` | `NANOCHAT_SEALED_SHARDS` env exclusion | unset |
| `nanochat/dpo.py` | new file — DPO loss + masked sequence log-probs | n/a |

`pytest tests/` passes unchanged (43 passed). The one failure,
`test_execution.py::test_memory_limit`, reproduces on a clean checkout and is a macOS
RLIMIT issue unrelated to any of this.
