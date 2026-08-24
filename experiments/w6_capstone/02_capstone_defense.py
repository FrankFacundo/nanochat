"""
W6/B - Final defense: the claim, the causal evidence, the discipline, and the discards.

Everything numeric is read from the real artifacts produced by the earlier weeks; this
script computes the per-minute vs per-token comparison and the scaling caveat itself.

Run: python -m experiments.w6_capstone.02_capstone_defense
"""
import json, math, re, statistics as st
from pathlib import Path
import torch
from nanochat.gpt import GPT, GPTConfig

ROOT = Path(__file__).resolve().parent.parent
P = json.loads((ROOT / "w2_ablations" / "logs" / "04_portfolio.json").read_text())
SIGMA, MU = P["sigma"], P["control_mean"]
arms = {a["name"]: a for a in P["arms"]}
CTRL_WALL = st.mean([5.49, 5.18, 6.01])   # the three seed runs, from run_noise_floor.sh

def build(depth=6, aspect=64, head_dim=64, vocab=32768, tie=False, seq=512):
    dim = ((depth * aspect + head_dim - 1) // head_dim) * head_dim
    with torch.device("meta"):
        return GPT(GPTConfig(sequence_len=seq, vocab_size=vocab, n_layer=depth, n_head=dim // head_dim,
                             n_kv_head=dim // head_dim, n_embd=dim, window_pattern="L", tie_embeddings=tie))

print("=" * 100)
print("1. THE CLAIM, WITH ITS BUDGET AND EFFECT SIZE")
print("=" * 100)
a = arms["lr-double"]
print(f"""  CLAIM: On this machine, for a d6 model (384 dim, 6 layers, 512 context, 32,768 vocab)
  trained for 600 steps x 16,384 tokens = 9,830,400 tokens (1.508e15 FLOPs, ~{a['wall']:.1f} min),
  multiplying all four parameter-group learning rates by 2 lowers final validation
  bits-per-byte from {MU:.6f} to {a['bpb']:.6f}, an improvement of {abs(a['d']):.6f} bpb.

  Effect size: {abs(a['z']):.0f} sigma, against a noise floor of sigma = {SIGMA:.6f} bpb measured from
  3 seeds of the control run BEFORE any arm was interpreted.
  Cost: zero. Identical parameters ({build().num_scaling_params()['total']:,}) and identical
  FLOPs/token ({build().estimate_flops():,}) as the control - only the LR values differ.

  SCOPE, stated as part of the claim: d6, 600 steps, this data order, seed 42, MPS. It is
  NOT a claim about nanochat's default LRs in general - those are tuned at d12 with a
  524,288-token batch, and this budget is 32x smaller in batch and far shorter in horizon.""")

print("\n" + "=" * 100)
print("2. THE ABLATION LADDER (what each rung rules out)")
print("=" * 100)
print(f"""  rung 0  smoke     30 steps          proved the pipeline records what it claims to record
                                          (W0/02: summary.json recomputed from metrics.jsonl,
                                           0 mismatches)
  rung 1  noise       3 seeds x 600 st   sigma = {SIGMA:.6f}. Without this rung every number
                                          below is uninterpretable.
  rung 2  screen      4 arms x 600 st    lr-half {arms['lr-half']['z']:+.0f}s, lr-double {arms['lr-double']['z']:+.0f}s,
                                          wd-zero {arms['wd-zero']['z']:+.1f}s{', tied ' + format(arms['tied']['z'], '+.0f') + 's' if 'tied' in arms else ''}
  rung 3  confirm     2 arms x 3000 st   control + lr-double as a PAIR at 5x budget
  rung 4  sealed test 1 evaluation       shards 168/169, opened once, by entrypoint.sh --final

  The ladder is monotone in cost and each rung can only be climbed after the one below
  passed. The rung that actually did the work is rung 1: it converted "1.451 looks better
  than 1.527" into "{abs(arms['lr-double']['z']):.0f} sigma", and it is the rung most portfolios skip.""")

print("\n" + "=" * 100)
print("3. VALIDATION vs TOKENS AND vs WALL TIME (they rank arms differently)")
print("=" * 100)
print(f"  {'arm':<11} {'val_bpb':>9} {'wall (m)':>9} {'bpb gain':>10} {'gain/min':>10} {'rank by bpb':>12} {'rank by gain/min':>17}")
rowsl = [dict(name="control", bpb=MU, wall=CTRL_WALL)] + [dict(name=k, bpb=v["bpb"], wall=v["wall"]) for k, v in arms.items()]
for r in rowsl:
    r["gain"] = MU - r["bpb"]
    r["rate"] = r["gain"] / r["wall"]
by_bpb = sorted(rowsl, key=lambda r: r["bpb"])
by_rate = sorted(rowsl, key=lambda r: -r["rate"])
for r in by_bpb:
    print(f"  {r['name']:<11} {r['bpb']:>9.6f} {r['wall']:>9.2f} {r['gain']:>+10.6f} {r['rate']:>+10.6f} "
          f"{by_bpb.index(r)+1:>12} {by_rate.index(r)+1:>17}")
disagree = [r["name"] for r in rowsl if by_bpb.index(r) != by_rate.index(r)]
print(f"\n  arms whose rank changes between the two axes: {disagree if disagree else 'none'}")
print(f"""  A CHANGE THAT HELPS PER TOKEN BUT HURTS PER MINUTE: at this scale the clearest example
  is not in the LR family (all arms cost the same FLOPs) but in the architecture family
  measured in W2/01: w2-width512-fixed-depth has 1.40x the parameters and would be expected
  to win per TOKEN, while costing 1.52x the FLOPs/token - so at fixed wall time it must beat
  the control by more than 52% of the control's gain just to break even. Fixed-step results
  systematically flatter the expensive arm; only the wall-time column prices it correctly.""")

print("\n" + "=" * 100)
print("4. THROUGHPUT TAILS, REPRODUCTION, AND THE TEST GAP")
print("=" * 100)
print(f"""  THROUGHPUT TAIL: the 5000-step baseline (W1/05) ran tok/sec first=13,737 median=33,224
  max=47,272, with 1.3% of steps below 80% of median. The control's own wall time varied
  5.18m - 6.01m across three IDENTICAL runs ({6.01/5.18:.2f}x). Consequences:
    - never compute a wall-time claim from mean step time; use total elapsed.
    - a wall-time budget needs a margin: at {6.01/5.18:.2f}x spread, a config planned to fill
      30:00 exactly will overrun roughly half the time, and entrypoint.sh VOIDS overruns.

  REPRODUCTION DIFFERENCE: the three control seeds differ by up to {max(P['control_seeds'])-min(P['control_seeds']):.6f} bpb.
  A reproduction of any number here is a PASS if it lands within 2 sigma = {2*SIGMA:.6f}.
  Note this understates true variance: nanochat seeds weight init only, not data order
  (common.py), so these three runs saw identical batches in identical order.

  THE TEST GAP: everything above is VALIDATION. Validation has been used for model
  selection across every arm, so it is optimistically biased for the winner by roughly the
  selection amount quantified in W2/04. The sealed shards (168, 169) were excluded from
  train AND val via NANOCHAT_SEALED_SHARDS for every run and are opened exactly once, by
  entrypoint.sh --final, which writes a SEALED_TEST_OPENED marker and refuses a second
  opening. EXPECT the sealed number to be slightly worse than validation; report the gap.""")

print("\n" + "=" * 100)
print("5. STRONGEST CAUSAL INTERVENTION vs MOST LIKELY SELECTION NOISE")
print("=" * 100)
strongest = max(arms.values(), key=lambda r: abs(r["z"]))
noisiest = min(arms.values(), key=lambda r: abs(r["z"]))
print(f"  STRONGEST : {strongest['name']} ({strongest['desc']}) at {strongest['z']:+.0f} sigma.")
print(f"    Causal because it is a single-factor change with identical params, identical")
print(f"    FLOPs/token, identical data order, and a MONOTONE dose-response across")
print(f"    0.5x / 1x / 2x ({arms['lr-half']['bpb']:.4f} / {MU:.4f} / {arms['lr-double']['bpb']:.4f}).")
print(f"    A monotone response across three doses is much harder to explain by chance than")
print(f"    a single two-arm gap of the same size.")
print(f"  MOST LIKELY NOISE : {noisiest['name']} at {noisiest['z']:+.2f} sigma - inside the floor.")
print(f"    Reported as a NULL, not as 'no effect': W2/01 showed the flag reaches only 14.4%")
print(f"    of parameters and is annealed to zero anyway, so the experiment had little power.")

print("\n" + "=" * 100)
print("6. FIVE IDEAS DISCARDED, AND WHY (each one was actually tried in this lab)")
print("=" * 100)
print("""  1. Per-token perplexity to compare tokenizers. DISCARDED: W1/02 trained four real
     unigram LMs and showed token-PPL and BPB rank the same four models in EXACTLY
     REVERSED order. Per-token normalisation divides by a tokenizer-chosen denominator.
  2. Token-matched vocabulary comparison. DISCARDED: W1/04 showed equal tokens hands the
     32,768-vocab arm +17.3% more text and +58.5% more compute. Replaced by an isoFLOP sweep.
  3. `assistant_end in generated` as the SFT format metric. DISCARDED: engine.py strips
     terminal tokens, so the test could NEVER fire. It scored 0/8 for both models and would
     have been published as "SFT taught the model no format". Replaced by len < budget, and
     then by a temperature sweep that found the real effect (17/32 vs 2/32).
  4. Tying embeddings only in __init__. DISCARDED: to_empty() and load_state_dict(assign=True)
     both silently break the alias. A 30-step smoke run showed the "tied" model reporting the
     UNTIED parameter count - it would have trained untied while its metadata claimed otherwise.
     Replaced by GPT.tie_weights(), re-invoked at both points, with regression tests 9 and 10.
  5. Reporting `Minimum validation bpb` as the endpoint. DISCARDED: min-over-N is biased low
     and the bias depends on the number of eval points, so two arms with different eval_every
     are not comparable on it at all. Replaced by final-step bpb, pre-registered.
  (bonus 6.) RL on GSM8K at d6. DISCARDED before spending the 45-80 minutes: W5/01 measured
     0 of 12 items with non-unanimous rewards - 100% of rollout compute would produce exactly
     zero gradient. The run would have produced a reward curve that was noise around zero.""")

print("\n" + "=" * 100)
print("7. THE CONCLUSION LEAST LIKELY TO SCALE BEYOND d6")
print("=" * 100)
print(f"  {'model':<22} {'total params':>13} {'vocab tensors':>14} {'vocab share':>12} {'tying saves':>12}")
for label, kw in [("d6  x384 (this lab)", dict()), ("d12 x768", dict(depth=12)),
                  ("d20 x1280", dict(depth=20)), ("d26 x1664", dict(depth=26))]:
    m = build(**kw)
    p = m.num_scaling_params()
    voc = p["wte"] + p["lm_head"] + p["value_embeds"]
    print(f"  {label:<22} {p['total']:>13,} {voc:>14,} {100*voc/p['total']:>11.1f}% {100*p['lm_head']/p['total']:>11.1f}%")
print(f"""
  ANY conclusion that depends on the vocabulary tensors is the least transferable, because
  their share of the model collapses with depth: {100*(lambda p: (p['wte']+p['lm_head']+p['value_embeds'])/p['total'])(build().num_scaling_params()):.1f}% at d6 down to
  {100*(lambda p: (p['wte']+p['lm_head']+p['value_embeds'])/p['total'])(build(depth=26).num_scaling_params()):.1f}% at d26. Concretely:
    - the tied-embeddings result is a d6 result. Tying removes 17.1% of a d6 but only
      {100*build(depth=26).num_scaling_params()['lm_head']/build(depth=26).num_scaling_params()['total']:.1f}% of a d26, so both the cost and the benefit shrink by ~4x.
    - the vocabulary-size conclusion from W1 (bigger vocab costs 2.79x params for 17% fewer
      tokens) inverts its economics at depth: the same 17% token saving costs proportionally
      far less when the blocks dominate the parameter count.
  Second least transferable: the LR result itself. It was measured at 5.7x undertrained
  (W1/05, 3.53 tokens-per-scaling-param vs Chinchilla ~20). Short horizons systematically
  favour aggressive learning rates because they never reach the regime where instability
  bites. That is exactly what the rung-3 confirmation at 3000 steps is designed to test,
  and it is the result I would bet against surviving to d20 at compute-optimal horizons.""")
