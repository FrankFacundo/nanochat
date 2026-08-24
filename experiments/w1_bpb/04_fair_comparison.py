"""
W1/E4 - Design the fair follow-up, with the knobs computed from measured numbers.

A 600-step token-matched pilot is confounded twice over:
  - equal TOKENS != equal DATA   (a vocab-32768 token carries 4.68 bytes, a
    vocab-8192 token 3.99 -> the big-vocab arm reads ~17% more text)
  - equal TOKENS != equal COMPUTE (FLOPs/token differ by the lm_head term)
This script turns the measured bytes/token (01) and FLOPs/token (03) into the
exact iteration counts for a byte-matched arm and an isoFLOP sweep.

Run: python -m experiments.w1_bpb.04_fair_comparison
"""
import json, os, torch
from nanochat.gpt import GPT, GPTConfig

OUT = os.path.dirname(os.path.abspath(__file__))
DEPTH, HEAD_DIM, ASPECT, SEQ_LEN, WINDOW = 6, 64, 64, 512, "L"
TOTAL_BATCH = 16384          # tokens/step, from runs/runcpu.sh
PILOT_STEPS = 600
ARMS = [8192, 32768]

bpt = {int(k): v["bytes_per_token"]
       for k, v in json.load(open(f"{OUT}/logs/01_results.json"))["results"].items()}

def model_stats(vocab):
    dim = ((DEPTH * ASPECT + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    cfg = GPTConfig(sequence_len=SEQ_LEN, vocab_size=vocab, n_layer=DEPTH, n_head=dim // HEAD_DIM,
                    n_kv_head=dim // HEAD_DIM, n_embd=dim, window_pattern=WINDOW)
    with torch.device("meta"):
        m = GPT(cfg)
    return m.estimate_flops(), m.num_scaling_params()["total"]

st = {v: model_stats(v) for v in ARMS}

print("=" * 78)
print("1. WHY THE 600-STEP TOKEN-MATCHED PILOT IS CONFOUNDED")
print("=" * 78)
print(f"   Both arms: d{DEPTH}, {TOTAL_BATCH:,} tok/step, {PILOT_STEPS} steps = "
      f"{TOTAL_BATCH*PILOT_STEPS:,} tokens each.\n")
print(f"   {'vocab':>7} {'bytes/tok':>10} {'bytes seen':>14} {'FLOPs/tok':>12} {'total FLOPs':>13} {'params':>12}")
for v in ARMS:
    f, p = st[v]
    tot = TOTAL_BATCH * PILOT_STEPS
    print(f"   {v:>7} {bpt[v]:>10.3f} {tot*bpt[v]:>14,.0f} {f:>12,} {f*tot:>13.3e} {p:>12,}")
db = (bpt[ARMS[1]] / bpt[ARMS[0]] - 1) * 100
df = (st[ARMS[1]][0] / st[ARMS[0]][0] - 1) * 100
print(f"\n   -> vocab {ARMS[1]} silently gets {db:+.1f}% more TEXT and spends {df:+.1f}% more COMPUTE.")
print(f"      Any BPB gap it shows is a mixture of vocab effect + data effect + compute effect.")

print("\n" + "=" * 78)
print("2. FOLLOW-UP ARM A - BYTE-MATCHED (equal data, compute reported as covariate)")
print("=" * 78)
BUDGET_BYTES = TOTAL_BATCH * PILOT_STEPS * bpt[ARMS[1]]
print(f"   Fix the data budget at D = {BUDGET_BYTES:,.0f} bytes of ClimbMix for every arm.")
print(f"   steps = D / (bytes_per_token * {TOTAL_BATCH:,})\n")
print(f"   {'vocab':>7} {'steps':>8} {'tokens':>13} {'bytes':>14} {'FLOPs':>12}")
for v in ARMS:
    steps = round(BUDGET_BYTES / (bpt[v] * TOTAL_BATCH))
    toks = steps * TOTAL_BATCH
    print(f"   {v:>7} {steps:>8,} {toks:>13,} {toks*bpt[v]:>14,.0f} {st[v][0]*toks:>12.3e}")
    print(f"           python -m scripts.base_train --depth={DEPTH} --head-dim={HEAD_DIM} "
          f"--window-pattern={WINDOW} \\\n"
          f"             --max-seq-len={SEQ_LEN} --device-batch-size=32 --total-batch-size={TOTAL_BATCH} \\\n"
          f"             --num-iterations={steps} --eval-every=100 --eval-tokens=524288 "
          f"--core-metric-every=-1 \\\n"
          f"             --model-tag=v{v}_bytematched   # after tok_train --vocab-size={v}")

print("\n" + "=" * 78)
print("3. FOLLOW-UP ARM B - isoFLOP SWEEP (the actually decisive test)")
print("=" * 78)
print("   Byte-matching and FLOP-matching cannot both hold by tuning steps alone, so run")
print("   each arm at 3 step counts, plot BPB vs cumulative FLOPs, and read both arms off")
print("   at a COMMON FLOP budget. base_train.py already supports --target-flops directly.\n")
budgets = [4e15, 8e15, 1.6e16]
print(f"   {'FLOP budget':>12} " + " ".join(f"{'v'+str(v)+' steps':>14}" for v in ARMS) +
      "   " + " ".join(f"{'v'+str(v)+' bytes':>14}" for v in ARMS))
for Fb in budgets:
    steps = {v: round(Fb / (st[v][0] * TOTAL_BATCH)) for v in ARMS}
    print(f"   {Fb:>12.1e} " + " ".join(f"{steps[v]:>14,}" for v in ARMS) + "   " +
          " ".join(f"{steps[v]*TOTAL_BATCH*bpt[v]:>14,.0f}" for v in ARMS))
print(f"\n   e.g.  python -m scripts.base_train --depth={DEPTH} --head-dim={HEAD_DIM} "
      f"--window-pattern={WINDOW} \\\n"
      f"           --max-seq-len={SEQ_LEN} --device-batch-size=32 --total-batch-size={TOTAL_BATCH} \\\n"
      f"           --target-flops=8e15 --num-iterations=-1 --model-tag=v32768_iso8e15")

print("\n" + "=" * 78)
print("4. WHAT MAKES THE READOUT VALID")
print("=" * 78)
print("   a) Metric: val BPB only. Token loss and token PPL are uninterpretable here")
print("      (experiment 02: they reverse the ranking). base_train already logs val/bpb.")
print("   b) Same held-out bytes: both arms evaluate the same val shards. --eval-tokens is")
print("      in TOKENS, so at equal --eval-tokens the arms score DIFFERENT amounts of text;")
print("      set eval_tokens_v = round(EVAL_BYTES / bytes_per_token_v) so the denominators")
print("      cover the same corpus. Suggested EVAL_BYTES = 2,000,000:")
for v in ARMS:
    n = round(2_000_000 / bpt[v]); n -= n % (32 * SEQ_LEN)
    print(f"        vocab {v}: --eval-tokens={n:,}  (~{n*bpt[v]:,.0f} bytes)")
print("   c) Noise floor first: 3 seeds of ONE arm at the pilot budget, report the")
print("      std of final val BPB. A vocab gap smaller than ~2 std is not a result.")
print("   d) Everything else frozen: same data order, same LR schedule, same seq_len,")
print("      same window pattern, same optimizer. Only --vocab-size differs.")
print("   e) Calibrate the claim: the honest conclusion is 'at d6 / this FLOP budget /")
print("      this corpus, vocab X wins by Y bpb (+/- noise)'. It does NOT extrapolate to")
print("      other depths - the vocab-dependent share of parameters shrinks as depth grows")
print("      (experiment 03: 85.6% of d6 params are vocab-dependent at V=32768), so a")
print("      large vocab is far cheaper, relatively, for a d20 model than for this one.")
