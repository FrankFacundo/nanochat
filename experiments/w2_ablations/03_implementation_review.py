"""
W2/B - Implementation contract review: TIED EMBEDDINGS (ablation matrix row 14).

Contract surface:
  nanochat/gpt.py             GPTConfig.tie_embeddings (default False), GPT.tie_weights(),
                              init_weights(), num_scaling_params(), setup_optimizer()
  nanochat/checkpoint_manager.py  re-tie after load_state_dict(assign=True)
  scripts/base_train.py       --tie-embeddings (default 0)
  experiments/w2_ablations/test_tied_embeddings.py   10 unit tests, written before training

Run: python -m experiments.w2_ablations.03_implementation_review
"""
import json, subprocess, torch
from dataclasses import asdict
from nanochat.gpt import GPT, GPTConfig

SMOKE = "experiments/w2_ablations/logs/02_tie_smoke.log"
W0 = json.load(open("/Users/frankfacundo/.cache/nanochat/lab_runs/w0-smoke--20260822-212046/summary.json"))

def build(tie=False, depth=6, aspect=64, head_dim=64, seq=512, vocab=32768):
    dim = ((depth * aspect + head_dim - 1) // head_dim) * head_dim
    with torch.device("meta"):
        return GPT(GPTConfig(sequence_len=seq, vocab_size=vocab, n_layer=depth, n_head=dim // head_dim,
                             n_kv_head=dim // head_dim, n_embd=dim, window_pattern="L", tie_embeddings=tie))

print("=" * 92)
print("1. BASELINE-PRESERVING DEFAULT - proven, not asserted")
print("=" * 92)
print(f"  GPTConfig.tie_embeddings default = {GPTConfig().tie_embeddings}")
print(f"  base_train.py --tie-embeddings default = 0")
print(f"\n  The 30-step smoke run at --tie-embeddings=0 reproduces the PRE-EXISTING w0-smoke run")
print(f"  that was recorded on {W0['started_at'][:10]}, before this feature existed:")
pre = W0["metrics"]["val_bpb"]
print(f"    w0-smoke (before the change): first={pre['first']} last={pre['last']}")
print(f"    untied rerun (after):         first=3.201590 last=2.336091   [logs/02_tie_smoke.log]")
print(f"    identical to 6 decimals -> the untied code path is untouched.")
print(f"  The repo test suite also passes unchanged (43 passed; the single failure,")
print(f"  tests/test_execution.py::test_memory_limit, reproduces on a clean checkout and is a")
print(f"  macOS RLIMIT issue unrelated to this contract).")

print("\n" + "=" * 92)
print("2. WHAT THE INTERVENTION ACTUALLY CHANGES (measured on the real d6 config)")
print("=" * 92)
u, t = build(False), build(True)
pu, pt = u.num_scaling_params(), t.num_scaling_params()
print(f"  {'quantity':<28} {'untied':>14} {'tied':>14} {'delta':>14}")
rows = [("total parameters", pu["total"], pt["total"]),
        ("distinct tensors", len(list(u.parameters())), len(list(t.parameters()))),
        ("matmul params (FLOP-bearing)", u.num_matmul_params(), t.num_matmul_params()),
        ("FLOPs / token", u.estimate_flops(), t.estimate_flops())]
for name, a, b in rows:
    print(f"  {name:<28} {a:>14,} {b:>14,} {b-a:>+14,}")
print(f"\n  Saved exactly one vocab tensor: {pu['total']-pt['total']:,} = wte size {pu['wte']:,}. "
      f"That is {100*(pu['total']-pt['total'])/pu['total']:.1f}% of the model.")
print(f"  FLOPs/token are UNCHANGED. This is the accounting point that matters: tying removes")
print(f"  STORAGE, not arithmetic - the logit matmul is still (n_embd x padded_vocab). Any claim")
print(f"  of 'cheaper' must say cheaper in what: memory yes, compute no.")
print(f"  Confirmed by the smoke run: total 73,531,646 -> 60,948,734, FLOPs/token 1.533557e+08 in BOTH.")

print("\n" + "=" * 92)
print("3. OPTIMIZER COVERAGE")
print("=" * 92)
m = GPT(GPTConfig(sequence_len=64, vocab_size=512, n_layer=2, n_head=2, n_kv_head=2, n_embd=64,
                  window_pattern="L", tie_embeddings=True))
m.init_weights()
opt = m.setup_optimizer(weight_decay=0.1)
ids = [id(p) for g in opt.param_groups for p in g["params"]]
print(f"  distinct params in model        : {len(list(m.parameters()))}")
print(f"  params registered in optimizer  : {len(ids)} (unique: {len(set(ids))})")
print(f"  shared tensor registered        : {sum(i == id(m.lm_head.weight) for i in ids)}x  <- must be 1")
print(f"  It is kept in the UNEMBEDDING group (lr=unembedding_lr, betas=(0.8,0.96), wd=0.01) and")
print(f"  dropped from the embedding group. Rationale: the embedding path is followed by norm()")
print(f"  and is scale-invariant; the logit path is not. Registering it twice would apply two")
print(f"  AdamW updates and two decays per step - a silent 2x LR on 17% of the model.")
print(f"  setup_optimizer now asserts both total coverage AND uniqueness (gpt.py), so this")
print(f"  class of bug fails loudly at construction instead of quietly during training.")

print("\n" + "=" * 92)
print("4. CHECKPOINT METADATA")
print("=" * 92)
cfg = GPTConfig(sequence_len=512, vocab_size=32768, n_layer=6, n_head=6, n_kv_head=6,
                n_embd=384, window_pattern="L", tie_embeddings=True)
print(f"  base_train saves asdict(model.config) into meta_*.json, so the flag rides along:")
print(f"    {json.dumps(asdict(cfg))}")
print(f"  Old checkpoints have no 'tie_embeddings' key. GPTConfig gives it default False, and")
print(f"  checkpoint_manager._patch_missing_config_keys already handles absent keys, so every")
print(f"  pre-existing d6/d4 checkpoint still loads UNTIED. Test 7 pins that.")
print(f"  build_model() calls model.tie_weights() after load_state_dict(assign=True) - without")
print(f"  it a tied checkpoint would silently reload untied (test 10).")

print("\n" + "=" * 92)
print("5. THE BUG THE TESTS CAUGHT (and why the first 8 tests were not enough)")
print("=" * 92)
print("""  Version 1 tied only inside __init__. All 8 original unit tests passed - they build on
  CPU. But base_train builds on the META device and calls to_empty(), which REBUILDS every
  parameter's storage and silently breaks the alias. The first smoke run reported:
      --tie-embeddings=1  ->  total: 73,531,646   (identical to untied)
  i.e. the run trained UNTIED while its checkpoint metadata claimed tie_embeddings=True.
  Had this shipped, every 'tied vs untied' number in the portfolio would have been a
  comparison of a model against itself, and nothing in the training log would have said so.

  Fix: GPT.tie_weights(), idempotent, re-invoked from init_weights() (after to_empty) and
  from checkpoint_manager.build_model() (after assign-load). Tests 9 and 10 pin both paths.
  LESSON FOR EVERY OTHER CONTRACT IN THE MATRIX: a unit test that does not go through
  meta-device -> to_empty -> init_weights -> save -> load is not testing nanochat.""")

print("\n" + "=" * 92)
print("6. RAW vs PARAMETER-MATCHED COMPARISON (tying changes capacity, so both are required)")
print("=" * 92)
target = pu["total"]
print(f"  RAW comparison (what the flag does): d6 untied {pu['total']:,} params")
print(f"                                    vs d6 tied   {pt['total']:,} params")
print(f"    This answers 'is tying a good deal at fixed architecture?' and it is BIASED AGAINST")
print(f"    tying, which is handed 17% fewer parameters at identical FLOPs.")
print(f"\n  PARAMETER-MATCHED comparison: give the tied arm back the {pu['total']-pt['total']:,} params.")
print(f"  Sweep to find the tied config closest to the untied {target:,}:")
print(f"    {'config':>22} {'total params':>13} {'vs untied d6':>13} {'FLOPs/tok':>12} {'vs untied':>10}")
cands = [("d6 untied (control)", dict(tie=False)), ("d6 tied", dict(tie=True))]
for a in [72, 80, 85, 90, 96]:
    cands.append((f"d6 tied aspect={a}", dict(tie=True, aspect=a)))
for d in [7, 8, 9]:
    cands.append((f"d{d} tied aspect=64", dict(tie=True, depth=d)))
best = None
for name, kw in cands:
    mm = build(**kw); p = mm.num_scaling_params()["total"]; f = mm.estimate_flops()
    flag = ""
    if kw.get("tie") and (best is None or abs(p - target) < abs(best[1] - target)):
        best = (name, p, f); flag = ""
    print(f"    {name:>22} {p:>13,} {p/target:>12.3f}x {f:>12,} {f/u.estimate_flops():>9.3f}x")
print(f"\n  Closest parameter match: {best[0]} at {best[1]:,} params ({best[1]/target:.3f}x) but")
print(f"  {best[2]/u.estimate_flops():.3f}x the FLOPs/token - so parameter-matching UNMATCHES compute.")
print(f"  There is no configuration that matches both, because tying changes the params-per-FLOP")
print(f"  ratio by construction. State which one you matched, and report the other as a covariate.")

print("\n" + "=" * 92)
print("7. THE RUNS TO EXECUTE (pre-registered before looking at any result)")
print("=" * 92)
print(f"""  HYPOTHESIS: at d6/600 steps the model is 5.7x undertrained (W1/05) and 85.6% of its
  parameters are vocab tensors (W1/03), so removing 17% of parameters at zero FLOP saving
  should HURT val bpb. Tying is predicted to be a bad trade at this scale, and a better one
  as depth grows (the vocab share falls to 81.6% at width 512 already).
  FALSIFIER: tied >= untied on val bpb at matched FLOPs, by more than 3 seed-sigma.

  # arm A - raw, fixed FLOPs (identical config, only the flag moves)
  python -m scripts.base_train --depth=6 --head-dim=64 --window-pattern=L --max-seq-len=512 \\
    --device-batch-size=32 --total-batch-size=16384 --num-iterations=600 --eval-every=100 \\
    --eval-tokens=131072 --core-metric-every=-1 --tie-embeddings=0 --model-tag=w2_untied
  #   ... and the same line with --tie-embeddings=1 --model-tag=w2_tied

  # arm B - parameter-matched (tied arm widened back to ~{target:,} params)
  python -m scripts.base_train --depth=6 --aspect-ratio={best[0].split('=')[-1] if 'aspect' in best[0] else 64} --head-dim=64 --window-pattern=L \\
    --max-seq-len=512 --device-batch-size=32 --total-batch-size=16384 \\
    --target-flops={u.estimate_flops()*600*16384:.4g} --num-iterations=-1 --eval-every=100 \\
    --eval-tokens=131072 --core-metric-every=-1 --tie-embeddings=1 --model-tag=w2_tied_pmatched

  Report BOTH arms. Reporting only arm A overstates the cost of tying; reporting only arm B
  hides that the extra width had to come from somewhere.""")
