"""
W2/A - Untangle a joint intervention. Two families, proven against the code.

  OPTIMIZATION family : weight decay  (catalog: w2-wd-zero / w1-pilot / w2-wd-double)
  ARCHITECTURE family : depth vs width (w2-depth8-fixed-width / w1-pilot / w2-width512-fixed-depth)

Every "hidden intervention" below is demonstrated by instantiating the real
optimizer and the real models, never by reading the docs.

Run: python -m experiments.w2_ablations.01_control_design
"""
import math, json, torch
from nanochat.gpt import GPT, GPTConfig

B_REF = 2**19
NOMINAL_WD, STEPS, TBS = 0.28, 600, 16384

def build(depth=6, aspect=64, head_dim=64, seq=512, window="L", vocab=32768, meta=True):
    dim = ((depth * aspect + head_dim - 1) // head_dim) * head_dim
    cfg = GPTConfig(sequence_len=seq, vocab_size=vocab, n_layer=depth, n_head=dim // head_dim,
                    n_kv_head=dim // head_dim, n_embd=dim, window_pattern=window)
    ctx = torch.device("meta") if meta else torch.device("cpu")
    with ctx:
        return GPT(cfg)

# ============================================================ FAMILY 1
print("=" * 94)
print("FAMILY 1 - WEIGHT DECAY (optimization)")
print("=" * 94)
print("""  CAUSAL QUESTION: does increasing weight decay on the transformer matrices reduce
    held-out bpb at a fixed 600-step / 9.83M-token budget, or does it only shrink the
    train-val gap without improving generalization at this (undertrained) horizon?
  MECHANISM: decay shrinks weights toward zero each step, lowering effective capacity
    and the norm of the residual stream, which can act as a regularizer - but only for
    the parameters it is actually applied to.
  PREDICTION: at 3.5 tokens-per-param (5.7x undertrained, see W1/05) the model is
    capacity-limited, not overfitting, so MORE decay should HURT: wd=0.56 > wd=0.28 > wd=0
    in final val bpb, with wd=0 within noise of wd=0.28.
  FALSIFIER: wd=0.56 beats wd=0.28 by more than 3 seed-sigma, or wd=0 is clearly worst.""")

print("\n  HIDDEN JOINT INTERVENTION #1 - which parameters --weight-decay actually reaches.")
m = build(meta=False)
m.to_empty(device="cpu"); m.init_weights()
opt = m.setup_optimizer(weight_decay=NOMINAL_WD)
ids = {id(p): n for n, p in m.named_parameters()}
print(f"    {'group':>6} {'kind':>6} {'weight_decay':>13} {'#params':>12}   representative tensors")
tot_decayed = 0
for i, g in enumerate(opt.param_groups):
    n = sum(p.numel() for p in g["params"])
    names = sorted({ids[id(p)].split(".h.")[0] if ".h." in ids[id(p)] else ids[id(p)] for p in g["params"]})
    if g["weight_decay"] == NOMINAL_WD:
        tot_decayed += n
    print(f"    {i:>6} {g['kind']:>6} {g['weight_decay']:>13} {n:>12,}   {', '.join(names)[:52]}")
total = sum(p.numel() for p in m.parameters())
print(f"\n    --weight-decay reaches {tot_decayed:,} of {total:,} params = {100*tot_decayed/total:.1f}%.")
print(f"    Every AdamW group carries a HARDCODED decay (0.01 / 0.001 / 0.01 / 0.05 / 0 / 0)")
print(f"    that the CLI flag cannot touch (nanochat/gpt.py setup_optimizer). So w2-wd-zero is")
print(f"    NOT 'no weight decay' - it is 'no decay on the 14.4% of params that are matrices,")
print(f"    unchanged decay on the 85.6% that are embeddings and lm_head'.")

print("\n  HIDDEN JOINT INTERVENTION #2 - the value you type is not the value applied.")
sp6, sp12 = build().num_scaling_params(), build(depth=12).num_scaling_params()
target_tokens = 12 * (sp6["transformer_matrices"] + sp6["lm_head"])
D_REF = 12 * (sp12["transformer_matrices"] + sp12["lm_head"])
scaled = lambda wd, B=TBS: wd * math.sqrt(B / B_REF) * (D_REF / target_tokens)
print(f"    base_train.py:308 rescales it: lambda = wd * sqrt(B/B_ref) * (D_ref/D_target)")
print(f"    base_train.py:392 then anneals it on a cosine: lambda(t) = lambda * 0.5*(1+cos(pi*t/T))")
print(f"    {'--weight-decay':>16} {'-> scaled':>12} {'lambda(0)':>11} {'lambda(T/2)':>12} {'lambda(T)':>11} {'time-avg':>10}")
for wd in [0.0, 0.28, 0.56]:
    s = scaled(wd)
    traj = [s * 0.5 * (1 + math.cos(math.pi * t / STEPS)) for t in range(STEPS + 1)]
    print(f"    {wd:>16} {s:>12.6f} {traj[0]:>11.6f} {traj[STEPS//2]:>12.6f} {traj[-1]:>11.6f} "
          f"{sum(traj)/len(traj):>10.6f}")
print(f"    The cosine drives lambda(T)=0 in every arm, so all arms END identical and differ")
print(f"    most early. Doubling the flag doubles the time-integral, not the endpoint - the")
print(f"    intervention is 'more decay during the first ~2/3 of training', which is a")
print(f"    different causal claim from 'more decay'.")

print("\n  HIDDEN JOINT INTERVENTION #3 - it is coupled to batch size and to the horizon.")
print(f"    {'total_batch_size':>17} {'scaled lambda @0.28':>21}      {'num_iterations':>15} {'scaled lambda @0.28':>21}")
for B, it in [(8192, 600), (16384, 1250), (32768, 10000)]:
    print(f"    {B:>17,} {scaled(0.28, B):>21.6f}      {it:>15,} {scaled(0.28):>21.6f}")
print(f"    (the horizon enters through D_target, which depends on the MODEL not the run,")
print(f"     so changing --num-iterations does NOT rescale lambda - but changing --depth does.")
print(f"     A wd ablation is therefore only valid within one fixed depth and batch size.)")
print(f"\n  CORRECT BUDGET AXIS: fixed STEPS. Weight decay is a per-update operation, so the")
print(f"    comparison must hold the number of updates constant. All three arms are already")
print(f"    600 steps x 16,384 tokens with identical architecture, so params and FLOPs/token")
print(f"    are identical too - this is the one family where fixed-step IS compute-matched.")

# ============================================================ FAMILY 2
print("\n" + "=" * 94)
print("FAMILY 2 - DEPTH vs WIDTH (architecture)")
print("=" * 94)
print("""  CAUSAL QUESTION: at a fixed compute budget, is bpb better spent on more layers
    (d8 x 384) or more width (d6 x 512) relative to the d6 x 384 control?
  MECHANISM: depth adds sequential nonlinear composition; width adds per-layer capacity
    and attention head count. They cost compute differently: depth is linear in layers,
    width is quadratic in the block matrices but linear in the vocab tensors.
  PREDICTION: at 600 steps every arm is deeply undertrained, so the arm with the most
    parameters per FLOP wins on bpb-per-step and the SMALLEST arm wins on bpb-per-minute.
  FALSIFIER: the ranking is identical under fixed-step and fixed-FLOP accounting.""")

arms = {"w2-depth8-fixed-width": dict(depth=8, aspect=48),
        "w1-pilot (control)": dict(),
        "w2-width512-fixed-depth": dict(depth=6, aspect=85)}
print(f"\n  {'arm':>25} {'layers':>7} {'n_embd':>7} {'heads':>6} {'blocks':>12} {'total params':>13} {'FLOPs/tok':>12}")
info = {}
for name, kw in arms.items():
    mm = build(**kw); p = mm.num_scaling_params(); f = mm.estimate_flops()
    info[name] = (p, f)
    print(f"  {name:>25} {mm.config.n_layer:>7} {mm.config.n_embd:>7} {mm.config.n_head:>6} "
          f"{p['transformer_matrices']:>12,} {p['total']:>13,} {f:>12,}")

cp, cf = info["w1-pilot (control)"]
print(f"\n  WHY FIXED-STEP IS NOT COMPUTE-MATCHED at {STEPS} steps x {TBS:,} tokens:")
print(f"  {'arm':>25} {'total FLOPs':>12} {'vs control':>11} {'params vs ctrl':>15} {'FLOP-matched steps':>19}")
for name, (p, f) in info.items():
    tot = f * STEPS * TBS
    fair = round(cf * STEPS * TBS / (f * TBS))
    print(f"  {name:>25} {tot:>12.3e} {tot/(cf*STEPS*TBS):>10.2f}x {p['total']/cp['total']:>14.2f}x {fair:>19,}")
print(f"\n  => at 600 fixed steps the width-512 arm is handed {info['w2-width512-fixed-depth'][1]/cf:.2f}x the compute")
print(f"     and {info['w2-width512-fixed-depth'][0]['total']/cp['total']:.2f}x the parameters. If it wins, nothing has been shown.")
print(f"     Run each arm with --target-flops=1.508e15 --num-iterations=-1 instead, or use the")
print(f"     FLOP-matched step counts above.")

print(f"\n  THE nanochat-SPECIFIC TRAP: 'params' is dominated by vocab tensors, not architecture.")
for name, (p, f) in info.items():
    voc = p["wte"] + p["lm_head"] + p["value_embeds"]
    print(f"    {name:>25}: vocab tensors {voc:>11,} ({100*voc/p['total']:>4.1f}%)  "
          f"architecture {p['transformer_matrices']:>10,} ({100*p['transformer_matrices']/p['total']:>4.1f}%)")
print(f"    Total parameter count is therefore almost useless for ranking these arms - it moves")
print(f"    {info['w2-width512-fixed-depth'][0]['total']/cp['total']:.2f}x while the thing under test (blocks) moves "
      f"{info['w2-width512-fixed-depth'][0]['transformer_matrices']/cp['transformer_matrices']:.2f}x.")
print(f"    base_train.py:271 already knows this: it uses transformer_matrices + lm_head as")
print(f"    'scaling params'. Report THAT, plus FLOPs/token, and never bare total params.")

# ============================================================ HEADS / CONTEXT
print("\n" + "=" * 94)
print("APPENDIX - why head-count and context arms need different care again")
print("=" * 94)
for name, kw in [("w1-pilot", {}), ("w2-heads-4", dict(head_dim=96)), ("w2-heads-3", dict(head_dim=128)),
                 ("w2-context-256", dict(seq=256)), ("w2-context-1024", dict(seq=1024))]:
    mm = build(**kw)
    hd = mm.config.n_embd // mm.config.n_head
    print(f"  {name:>16} heads={mm.config.n_head} head_dim={hd:>3} seq={mm.config.sequence_len:>5} "
          f"params={mm.num_scaling_params()['total']:>11,} FLOPs/tok={mm.estimate_flops():>11,}")
print(f"  Heads: params and FLOPs are essentially IDENTICAL across 3/4/6 heads (n_embd is fixed,")
print(f"    so q/k/v/proj shapes are unchanged; only ve_gate differs by a few dozen params).")
print(f"    So the head ablation IS compute-matched at fixed steps - but head_dim changes from")
print(f"    64 to 96 to 128, which changes the RoPE frequency spectrum (gpt.py _precompute_rotary_")
print(f"    embeddings uses head_dim). 'More heads' and 'different positional encoding' are")
print(f"    confounded, and no budget axis fixes that; only a control that holds head_dim fixed does.")
print(f"  Context: FLOPs/token moves with seq_len through the attention term, AND device_batch_size")
print(f"    is changed in the catalog to keep memory constant (256->bs64, 1024->bs16), AND the")
print(f"    packing distribution changes because documents are bin-packed into sequences.")
print(f"    Three interventions in one flag.")
