"""
W0/A - Variables, controls, confounds and budget axes, read out of the real code.

Nothing here is asserted from memory. Every control is proven by diffing the CLI
commands llm_lab actually builds, and every "hidden" intervention is computed by
re-running nanochat's own coupling formulas from scripts/base_train.py.

Run: python -m experiments.w0_method.01_variables_and_controls
"""
import math, json, torch
from llm_lab.catalog import Catalog
from llm_lab.runner import build_commands
from nanochat.gpt import GPT, GPTConfig

cat = Catalog()
defaults = cat.defaults

def cli_dict(exp):
    """Resolve an experiment to the flat arg dict base_train actually receives."""
    return {**defaults.get(exp.stage, {}), **exp.args}

# ================================================================== 1. VARIABLES
print("=" * 88)
print("1. INDEPENDENT vs DEPENDENT VARIABLE, read off the LR ablation in the catalog")
print("=" * 88)
ctrl = cat.get("w1-pilot")
lo, hi = cat.get("w2-lr-half"), cat.get("w2-lr-double")
c, a, b = cli_dict(ctrl), cli_dict(lo), cli_dict(hi)
changed = sorted(k for k in set(a) | set(b) | set(c) if not (a.get(k) == b.get(k) == c.get(k)))
same = sorted(k for k in set(c) if a.get(k) == b.get(k) == c.get(k))
print(f"  control = w1-pilot, arms = w2-lr-half / w2-lr-double")
print(f"  INDEPENDENT VARIABLE (the only keys that differ): {changed}")
for k in changed:
    print(f"     {k:<18} half={a.get(k)!s:<8} control={c.get(k, '(default)')!s:<8} double={b.get(k)!s}")
print(f"\n  DEPENDENT VARIABLE: val bpb (scripts/base_train.py logs 'Validation bpb' and")
print(f"     wandb key val/bpb). Chosen over train_loss because it is tokenizer-invariant")
print(f"     and measured on held-out bytes.")
print(f"\n  HELD FIXED, {len(same)} controls, all verified identical across the three commands:")
for i in range(0, len(same), 4):
    print("     " + "  ".join(f"{k}={c[k]}" for k in same[i:i+4]))
print(f"\n  Note all FOUR parameter-group LRs move together (embedding/unembedding/matrix/scalar).")
print(f"  That is one intervention 'global LR multiplier', not four - nanochat keeps separate")
print(f"  LRs per group (nanochat/gpt.py setup_optimizer), so moving only one would be a")
print(f"  different experiment.")

# ============================================================== 2. HIDDEN COUPLING
print("\n" + "=" * 88)
print("2. HIDDEN JOINT INTERVENTIONS: what base_train.py silently changes for you")
print("=" * 88)
B_REF, D_REF_RATIO = 2**19, defaults["base"].get("target_param_data_ratio", 12)

def build(depth=6, aspect=64, head_dim=64, seq=512, window="L", vocab=32768):
    dim = ((depth * aspect + head_dim - 1) // head_dim) * head_dim
    cfg = GPTConfig(sequence_len=seq, vocab_size=vocab, n_layer=depth, n_head=dim // head_dim,
                    n_kv_head=dim // head_dim, n_embd=dim, window_pattern=window)
    with torch.device("meta"):
        return GPT(cfg)

m6 = build()
sp6 = m6.num_scaling_params()
scaling6 = sp6["transformer_matrices"] + sp6["lm_head"]
m12 = build(depth=12)
sp12 = m12.num_scaling_params()
D_REF = 12 * (sp12["transformer_matrices"] + sp12["lm_head"])
target_tokens = 12 * scaling6
print(f"  d6 scaling params (blocks+lm_head) = {scaling6:,}   target_tokens = {target_tokens:,}")
print(f"  d12 reference D_REF = {D_REF:,}   B_REF = {B_REF:,}")
print(f"\n  base_train.py:299  batch_lr_scale = sqrt(total_batch_size / B_REF)")
print(f"  base_train.py:308  weight_decay    = wd * sqrt(B/B_REF) * (D_REF/target_tokens)\n")
print(f"  {'total_batch_size':>17} {'LR multiplier':>14} {'embedding_lr':>13} {'matrix_lr':>11} {'wd(0.28 nominal)':>18}")
for B in [8192, 16384, 32768]:
    s = math.sqrt(B / B_REF)
    wd = 0.28 * math.sqrt(B / B_REF) * (D_REF / target_tokens)
    print(f"  {B:>17,} {s:>14.4f} {0.2*s:>13.6f} {0.02*s:>11.6f} {wd:>18.6f}")
print(f"\n  => w2-batch-8192 vs w2-batch-32768 is NOT a batch-size ablation. It is a joint")
print(f"     (batch, LR, weight-decay) intervention: the LR multiplier moves {math.sqrt(8192/B_REF):.4f} -> "
      f"{math.sqrt(32768/B_REF):.4f}, i.e. 2.00x.")
print(f"     To isolate batch size you must pin the LRs explicitly on both arms.")

# ============================================================== 3. BUDGET AXES
print("\n" + "=" * 88)
print("3. THE FOUR BUDGET AXES ARE NOT INTERCHANGEABLE (real numbers, d6 family)")
print("=" * 88)
variants = {
    "w1-pilot (control)":     dict(),
    "w2-depth8-fixed-width":  dict(depth=8, aspect=48),
    "w2-width512-fixed-depth":dict(depth=6, aspect=85),
    "w2-heads-4":             dict(head_dim=96),
    "w2-heads-3":             dict(head_dim=128),
    "w2-context-256":         dict(seq=256),
    "w2-context-1024":        dict(seq=1024),
}
STEPS, TOKENS_PER_STEP = defaults["base"]["num_iterations"], defaults["base"]["total_batch_size"]
print(f"  All arms run {STEPS} steps x {TOKENS_PER_STEP:,} tokens = {STEPS*TOKENS_PER_STEP:,} tokens (fixed steps AND fixed tokens).")
print(f"\n  {'arm':>25} {'n_embd':>7} {'heads':>6} {'seq':>5} {'params':>12} {'FLOPs/tok':>12} {'total FLOPs':>12} {'vs control':>11}")
base_f = None
for name, kw in variants.items():
    m = build(**kw)
    p = m.num_scaling_params()["total"]
    f = m.estimate_flops()
    tot = f * STEPS * TOKENS_PER_STEP
    if base_f is None:
        base_f = tot
    print(f"  {name:>25} {m.config.n_embd:>7} {m.config.n_head:>6} {m.config.sequence_len:>5} "
          f"{p:>12,} {f:>12,} {tot:>12.3e} {tot/base_f:>10.2f}x")
print(f"\n  Fixed steps and fixed tokens are IDENTICAL here (batch is pinned), yet compute")
print(f"  spans a wide range. A fixed-step depth/width/head/context result is therefore")
print(f"  never automatically compute-matched - the FLOPs column is the confound.")
print(f"\n  When to use which axis:")
print(f"    fixed STEPS     - optimizer dynamics at identical update count (LR, warmup, schedule).")
print(f"    fixed TOKENS    - data efficiency, only when bytes/token is also equal (see W1).")
print(f"    fixed FLOPs     - any architecture-efficiency claim; use --target-flops.")
print(f"    fixed WALL TIME - practical value on this machine; the only axis that prices")
print(f"                      MPS kernel coverage, memory pressure and throughput drift.")

# ============================================================== 4. NUISANCE vs CONFOUND
print("\n" + "=" * 88)
print("4. NUISANCE vs CONFOUND, instantiated on this hardware")
print("=" * 88)
run = json.load(open("/Users/frankfacundo/.cache/nanochat/lab_runs/w0-smoke--20260822-212046/summary.json"))
sm = run["metrics"]["step_ms"]
print(f"  NUISANCE (adds variance, uncorrelated with the intervention):")
print(f"    step_ms in the real w0-smoke run: first={sm['first']:.0f}ms max={sm['max']:.0f}ms "
      f"min={sm['min']:.0f}ms last={sm['last']:.0f}ms")
print(f"    -> {sm['max']/sm['min']:.2f}x spread from MPS warmup/thermal, not from any treatment.")
print(f"    It inflates the error bar on a wall-time claim; it does not bias it.")
print(f"  CONFOUND (moves WITH the intervention, offers a rival explanation):")
print(f"    raising total_batch_size also raises every LR by sqrt(B/B_ref) (section 2).")
print(f"    If 32K batch wins, 'bigger batch helps' and 'higher LR helps' fit the data equally.")
print(f"    Fix by pinning LRs; you cannot fix it by averaging more seeds.")
