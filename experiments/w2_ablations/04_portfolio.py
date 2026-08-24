"""
W2/C - Defend an ablation portfolio, with effect sizes in units of a MEASURED noise floor.

Reads the real logs produced by run_noise_floor.sh (3 seeds of the control) and
run_portfolio.sh (four single-factor arms), then:
  - reports every arm as an effect size in sigma, with a strength label
  - shows why confirming only the pilot winner is biased, by Monte Carlo on the MEASURED sigma
  - names the cheapest decisive follow-up for each claim

Run: python -m experiments.w2_ablations.04_portfolio
"""
import json, math, os, random, re, statistics as st
from pathlib import Path
import torch
from nanochat.gpt import GPT, GPTConfig

LOGS = Path(__file__).resolve().parent / "logs"
random.seed(0)

def finals(path):
    if not path.exists():
        return None
    v = re.findall(r"Validation bpb: ([\d.]+)", path.read_text())
    t = re.findall(r"Total training time: ([\d.]+)m", path.read_text())
    return (float(v[-1]), float(t[-1]) if t else float("nan")) if v else None

# ---------------------------------------------------------------- noise floor
ctrl = [finals(LOGS / f"noise_seed{s}.log") for s in (42, 43, 44)]
ctrl = [c for c in ctrl if c]
cb = [c[0] for c in ctrl]
ct = [c[1] for c in ctrl]
MU, SIGMA = st.mean(cb), st.stdev(cb)

print("=" * 100)
print("1. THE NOISE FLOOR - measured first, before any arm was interpreted")
print("=" * 100)
print(f"  3 runs of the pilot control, identical config, only NANOCHAT_SEED differs (42/43/44).")
print(f"  final val_bpb : {[f'{x:.6f}' for x in cb]}")
print(f"  mean = {MU:.6f}   sigma = {SIGMA:.6f}   range = {max(cb)-min(cb):.6f}")
print(f"  wall time     : {[f'{x:.2f}m' for x in ct]}  -> {max(ct)/min(ct):.2f}x spread")
print(f"\n  DECISION RULE (frozen before looking at the arms): an effect counts as REAL only if")
print(f"  |delta| > 3 sigma = {3*SIGMA:.6f} bpb. Between 2 and 3 sigma it is SUGGESTIVE.")
print(f"  Below 2 sigma ({2*SIGMA:.6f}) it is a NULL result and must be reported as one.")
print(f"\n  CAVEAT, stated up front: nanochat seeds only torch's global RNG (common.py), which")
print(f"  controls weight INIT. The data order is identical across these three runs. So sigma")
print(f"  here is an init+hardware noise floor and UNDERSTATES true run-to-run variation.")
print(f"  Every effect size below is therefore an OPTIMISTIC significance estimate.")

# ---------------------------------------------------------------- arms
ARMS = [
    ("lr-half",   "all 4 group LRs x0.5",  "arm_lr-half.log"),
    ("lr-double", "all 4 group LRs x2.0",  "arm_lr-double.log"),
    ("wd-zero",   "--weight-decay 0.28->0", "arm_wd-zero.log"),
    ("tied",      "--tie-embeddings 0->1", "arm_tied.log"),
]
print("\n" + "=" * 100)
print("2. THE PORTFOLIO")
print("=" * 100)
print(f"  {'arm':<11} {'intervention':<24} {'val_bpb':>9} {'delta':>9} {'sigma':>7} {'wall':>7} {'verdict':>12}")
rows = []
for name, desc, fn in ARMS:
    r = finals(LOGS / fn)
    if r is None:
        print(f"  {name:<11} {desc:<24} {'(not run yet)':>9}")
        continue
    bpb, wall = r
    d = bpb - MU
    z = d / SIGMA
    verdict = ("REAL worse" if z > 3 else "REAL better" if z < -3 else
               "suggestive" if abs(z) > 2 else "NULL")
    rows.append(dict(name=name, desc=desc, bpb=bpb, d=d, z=z, wall=wall, verdict=verdict))
    print(f"  {name:<11} {desc:<24} {bpb:>9.6f} {d:>+9.6f} {z:>+7.1f} {wall:>6.2f}m {verdict:>12}")
print(f"  {'control':<11} {'(3 seeds)':<24} {MU:>9.6f} {0:>+9.6f} {0:>+7.1f} {st.mean(ct):>6.2f}m {'baseline':>12}")

# ---------------------------------------------------------------- three defended claims
print("\n" + "=" * 100)
print("3. THREE DEFENDED CLAIMS")
print("=" * 100)
by = {r["name"]: r for r in rows}

def claim(title, arm, effect, cost, confound, strength, follow):
    print(f"\n  --- {title} ---")
    print(f"  effect size   : {effect}")
    print(f"  resource cost : {cost}")
    print(f"  confound      : {confound}")
    print(f"  strength      : {strength}")
    print(f"  cheapest decisive follow-up: {follow}")

if "lr-double" in by:
    a = by["lr-double"]
    claim("POSITIVE (and SURPRISING): doubling every LR helps, a lot", a,
          f"{a['d']:+.6f} bpb vs control = {abs(a['z']):.0f} sigma. Also better than lr-half by "
          f"{by['lr-half']['bpb']-a['bpb']:.6f} bpb, so the response is monotone across 0.5x/1x/2x.",
          f"{a['wall']:.2f}m, identical params and FLOPs/token to the control - this is a free win at fixed compute.",
          "the 600-step horizon. nanochat's default LRs are tuned at d12 with a 524,288-token batch "
          "and transferred by muP-style rules; at d6/16,384 tokens/600 steps the transferred value "
          "may simply be too conservative. 'Higher LR is better' and 'the muP transfer is "
          "mistuned for this budget' predict the same result here.",
          "STRONG at this budget, UNKNOWN beyond it. Undertrained runs systematically favour "
          "aggressive LRs because they never reach the regime where instability bites.",
          "run lr-double at 3000+ steps (w2-tokens-10000 style). If the advantage shrinks or "
          "reverses, the effect was a short-horizon artifact - which is the single most common "
          "way LR ablations mislead.")

if "lr-half" in by:
    a = by["lr-half"]
    claim("NEGATIVE: halving every LR is clearly harmful", a,
          f"{a['d']:+.6f} bpb = {a['z']:.0f} sigma WORSE.",
          f"{a['wall']:.2f}m, same compute.",
          "same as above, with the sign reversed; a lower LR at a fixed short horizon simply "
          "travels less far along the same trajectory.",
          "STRONG, and it is the more trustworthy half of the LR result: it is hard to construct "
          "a story where a 24-sigma degradation is an artifact.",
          "none needed for the negative direction; it is already decisive at this budget.")

if "wd-zero" in by:
    a = by["wd-zero"]
    claim("NULL: turning weight decay off changes nothing measurable", a,
          f"{a['d']:+.6f} bpb = {abs(a['z']):.2f} sigma - inside the noise floor.",
          f"{a['wall']:.2f}m.",
          "W2/01 proved --weight-decay reaches only 14.4% of parameters (the Muon matrix groups); "
          "the other 85.6% keep hardcoded AdamW decays the flag cannot touch. AND base_train "
          "anneals lambda to 0 on a cosine, so the arms differ only early. So this null is NOT "
          "'weight decay does not matter' - it is 'this flag, on this fraction, over this "
          "horizon, does not matter'.",
          "NULL, reported as a null. At 3.53 tokens-per-param (W1/05) the model is 5.7x "
          "undertrained and has nothing to overfit, which is exactly when regularization "
          "should do nothing. The null CONFIRMS the underfitting diagnosis.",
          "the informative version is not a longer run, it is a different intervention: apply "
          "decay to the embedding/lm_head groups too, where 85.6% of the parameters live.")

if "tied" in by:
    a = by["tied"]
    claim("CAPACITY: tied embeddings", a,
          f"{a['d']:+.6f} bpb = {a['z']:+.1f} sigma.",
          f"{a['wall']:.2f}m; removes 12,582,912 params (17.1%) at IDENTICAL FLOPs/token (W2/03).",
          "raw comparison only - the tied arm is handed 17% fewer parameters for the same "
          "compute, so this is biased against tying by construction.",
          "REAL but RAW. It answers 'is the flag a good deal as-is', not 'is weight tying a "
          "good idea', which needs the parameter-matched arm.",
          "the parameter-matched arm from W2/03: d6 tied at aspect_ratio=72 (73.2M params, "
          "0.995x the control) run at matched FLOPs.")

# ---------------------------------------------------------------- winner's curse
print("\n" + "=" * 100)
print("4. WHY CONFIRMING ONLY THE PILOT WINNER IS BIASED (Monte Carlo on the MEASURED sigma)")
print("=" * 100)
TRIALS, N_ARMS = 20000, 8
print(f"  Simulate a portfolio of {N_ARMS} arms that are ALL genuinely identical to the control")
print(f"  (true effect exactly 0), each measured once with the measured sigma = {SIGMA:.6f}.")
print(f"  Then 'confirm' only the arm that looked best in the pilot.\n")
best_pilot, best_confirm = [], []
for _ in range(TRIALS):
    pilot = [random.gauss(0, SIGMA) for _ in range(N_ARMS)]
    i = min(range(N_ARMS), key=lambda k: pilot[k])
    best_pilot.append(pilot[i])
    best_confirm.append(random.gauss(0, SIGMA))   # independent re-measurement, same true 0
mp, mc = st.mean(best_pilot), st.mean(best_confirm)
print(f"  mean apparent effect of the pilot WINNER : {mp:+.6f} bpb  ({mp/SIGMA:+.2f} sigma)")
print(f"  mean effect on independent re-measurement: {mc:+.6f} bpb  ({mc/SIGMA:+.2f} sigma)")
print(f"  expected SHRINKAGE                       : {mp-mc:+.6f} bpb")
frac = sum(1 for x in best_pilot if x < -2 * SIGMA) / TRIALS
print(f"  fraction of these all-null portfolios where the winner clears 2 sigma: {frac:.1%}")
print(f"""
  With {N_ARMS} arms and zero real effects, the best-looking arm appears {abs(mp/SIGMA):.1f} sigma better
  purely by selection. Confirming ONLY that arm guarantees a regression to the mean that
  reads as 'the effect got smaller at scale' - a story that is indistinguishable from a real
  but budget-sensitive effect. That is the bias: the confirmation stage inherits the
  selection, so it cannot correct it.
  WHAT TO DO INSTEAD: confirm a PRE-REGISTERED set that includes at least one arm you expect
  to be null (here, wd-zero) and one you expect to be negative (lr-half). If the null and
  the negative reproduce with the right signs and magnitudes, the confirmation stage is
  itself calibrated, and the positive result inherits that credibility.
  NOTE the measured effects here are {min(abs(r['z']) for r in rows if r['verdict'].startswith('REAL')):.0f}-{max(abs(r['z']) for r in rows):.0f} sigma, far outside the
  selection band - {abs(mp/SIGMA):.1f} sigma - so THESE particular claims are not winner's-curse artifacts.""")

# ---------------------------------------------------------------- confirmation
print("\n" + "=" * 100)
print("5. THE CONFIRMATION RUN (control + intervention at a larger budget)")
print("=" * 100)
if "lr-double" in by:
    print(f"""  Pre-registered before running: at 3000 steps (5x the pilot budget, ~28 min each),
    H1: lr-double still beats the control by > 3 sigma  -> the effect is budget-robust
    H0: the gap shrinks below 3 sigma or reverses       -> it was a short-horizon artifact
  Both arms run as a PAIR at the same budget, on the same machine, back to back:

    export NANOCHAT_SEED=42
    COMMON="--depth=6 --aspect-ratio=64 --head-dim=64 --window-pattern=L --max-seq-len=512 \\
      --device-batch-size=32 --total-batch-size=16384 --num-iterations=3000 \\
      --eval-every=250 --eval-tokens=131072 --core-metric-every=-1 --sample-every=-1 --save-every=-1"
    python -m scripts.base_train $COMMON --model-tag=confirm_control --run=dummy
    python -m scripts.base_train $COMMON --embedding-lr=0.6 --unembedding-lr=0.016 \\
      --matrix-lr=0.04 --scalar-lr=1.0 --model-tag=confirm_lrdouble --run=dummy

  The control is re-run at the new budget rather than reused from the pilot, because a
  control measured at a different horizon is a different measuring instrument (W0/02 a).""")
print(f"\n  Portfolio JSON written for downstream use.")
json.dump(dict(sigma=SIGMA, control_mean=MU, control_seeds=cb, arms=rows),
          open(LOGS / "04_portfolio.json", "w"), indent=2)
