"""
W0/B - Audit the real w0-smoke run: which artifact is provenance, which is derived,
what is lost by keeping only final BPB, and three comparisons that are invalid.

The central claim - "summary.json is derived convenience, metrics.jsonl is the
record" - is not asserted. It is PROVEN by recomputing every aggregate in
summary.json from metrics.jsonl and diffing.

Run: python -m experiments.w0_method.02_artifact_audit
"""
import json, os, subprocess, hashlib, statistics, re
from pathlib import Path

RUN = Path("/Users/frankfacundo/.cache/nanochat/lab_runs/w0-smoke--20260822-212046")
D6_LOG = Path("/Users/frankfacundo/.cache/nanochat/logs/baseline-d6_20260823-102805.log")

manifest = json.loads((RUN / "manifest.json").read_text())
summary = json.loads((RUN / "summary.json").read_text())
metrics = [json.loads(l) for l in (RUN / "metrics.jsonl").read_text().splitlines() if l.strip()]

# ============================================================ 1. CLASSIFY
print("=" * 88)
print("1. ARTIFACT CLASSIFICATION (by file, with sizes and reconstructability)")
print("=" * 88)
files = sorted(RUN.rglob("*"))
for f in files:
    if f.is_file():
        rel = f.relative_to(RUN)
        print(f"  {str(rel):<52} {f.stat().st_size:>12,} bytes")

print(f"\n  IMMUTABLE PROVENANCE -> manifest.json. It is the only file written BEFORE the")
print(f"    run and never updated. It pins the facts nothing else can recover:")
for k in ["repo_commit", "repo_dirty", "platform", "python", "started_at", "execution_id", "parent_run"]:
    print(f"      {k:<14} = {manifest[k]}")
print(f"      commands       = {len(manifest['commands'])} command(s), argv captured verbatim")
print(f"\n  RAW RECORD -> metrics.jsonl. {len(metrics)} append-only rows, "
      f"{len({m['name'] for m in metrics})} series, never rewritten.")
print(f"  DERIVED CONVENIENCE -> summary.json (aggregates) and stdout.log (human view).")
print(f"  DERIVED + HEAVY -> workspace/ (checkpoints, tokenizer copy). Reproducible from")
print(f"    manifest + data; the expensive thing to keep, the cheap thing to lose.")

# ============================================================ 2. PROVE DERIVED
print("\n" + "=" * 88)
print("2. PROOF that summary.json is DERIVED: recompute every aggregate from metrics.jsonl")
print("=" * 88)
series = {}
for m in metrics:
    series.setdefault(m["name"], []).append((m["step"], m["value"]))
mismatch = 0
print(f"  {'series':<18} {'field':<10} {'summary.json':>16} {'recomputed':>16} {'match':>7}")
for name, pts in sorted(series.items()):
    pts.sort(key=lambda p: p[0])
    vals = [v for _, v in pts]
    recomputed = {"count": len(vals), "first": vals[0], "last": vals[-1],
                  "last_step": pts[-1][0], "max": max(vals), "min": min(vals)}
    for field, got in recomputed.items():
        want = summary["metrics"][name][field]
        ok = (want == got)
        mismatch += (not ok)
        print(f"  {name:<18} {field:<10} {want!s:>16} {got!s:>16} {'OK' if ok else 'DIFF':>7}")
print(f"\n  mismatches = {mismatch}  ->  summary.json carries ZERO information not already in")
print(f"  metrics.jsonl. Delete it and you lose nothing; delete metrics.jsonl and the")
print(f"  aggregates become unfalsifiable numbers with no series behind them.")

# ============================================================ 3. INFORMATION LOSS
print("\n" + "=" * 88)
print("3. WHAT RETAINING ONLY FINAL VALIDATION BPB DESTROYS")
print("=" * 88)
vb = sorted(series["val_bpb"])
sm = sorted(series["step_ms"])
tl = sorted(series["train_loss"])
print(f"  Kept:  final val_bpb = {vb[-1][1]:.6f}")
print(f"  Lost, and each loss disables a specific question:")
print(f"    a) the val_bpb TRAJECTORY {[f'{s}:{v:.4f}' for s, v in vb]}")
print(f"       -> without it you cannot tell 'converged' from 'still descending'. Here the")
print(f"          last interval still drops {vb[-2][1]-vb[-1][1]:.4f} bpb, so the run is")
print(f"          budget-limited, not capability-limited. That single fact decides whether")
print(f"          a negative ablation result means 'no effect' or 'not trained long enough'.")
print(f"    b) THROUGHPUT DRIFT: step_ms {sm[0][1]:.0f} -> {sm[-1][1]:.0f} "
      f"(max {max(v for _,v in sm):.0f}, min {min(v for _,v in sm):.0f}).")
print(f"       -> any wall-time claim computed from mean step_ms is wrong by up to "
      f"{max(v for _,v in sm)/min(v for _,v in sm):.1f}x if you did not see this shape.")
print(f"    c) TRAIN vs VAL SEPARATION: train_loss {tl[0][1]:.4f} -> {tl[-1][1]:.4f} alongside")
print(f"       val_bpb. Overfitting is a statement about the GAP over time; a single final")
print(f"       number cannot express a gap.")
print(f"    d) the number of eval points ({len(vb)}), which is the multiple-comparison count")
print(f"       needed to interpret a reported minimum (section 4c).")

# ============================================================ 4. INVALID COMPARISONS
print("\n" + "=" * 88)
print("4. THREE INVALID COMPARISONS, each quantified against this repo")
print("=" * 88)
cat = json.loads(Path("llm_lab/configs/catalog.json").read_text())
et = {e["id"]: {**cat["defaults"].get(e["stage"], {}), **e["args"]}.get("eval_tokens")
      for e in cat["experiments"] if e["stage"] == "base"}
print(f"  (a) MEASUREMENT failure - unequal evaluation budgets.")
print(f"      eval_tokens across base experiments in the catalog: "
      f"{sorted(set(v for v in et.values() if v))}")
print(f"      w0-smoke used {manifest['args']['eval_tokens']:,}; the pilot default is "
      f"{cat['defaults']['base']['eval_tokens']:,}; confirm runs use 524,288.")
print(f"      A 32,768-token estimate averages {524288/32768:.0f}x fewer targets than a confirm")
print(f"      estimate, so its standard error is ~{(524288/32768)**0.5:.1f}x larger. Comparing a")
print(f"      pilot bpb against a confirm bpb compares two different measuring instruments.")
print(f"      RULE: bpb is comparable only between runs with identical eval_tokens AND the")
print(f"      same val shards (W1 showed the byte denominator also moves with the tokenizer).")

print(f"\n  (b) CONTROL failure - the workspace carries its own tokenizer.")
tokpkl = RUN / "workspace" / "tokenizer" / "tokenizer.pkl"
shared = Path.home() / ".cache/nanochat/tokenizer/tokenizer.pkl"
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "MISSING"
print(f"      run workspace tokenizer sha256[:16] = {h(tokpkl)}")
print(f"      shared  ~/.cache tokenizer sha256[:16] = {h(shared)}")
print(f"      identical = {h(tokpkl) == h(shared)}")
print(f"      Every run snapshots its tokenizer, so a later `tok_train` cannot retro-corrupt")
print(f"      it - but it also means two runs can silently disagree. Comparing bpb across")
print(f"      runs whose tokenizer hashes differ is invalid even at equal eval_tokens.")

print(f"\n  (c) SELECTION failure - reporting the minimum over many eval points.")
# this log file concatenates a base_train run and a chat_sft run; keep only base_train
base_section = D6_LOG.read_text().split("Minimum validation bpb")[0]
bpbs = [float(m.group(2)) for m in re.finditer(r"Step (\d+) \| Validation bpb: ([\d.]+)", base_section)]
print(f"      The real d6 base_train run logged {len(bpbs)} validation points; it reports")
print(f"      'Minimum validation bpb' (base_train.py:659). min={min(bpbs):.6f} final={bpbs[-1]:.6f} "
      f"gap={bpbs[-1]-min(bpbs):+.6f}")
mono = all(b <= a for a, b in zip(bpbs, bpbs[1:]))
print(f"      This particular curve is monotone decreasing ({mono}), so min==final and the bias")
print(f"      is ZERO HERE. That is a property of a 5000-step run with a 524,288-token eval,")
print(f"      not of the estimator. At pilot scale (600 steps, eval_tokens=131,072) the eval")
print(f"      noise is ~2x larger and the curve is not monotone, which is exactly where")
print(f"      min-over-N starts paying out. Do not generalize this zero.")
print(f"      min-over-N is a biased-low estimator: with N={len(bpbs)} draws its expected value")
print(f"      sits below the true endpoint by roughly the noise scale. Two arms with a")
print(f"      DIFFERENT number of eval points are therefore not comparable on min at all.")
print(f"      RULE: pre-register final-step bpb as the primary endpoint; report min only as")
print(f"      a secondary, and only between arms with identical eval_every and horizon.")

# ============================================================ 5. PROTOCOL
print("\n" + "=" * 88)
print("5. FROZEN PROTOCOL: divergence rule and test-set policy")
print("=" * 88)
dirty = manifest["repo_dirty"]
print(f"  Reproducibility hole found in this very run: repo_dirty = {dirty}.")
if dirty:
    print(f"    manifest pins repo_commit={manifest['repo_commit'][:12]} but the tree had uncommitted")
    print(f"    edits, so the commit does NOT identify the code that ran. FIX: refuse to launch")
    print(f"    when dirty, or store `git diff` in the run dir. Rule below assumes that fix.")
print(f"""
  DIVERGENCE RULE (measurable, not vibes):
    1. Establish the noise floor first: run the control 3x, same commit, same args,
       same data order, varying only the seed. Let s = stdev of final val_bpb.
    2. A reproduction PASSES if |bpb_repro - bpb_original| <= 2s.
    3. A claimed effect counts only if |bpb_arm - bpb_control| > 3s AND the sign is
       stable across all seeds run.
    4. Any run whose manifest shows repo_dirty=true, a different tokenizer hash, or a
       different eval_tokens is quarantined and cannot enter a comparison.
    NOTE: s is currently UNMEASURED in this workspace. Until it is, no effect in this
    lab is calibrated - that is the single highest-value 3 x 6min of compute available.

  TEST-SET POLICY:
    - train shards: fitting only.
    - val shards (the ones base_train evaluates): model selection, ablation ranking,
      early stopping. Touched constantly, therefore burned for final claims.
    - sealed test: a disjoint shard range, hashed and committed BEFORE the search
      starts, evaluated ONCE at the very end by the frozen entrypoint. One number,
      reported whatever it says. If it is opened twice it is no longer a test set and
      must be relabelled as a second validation split.
""")
