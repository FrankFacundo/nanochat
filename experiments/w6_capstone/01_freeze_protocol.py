"""
W6/A - Freeze the Parameter Golf protocol BEFORE any search result is observed.

Produces protocol.json: a hashed, machine-checkable record of the timer, hardware,
data, tokenizer, sealed test split, and baselines. Re-running it VERIFIES the freeze
instead of silently re-freezing (any drift is reported as a violation).

Run: python -m experiments.w6_capstone.01_freeze_protocol
"""
import hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
import torch

OUT = Path(__file__).resolve().parent
BASE = Path(os.environ.get("NANOCHAT_BASE_DIR", Path.home() / ".cache/nanochat"))
DATA = BASE / "base_data_climbmix"
PROTOCOL = OUT / "protocol.json"

# The last shard is nanochat's val split (dataset.py:75). Seal the TWO before it as test:
# disjoint from both train and val, and enforceable via NANOCHAT_SEALED_SHARDS.
shards = sorted(p.name for p in DATA.glob("shard_*.parquet"))
VAL_SHARD = shards[-1]
SEALED = shards[-3:-1]

def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            if limit and f.tell() > limit:
                break
    return h.hexdigest()

def git(*a):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unavailable"

print("=" * 96)
print("FREEZING THE CAPSTONE PROTOCOL")
print("=" * 96)
protocol = {
    "schema": 1,
    "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "objective": {
        "metric": "validation bits-per-byte (val_bpb) at the final step",
        "direction": "lower is better",
        "budget": "30 minutes wall clock of scripts.base_train on the machine below",
        "tiebreak": "if two candidates are within the measured noise floor, the one with "
                    "fewer total parameters wins; if still tied, the earlier commit wins",
    },
    "timer": {
        "starts": "the moment scripts.base_train is invoked by entrypoint.sh",
        "stops": "when the process exits",
        "includes": ["model construction", "data loading", "training", "all in-run evals"],
        "excludes": ["tokenizer training", "dataset download", "the one-time sealed-test eval"],
        "rule": "num_iterations must be chosen in advance. A run that overruns 30:00 is VOID, "
                "not truncated - truncating after seeing the clock is selection on the outcome.",
    },
    "hardware": {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "rule": "all candidate runs and the final run must be on THIS machine, on AC power, "
                "with no other training process running (throughput varied 3.6x in W0/01).",
    },
    "data": {
        "dir": str(DATA),
        "n_shards_total": len(shards),
        "val_shard": VAL_SHARD,
        "sealed_test_shards": SEALED,
        "enforcement": "export NANOCHAT_SEALED_SHARDS='" + ",".join(SEALED) + "' for EVERY "
                       "search run; dataset.list_parquet_files then removes them from train AND val.",
        "sealed_sha256": {s: sha256(DATA / s, limit=1 << 24) for s in SEALED},
        "val_sha256": sha256(DATA / VAL_SHARD, limit=1 << 24),
    },
    "tokenizer": {
        "path": str(BASE / "tokenizer" / "tokenizer.pkl"),
        "sha256": sha256(BASE / "tokenizer" / "tokenizer.pkl"),
        "token_bytes_sha256": sha256(BASE / "tokenizer" / "token_bytes.pt"),
        "rule": "frozen. Re-running tok_train invalidates every prior number (W0/02 audit b).",
    },
    "code": {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "rule": "the final run must be made from a CLEAN tree at a tagged commit. A dirty tree "
                "means repo_commit does not identify the code that ran (W0/02 section 5).",
    },
    "checkpoints": {
        "limit": "one checkpoint per candidate, final step only (--save-every=-1)",
        "rule": "no mid-run checkpoint may be selected on val_bpb. The endpoint is the endpoint; "
                "min-over-N is a biased estimator (W0/02 section 4c).",
    },
    "precomputation": {
        "allowed": ["tokenizer training", "dataset download", "any analysis that touches no "
                    "sealed shard", "meta-device parameter/FLOP accounting"],
        "forbidden": ["warm-starting from a checkpoint trained outside the 30 minutes",
                      "caching tokenized batches that took >30 min to build",
                      "any read of the sealed shards before the candidate is locked"],
    },
    "retries": {
        "rule": "a run may be repeated ONLY for a documented infrastructure failure (OOM, crash, "
                "thermal throttle detected as >2x median step time). The failure and the retry "
                "are both recorded. A run may never be repeated because the result was bad.",
    },
    "noise_floor": {
        "source": "experiments/w2_ablations/run_noise_floor.sh, 3 seeds of the pilot control",
        "sigma_bpb": None,   # filled below from the real logs
        "rule": "a candidate beats a baseline only if the gap exceeds 3 sigma.",
    },
}

# pull the measured noise floor in from the real seed runs
import re, statistics as st
finals = []
for s in (42, 43, 44):
    f = OUT.parent / "w2_ablations" / "logs" / f"noise_seed{s}.log"
    if f.exists():
        v = re.findall(r"Validation bpb: ([\d.]+)", f.read_text())
        if v:
            finals.append(float(v[-1]))
if len(finals) >= 2:
    protocol["noise_floor"]["sigma_bpb"] = round(st.stdev(finals), 6)
    protocol["noise_floor"]["seed_finals"] = finals
    protocol["noise_floor"]["decision_threshold_bpb"] = round(3 * st.stdev(finals), 6)

protocol["baselines"] = {
    "official_d6": {
        "what": "runs/runcpu.sh d6 config, 600 steps (the pilot control)",
        "val_bpb": round(st.mean(finals), 6) if finals else None,
        "n_seeds": len(finals),
        "cmd": "scripts.base_train --depth=6 --head-dim=64 --window-pattern=L --max-seq-len=512 "
               "--device-batch-size=32 --total-batch-size=16384 --num-iterations=600",
    },
    "best_single_intervention": {
        "what": "the best single flag change found in the W2 portfolio, run alone",
        "source": "experiments/w2_ablations/logs/arm_*.log",
        "rule": "must be filled from the portfolio BEFORE the staged search begins, so the "
                "search has something to beat that is not just the default.",
    },
    "simple_alternative": {
        "what": "a d6 model trained for 30 minutes with NO changes except num_iterations set "
                "to whatever fits the budget",
        "why": "the honest null: if the tuned candidate cannot beat 'just train longer', the "
               "search found nothing. This is the baseline most searches quietly omit.",
    },
}

protocol["search"] = {
    "stage_1_screen": "one run per single-factor candidate at 600 steps. Keep any candidate "
                      "within 1 sigma of the best; discard the rest. No interactions yet.",
    "stage_2_interactions": "pair the surviving factors (full factorial if <=3 survive, else "
                            "a fold-over design). Interactions are the only reason stage 1 "
                            "cannot be trusted alone: LR and batch size are already coupled "
                            "by base_train (W2/01), so factors are not independent by default.",
    "stage_3_budget": "re-run the top 2 configurations at the full 30-minute budget. Rankings "
                      "at 600 steps do not transfer to 30 minutes - W1/05 showed the model is "
                      "5.7x undertrained, so short-horizon winners can be long-horizon losers.",
    "candidate_lock": "after stage 3, write the winning config into entrypoint.sh, commit, and "
                      "tag. No further edits. The lock is what makes the test valid.",
    "final_test": "ONE evaluation of the locked candidate on the sealed shards, run once, "
                  "reported whatever it says. Opening it twice converts it into a second "
                  "validation split and it must be relabelled as such.",
}

if PROTOCOL.exists():
    old = json.loads(PROTOCOL.read_text())
    drift = [k for k in ("data", "tokenizer", "objective", "timer")
             if json.dumps(old.get(k), sort_keys=True) != json.dumps(protocol[k], sort_keys=True)]
    print(f"  protocol.json already exists (frozen {old.get('frozen_at')})")
    if drift:
        print(f"  *** FREEZE VIOLATION *** these frozen sections changed: {drift}")
        print(f"      Every number measured under the old protocol is invalid. Investigate before rerunning.")
    else:
        print(f"  VERIFIED: data, tokenizer, objective and timer all match the frozen record.")
    print(f"  (not overwriting; delete the file deliberately if you really mean to re-freeze)")
else:
    PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"  wrote {PROTOCOL}")

print(f"\n  objective        : {protocol['objective']['metric']}, {protocol['objective']['budget']}")
print(f"  device           : {protocol['hardware']['device']} on {protocol['hardware']['platform']}")
print(f"  shards           : {len(shards)} total | val={VAL_SHARD} | SEALED={SEALED}")
print(f"  tokenizer sha256 : {protocol['tokenizer']['sha256'][:32]}...")
print(f"  code commit      : {protocol['code']['commit'][:12]} (dirty={protocol['code']['dirty']})")
nf = protocol["noise_floor"]
if nf["sigma_bpb"]:
    print(f"  noise floor      : sigma = {nf['sigma_bpb']:.6f} bpb over {len(finals)} seeds "
          f"{[round(x,6) for x in finals]}")
    print(f"  decision rule    : a candidate must beat a baseline by > {nf['decision_threshold_bpb']:.6f} bpb (3 sigma)")
else:
    print(f"  noise floor      : NOT MEASURED - run experiments/w2_ablations/run_noise_floor.sh first")
print(f"\n  Enforce the seal in every search run:")
print(f"    export NANOCHAT_SEALED_SHARDS='{','.join(SEALED)}'")
