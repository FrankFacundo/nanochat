"""
W6 - The one-time sealed-test evaluation. Called only by entrypoint.sh --final.

Evaluates BPB on the sealed shards named in protocol.json. Those shards were excluded
from train and val for every search run, so this is the first and only time the model
sees them.

Run: python -m experiments.w6_capstone.03_sealed_test --model-tag=capstone
"""
import argparse, hashlib, json, os
from pathlib import Path
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_token_bytes
from nanochat.loss_eval import evaluate_bpb

ap = argparse.ArgumentParser()
ap.add_argument("--model-tag", default="capstone")
ap.add_argument("--tokens", type=int, default=262144)
args = ap.parse_args()

OUT = Path(__file__).resolve().parent
protocol = json.loads((OUT / "protocol.json").read_text())
DATA = Path(protocol["data"]["dir"])
SEALED = protocol["data"]["sealed_test_shards"]

# verify the sealed shards are byte-identical to what was frozen
def sha256(p, limit=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            if f.tell() > limit:
                break
    return h.hexdigest()

print("=" * 88)
print("SEALED TEST - ONE EVALUATION, REPORTED WHATEVER IT SAYS")
print("=" * 88)
for s in SEALED:
    got, want = sha256(DATA / s), protocol["data"]["sealed_sha256"][s]
    print(f"  {s}: sha256 {'MATCHES' if got == want else '*** MISMATCH ***'} the frozen record")
    assert got == want, f"{s} changed since the freeze; the test set is compromised"

ddp, rank, lrank, world, device = compute_init(autodetect_device_type())
model, tok, meta = load_model("base", device, phase="eval", model_tag=args.model_tag)
token_bytes = get_token_bytes(device=device)
SEQ = meta["model_config"]["sequence_len"]
B = 4
steps = max(1, args.tokens // (B * SEQ))

# point the loader at ONLY the sealed shards: seal everything else
import nanochat.dataset as ds
all_shards = sorted(p.name for p in DATA.glob("shard_*.parquet"))
os.environ["NANOCHAT_SEALED_SHARDS"] = ",".join(s for s in all_shards if s not in SEALED)
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
loader = tokenizing_distributed_data_loader_bos_bestfit(tok, B, SEQ, "val", device=device)
bpb = evaluate_bpb(model, loader, steps, token_bytes)

print(f"\n  model      : {args.model_tag} step {meta['step']}")
print(f"  shards     : {SEALED}")
print(f"  budget     : {B}x{SEQ}x{steps} = {B*SEQ*steps:,} target tokens")
print(f"  SEALED TEST BPB = {bpb:.6f}")
sigma = protocol["noise_floor"].get("sigma_bpb")
base = protocol["baselines"]["official_d6"].get("val_bpb")
if sigma and base:
    print(f"\n  official d6 baseline (val)  = {base:.6f}")
    print(f"  noise floor sigma           = {sigma:.6f}")
    print(f"  NOTE the baseline number is a VAL number on different shards. The honest")
    print(f"  comparison requires evaluating the baseline on the SEALED shards too - do that")
    print(f"  in the same --final invocation, or the 'gap' mixes two different test sets.")
print(f"\n  This number is now spent. Any further tuning against it makes it a validation split.")
