"""
W1/E2 - BPB from total NLL, and why cross-tokenizer perplexity is invalid.

PART A  proves the normalizer is exactly the UTF-8 byte count, then reproduces
        nanochat's evaluate_bpb() by hand on the real d6 checkpoint and shows
        mean-token-loss / token-perplexity alongside it.
PART B  trains 4 REAL unigram LMs, one per tokenizer vocab (512..32768), on the
        same text and evaluates them on the same held-out bytes. Token perplexity
        and BPB then rank the SAME four models in OPPOSITE orders - which is the
        empirical proof that cross-tokenizer perplexity is not a comparison.

Run: python -m experiments.w1_bpb.02_bpb_derivation
"""
import math, os, json
from collections import Counter
import torch
import rustbpe
from nanochat.tokenizer import get_tokenizer, get_token_bytes, SPLIT_PATTERN, SPECIAL_TOKENS
from nanochat.dataset import parquets_iter_batched
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

OUT = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2)

val_docs = next(parquets_iter_batched(split="val"))
val_text = "\n".join(val_docs)
VAL_BYTES = len(val_text.encode("utf-8"))

# ============================================================ PART A.1
print("=" * 78)
print("A.1  THE NORMALIZER IS EXACTLY THE UTF-8 BYTE COUNT (not an approximation)")
print("=" * 78)
tok = get_tokenizer()
token_bytes = get_token_bytes(device="cpu")
V = tok.get_vocab_size()
special_ids = sorted(tok.encode_special(s) for s in tok.get_special_tokens())
print(f"  vocab_size={V:,}   token_bytes.shape={tuple(token_bytes.shape)}  dtype={token_bytes.dtype}")
print(f"  special token ids {special_ids[:3]}...{special_ids[-1]} -> token_bytes = "
      f"{sorted(set(token_bytes[i].item() for i in special_ids))}  (masked out of the metric)")

ids = tok.encode(val_text)                       # ordinary tokens only, no specials
summed = int(token_bytes[torch.tensor(ids)].sum())
print(f"  sum(token_bytes[ids]) = {summed:,}")
print(f"  len(val_text.utf8)    = {VAL_BYTES:,}")
print(f"  EXACT MATCH: {summed == VAL_BYTES}   <- tokenization partitions the byte string,")
print(f"  so summing per-token byte lengths reconstructs the corpus byte count exactly.")
print(f"  With <|bos|> prepended the sum is unchanged (specials carry 0 bytes):")
ids_bos = tok.encode(val_text, prepend="<|bos|>")
print(f"    tokens {len(ids):,} -> {len(ids_bos):,},  bytes "
      f"{summed:,} -> {int(token_bytes[torch.tensor(ids_bos)].sum()):,}")

# ============================================================ PART A.2
print("\n" + "=" * 78)
print("A.2  REPRODUCE evaluate_bpb() BY HAND ON THE REAL d6 CHECKPOINT")
print("=" * 78)
device_type = autodetect_device_type()
ddp, rank, local_rank, world_size, device = compute_init(device_type)
model, ck_tok, meta = load_model("base", device, phase="eval", model_tag="d6")
seq_len = meta["model_config"]["sequence_len"]
tb = get_token_bytes(device=device)
B, STEPS = 4, 8
print(f"  device={device}  model=d6 step={meta['step']}  seq_len={seq_len}  "
      f"batch={B} steps={STEPS}  ({B*seq_len*STEPS:,} target tokens)")

# hand-rolled: accumulate SUM of nats and SUM of bytes, never a mean
loader = tokenizing_distributed_data_loader_bos_bestfit(ck_tok, B, seq_len, "val", device=device)
total_nats, total_bytes, total_counted_tokens, total_tokens = 0.0, 0, 0, 0
with torch.no_grad():
    it = iter(loader)
    for _ in range(STEPS):
        x, y = next(it)
        loss2d = model(x, y, loss_reduction='none').view(-1)   # nats per target token
        yf = y.view(-1)
        nb = tb[yf]                                            # bytes per target token
        counted = nb > 0                                       # specials -> 0 bytes -> excluded
        total_nats += float((loss2d * counted).sum())
        total_bytes += int(nb.sum())
        total_counted_tokens += int(counted.sum())
        total_tokens += yf.numel()

bpb_hand = total_nats / (LN2 * total_bytes)
mean_token_nats = total_nats / total_counted_tokens
print(f"\n  total_nats            = {total_nats:,.2f} nats")
print(f"  total_bytes           = {total_bytes:,} non-special bytes")
print(f"  counted / all targets = {total_counted_tokens:,} / {total_tokens:,} "
      f"({total_tokens-total_counted_tokens:,} specials masked)")
print(f"  bytes per counted tok = {total_bytes/total_counted_tokens:.4f}")
print(f"\n  bpb = total_nats / (ln2 * total_bytes)")
print(f"      = {total_nats:,.2f} / ({LN2:.6f} * {total_bytes:,})")
print(f"      = {bpb_hand:.6f} bits/byte")

loader2 = tokenizing_distributed_data_loader_bos_bestfit(ck_tok, B, seq_len, "val", device=device)
bpb_lib = evaluate_bpb(model, loader2, STEPS, tb)
print(f"  nanochat evaluate_bpb = {bpb_lib:.6f}   |diff| = {abs(bpb_lib-bpb_hand):.2e}")
print(f"\n  For contrast, the tokenizer-DEPENDENT quantities on the same run:")
print(f"    mean loss/token   = {mean_token_nats:.6f} nats = {mean_token_nats/LN2:.6f} bits/token")
print(f"    token perplexity  = exp(mean nats/token) = {math.exp(mean_token_nats):.3f}")
print(f"    byte perplexity   = 2**bpb              = {2**bpb_hand:.4f}   <- comparable across tokenizers")
print(f"  Identity check: bits/token / (bytes/token) = "
      f"{(mean_token_nats/LN2)/(total_bytes/total_counted_tokens):.6f} == bpb  "
      f"-> bpb is exactly the per-token loss deflated by the compression rate.")

# ============================================================ PART B
print("\n" + "=" * 78)
print("B  FOUR REAL LMs, FOUR VOCABS: token-PPL and BPB RANK THEM OPPOSITELY")
print("=" * 78)
print("  Model class: order-0 unigram over token ids, MLE + add-alpha smoothing.")
print("  Every model sees the same training text and is scored on the same val BYTES.")

TRAIN_CHARS, DOC_CAP, ALPHA = 20_000_000, 10_000, 0.1
def text_iterator():
    n = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            d = doc[:DOC_CAP]; n += len(d); yield d
            if n > TRAIN_CHARS: return

train_text = "\n".join(list(text_iterator()))
rows = []
for VOCAB in [512, 2048, 8192, 32768]:
    t = rustbpe.Tokenizer()
    t.train_from_iterator(text_iterator(), VOCAB - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN)
    ranks = {bytes(k): v for k, v in t.get_mergeable_ranks()}
    nbytes = [0] * len(ranks)
    for b, r in ranks.items():
        nbytes[r] = len(b)
    train_ids = t.encode(train_text)
    val_ids = t.encode(val_text)
    assert t.decode(val_ids) == val_text
    # fit: add-alpha smoothed unigram over the real vocab
    cnt = Counter(train_ids)
    Vt = len(ranks)
    denom = len(train_ids) + ALPHA * Vt
    logp = [math.log((cnt.get(i, 0) + ALPHA) / denom) for i in range(Vt)]
    # score: SUM of nats and SUM of bytes
    nats = -sum(logp[i] for i in val_ids)
    bts = sum(nbytes[i] for i in val_ids)
    assert bts == VAL_BYTES, (bts, VAL_BYTES)
    rows.append(dict(vocab=VOCAB, n_tok=len(val_ids), nats=nats, bytes=bts,
                     bpb=nats / (LN2 * bts), ppl_tok=math.exp(nats / len(val_ids)),
                     bpt=(nats / len(val_ids)) / LN2, bytes_per_tok=bts / len(val_ids)))

print(f"\n  {'vocab':>7} {'val tokens':>11} {'total nats':>13} {'val bytes':>11} "
      f"{'bits/token':>11} {'bytes/token':>12} {'token PPL':>11} {'BPB':>8}")
for r in rows:
    print(f"  {r['vocab']:>7} {r['n_tok']:>11,} {r['nats']:>13,.0f} {r['bytes']:>11,} "
          f"{r['bpt']:>11.4f} {r['bytes_per_tok']:>12.4f} {r['ppl_tok']:>11.2f} {r['bpb']:>8.4f}")

best_ppl = min(rows, key=lambda r: r['ppl_tok'])
best_bpb = min(rows, key=lambda r: r['bpb'])
print(f"\n  Ranked by token perplexity (lower=better): "
      f"{[r['vocab'] for r in sorted(rows, key=lambda r: r['ppl_tok'])]}")
print(f"  Ranked by BPB              (lower=better): "
      f"{[r['vocab'] for r in sorted(rows, key=lambda r: r['bpb'])]}")
print(f"  token-PPL 'winner' = vocab {best_ppl['vocab']} (PPL {best_ppl['ppl_tok']:.1f});  "
      f"BPB winner = vocab {best_bpb['vocab']} (BPB {best_bpb['bpb']:.4f})")
print(f"  The vocab-{best_ppl['vocab']} model needs {best_ppl['nats']/best_bpb['nats']:.2f}x MORE total nats"
      f" to describe the identical {VAL_BYTES:,} bytes,")
print(f"  yet its per-token perplexity is {best_bpb['ppl_tok']/best_ppl['ppl_tok']:.1f}x LOWER. Per-token")
print(f"  normalization divides by a tokenizer-chosen denominator, so it is not a")
print(f"  property of the data. Only total nats / total bytes is tokenizer-invariant.")

with open(os.path.join(OUT, "logs", "02_results.json"), "w") as f:
    json.dump(dict(val_bytes=VAL_BYTES, d6=dict(step=meta["step"], bpb_hand=bpb_hand, bpb_lib=bpb_lib,
              total_nats=total_nats, total_bytes=total_bytes, token_ppl=math.exp(mean_token_nats)),
              unigrams=rows), f, indent=2)
print(f"\nwrote logs/02_results.json")
