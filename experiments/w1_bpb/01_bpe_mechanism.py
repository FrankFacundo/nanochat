"""
W1/E1 - BPE mechanism, from real trained tokenizers (not from memory).

Trains real rustbpe tokenizers at several vocab sizes on ClimbMix text, then:
  (a) shows the vocabulary starts as the 256 single bytes (ranks 0..255),
  (b) reconstructs and prints the first merges in learned order,
  (c) shows the pretokenizer regex is a hard wall merges cannot cross,
  (d) measures bytes/token on HELD-OUT val text -> the vocab/compression curve,
  (e) confirms encode->decode is lossless at every vocab size.

Run: python -m experiments.w1_bpb.01_bpe_mechanism
"""
import time, json, os
import rustbpe
from nanochat.tokenizer import SPLIT_PATTERN, SPECIAL_TOKENS
from nanochat.dataset import parquets_iter_batched

TRAIN_CHARS = 100_000_000
DOC_CAP = 10_000
VOCABS = [512, 2048, 8192, 32768]
OUT = os.path.dirname(os.path.abspath(__file__))

def text_iterator(max_chars):
    n = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            d = doc[:DOC_CAP]
            n += len(d)
            yield d
            if n > max_chars:
                return

# held-out val text, never seen by any tokenizer trained here
val_docs = next(parquets_iter_batched(split="val"))
val_text = "\n".join(val_docs)
val_bytes = len(val_text.encode("utf-8"))
print(f"Held-out val text: {len(val_docs):,} docs, {val_bytes:,} UTF-8 bytes\n")

results = {}
tokenizers = {}
for V in VOCABS:
    t0 = time.time()
    tok = rustbpe.Tokenizer()
    tok.train_from_iterator(text_iterator(TRAIN_CHARS), V - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN)
    dt = time.time() - t0
    ids = tok.encode(val_text)
    assert tok.decode(ids) == val_text, "BPE must be lossless"
    results[V] = dict(train_s=round(dt, 1), tokens=len(ids), bytes_per_token=val_bytes / len(ids))
    tokenizers[V] = tok
    print(f"vocab={V:>6}  trained in {dt:>5.1f}s  ->  {len(ids):>9,} val tokens  "
          f"{val_bytes/len(ids):>5.3f} bytes/token  (lossless roundtrip OK)")

# ---------------------------------------------------------------- (a) + (b)
print("\n" + "=" * 78)
print("(a) STARTING SYMBOLS: what occupies ranks 0..255 in the vocab=512 tokenizer?")
ranks = {bytes(k): v for k, v in tokenizers[512].get_mergeable_ranks()}
by_rank = sorted(ranks.items(), key=lambda kv: kv[1])
base = [b for b, r in by_rank if r < 256]
print(f"  ranks 0..255: {len(base)} tokens, all length-1? {all(len(b)==1 for b in base)}")
print(f"  set of their byte values == all 256 byte values? {set(b[0] for b in base) == set(range(256))}")
print(f"  i.e. BPE starts from the 256 raw UTF-8 bytes - never characters, never words,")
print(f"  which is why any byte string is encodable and roundtrip is lossless by construction.")

print("\n(b) FIRST LEARNED MERGES (rank >= 256), reconstructed by splitting each token")
print("    into two lower-rank vocab members:")
for tok_bytes, r in by_rank[256:286]:
    cands = [(tok_bytes[:i], tok_bytes[i:]) for i in range(1, len(tok_bytes))
             if tok_bytes[:i] in ranks and tok_bytes[i:] in ranks
             and ranks[tok_bytes[:i]] < r and ranks[tok_bytes[i:]] < r]
    pair = cands[0] if len(cands) == 1 else min(cands, key=lambda p: max(ranks[p[0]], ranks[p[1]]))
    amb = "" if len(cands) == 1 else f"  [{len(cands)} splits, took lowest-rank]"
    print(f"    rank {r:>4}: {pair[0]!r} + {pair[1]!r}  ->  {tok_bytes!r}{amb}")

# ---------------------------------------------------------------- (c)
print("\n(c) THE PRETOKENIZER IS A WALL: no token may span a regex split boundary.")
big = tokenizers[32768]
probe = "hello world"
print(f"  SPLIT_PATTERN = {SPLIT_PATTERN}")
for s in [probe, " the the", "1234567", "  \n\n  x"]:
    ids = big.encode(s)
    pieces = [big.decode([i]) for i in ids]
    print(f"    {s!r:>14} -> {pieces}")
print("  Note ' the the' never merges into one token and \\p{N}{1,2} caps digit runs at 2,")
print("  so 1234567 can never become a single token no matter how frequent it is.")

# ---------------------------------------------------------------- (d)
print("\n(d) VOCAB -> COMPRESSION, measured on held-out val text:")
print(f"    {'vocab':>7} {'val tokens':>12} {'bytes/token':>12} {'tokens vs V=512':>16}")
b512 = results[512]["tokens"]
for V in VOCABS:
    r = results[V]
    print(f"    {V:>7} {r['tokens']:>12,} {r['bytes_per_token']:>12.3f} {r['tokens']/b512:>15.1%}")
print("    Doubling the vocab buys progressively less compression (diminishing returns),")
print("    while parameters/FLOPs in the embedding+unembedding grow strictly linearly.")

with open(os.path.join(OUT, "logs", "01_results.json"), "w") as f:
    json.dump(dict(train_chars=TRAIN_CHARS, val_bytes=val_bytes, results=results), f, indent=2)
print(f"\nwrote logs/01_results.json")
