# W1 · Reason from bytes to BPB — evidence pack

Four runnable experiments, one per rubric criterion. Every number below was
produced by the scripts in this directory against this repo's real ClimbMix
shards, real rustbpe tokenizers, and the real `d6` checkpoint.

```bash
source .venv/bin/activate
python -m experiments.w1_bpb.01_bpe_mechanism     # ~40s
python -m experiments.w1_bpb.02_bpb_derivation    # ~3min (loads d6 on MPS)
python -m experiments.w1_bpb.03_vocab_capacity    # ~5s (meta device, no memory)
python -m experiments.w1_bpb.04_fair_comparison   # ~5s
```

Logs are committed under `logs/`. Attach the four `.py` files and `logs/*` as
evidence.

## 01 — BPE mechanism (rubric: 20 pts)
| claim | evidence |
|---|---|
| starts from the 256 raw bytes | ranks 0..255 are all length-1 and cover every byte value |
| learns merges by frequency | rank 256 `b' '+b't'`, 258 `b'i'+b'n'`, 262 `b' t'+b'he'` — merges compose earlier merges |
| pretokenizer is a hard wall | `' the the'` → `[' the',' the']`; `'1234567'` → `['12','34','56','7']` (`\p{N}{1,2}`) |
| lossless at every vocab | encode→decode asserted on 3.02 MB of held-out val text |
| diminishing compression | 2.021 → 3.038 → 3.993 → 4.683 bytes/token at V = 512/2048/8192/32768 |

## 02 — BPB derivation (rubric: 30 pts)
- **Normalizer is exact, not approximate:** `sum(token_bytes[ids]) = 3,024,593 = len(val_text.utf8)`. Specials carry 0 bytes, so prepending `<|bos|>` changes the token count but not the byte count.
- **Reproduced `evaluate_bpb` by hand on `d6` step 5000:** total 60,735.76 nats over 78,670 non-special bytes → `bpb = 60735.76 / (ln2 × 78670) = 1.113807`, matching the library to 5.4e-08.
- **The identity:** bits/token ÷ bytes/token = 5.356923 / 4.8096 = 1.113807 = bpb. BPB is per-token loss deflated by the compression rate — which is exactly why the tokenizer cancels.
- **Cross-tokenizer PPL is invalid, shown empirically:** four real add-α unigram LMs, one per vocab, same training text, same 3,024,593 val bytes.

| vocab | val tokens | total nats | token PPL | BPB |
|---|---|---|---|---|
| 512 | 1,496,724 | 8,003,111 | **210.00** | 3.8174 |
| 2048 | 995,600 | 6,609,377 | 764.01 | 3.1526 |
| 8192 | 759,787 | 5,598,751 | 1585.80 | 2.6705 |
| 32768 | 649,914 | 5,008,720 | 2223.29 | **2.3891** |

Token PPL ranks them `[512, 2048, 8192, 32768]`; BPB ranks them exactly reversed.
The V=512 model spends **1.60× more total nats** on the identical byte string yet
posts a **10.6× lower** perplexity, because per-token normalization divides by a
denominator the tokenizer chose.

## 03 — Capacity accounting (rubric: 25 pts)
d6 = `n_layer=6, n_embd=384, n_head=6, seq_len=512, window=L`.

**Vocab-dependent tensors — all three, not two:**
1. `transformer.wte` — `(padded_vocab, 384)`
2. `lm_head` — `(padded_vocab, 384)`, an `nn.Linear`, so it is the **only** one that also costs FLOPs
3. `value_embeds` — **3** ResFormer value embeddings (layers 1,3,5 via `has_ve`), each `(padded_vocab, kv_dim=384)`. This is the one people miss, and it is 3× the `wte` cost.

Vocab-invariant: `transformer.h` = 10,617,048 params, scalars = 38.

`d(params)/d(vocab) = 384 + 384 + 3×384 = 1,920` params per token id.
`d(FLOPs/token)/d(vocab) = 6 × 384 = 2,304` — embeddings are lookups, only `lm_head` matmuls.

| vocab | total params | matmul params | FLOPs/token |
|---|---|---|---|
| 4096 | 18,481,406 | 12,189,936 | 87,295,392 |
| 8192 | 26,345,726 | 13,762,800 | 96,732,576 |
| 32768 | 73,531,646 | 23,199,984 | 153,355,680 |
| 100352 | 203,292,926 | 49,152,240 | 309,069,216 |

At V=32768, **62.9M of 73.5M params (85.6%) are vocab-dependent**. Going 8192 → 32768
buys 17.3% fewer tokens per document but costs 2.79× total params and 1.59× FLOPs/token.

## 04 — Fair comparison (rubric: 25 pts)
A 600-step **token**-matched pilot is confounded twice: at equal tokens the V=32768
arm silently reads **+17.3% more text** and burns **+58.5% more FLOPs**.

**Follow-up (concrete):** an isoFLOP sweep. Run each arm at 3 budgets via
`--target-flops={4e15, 8e15, 1.6e16} --num-iterations=-1`, plot val BPB vs cumulative
FLOPs, and read both arms at a common budget. At 8e15 that is 5,048 steps for V=8192
vs 3,184 for V=32768. Plus a byte-matched arm (704 vs 600 steps for D = 46.0 MB) to
separate the data effect from the compute effect.

Readout hygiene: score **only** val BPB; set `--eval-tokens` per arm
(491,520 vs 425,984) so both denominators cover ~2 MB of the *same* val bytes;
establish the noise floor with 3 seeds of one arm first and refuse gaps under ~2σ;
freeze everything but `--vocab-size`. Conclusion is scoped to d6 at this budget —
the vocab-dependent parameter share falls as depth rises, so it does not transfer to d20.
