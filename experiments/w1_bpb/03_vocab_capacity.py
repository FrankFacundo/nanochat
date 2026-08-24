"""
W1/E3 - Capacity & compute accounting: which tensors move when vocab_size moves?

Builds the runcpu.sh d6 config on the meta device (shapes only, zero memory) at a
sweep of vocab sizes and prints every parameter group plus FLOPs/token. Nothing is
asserted that is not read straight off the model.

Run: python -m experiments.w1_bpb.03_vocab_capacity
"""
import torch
from nanochat.gpt import GPT, GPTConfig

DEPTH, HEAD_DIM, ASPECT, SEQ_LEN, WINDOW = 6, 64, 64, 512, "L"   # runs/runcpu.sh d6
VOCABS = [4096, 8192, 16384, 32768, 65536, 100352]               # last = cl100k-ish

def build(vocab):
    base_dim = DEPTH * ASPECT
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    cfg = GPTConfig(sequence_len=SEQ_LEN, vocab_size=vocab, n_layer=DEPTH,
                    n_head=model_dim // HEAD_DIM, n_kv_head=model_dim // HEAD_DIM,
                    n_embd=model_dim, window_pattern=WINDOW)
    with torch.device("meta"):
        return GPT(cfg)

print(f"d{DEPTH}: n_embd={DEPTH*ASPECT} n_head={DEPTH*ASPECT//HEAD_DIM} seq={SEQ_LEN} window={WINDOW}\n")
rows = []
for v in VOCABS:
    m = build(v)
    p = m.num_scaling_params()
    padded = m.transformer.wte.weight.shape[0]
    ve_dim = next(iter(m.value_embeds.values())).weight.shape[1]
    rows.append(dict(vocab=v, padded=padded, n_ve=len(m.value_embeds), ve_dim=ve_dim,
                     matmul=m.num_matmul_params(), flops=m.estimate_flops(), **p))

hdr = f"{'vocab':>7} {'padded':>7} | {'wte':>10} {'value_emb':>10} {'lm_head':>10} {'blocks':>10} {'scalars':>7} {'TOTAL':>11} | {'matmul_p':>10} {'FLOPs/tok':>11}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['vocab']:>7} {r['padded']:>7} | {r['wte']:>10,} {r['value_embeds']:>10,} {r['lm_head']:>10,} "
          f"{r['transformer_matrices']:>10,} {r['scalars']:>7,} {r['total']:>11,} | {r['matmul']:>10,} {r['flops']:>11,}")

b = rows[VOCABS.index(32768)]
print(f"\nInvariant tensors (identical at every vocab): transformer_matrices={b['transformer_matrices']:,}, scalars={b['scalars']:,}")
print(f"Vocab-dependent tensors: wte, lm_head, and {b['n_ve']} value_embeds of width kv_dim={b['ve_dim']}")
print(f"  => d(params)/d(vocab) = n_embd + n_embd + {b['n_ve']}*{b['ve_dim']} = "
      f"{b['wte']//b['padded'] + b['lm_head']//b['padded'] + b['n_ve']*b['ve_dim']} params per extra token id")
print(f"  => d(FLOPs/tok)/d(vocab) = 6 * n_embd (lm_head only; embeddings are lookups) = {6*(DEPTH*ASPECT)}")

print("\nRelative to vocab=32768:")
for r in rows:
    print(f"  vocab={r['vocab']:>6}: total params x{r['total']/b['total']:.3f}   "
          f"matmul x{r['matmul']/b['matmul']:.3f}   FLOPs/token x{r['flops']/b['flops']:.3f}")
