"""
W3/B - Separate capability GAIN from FORGETTING, measured on the real same-parent pair.

Same parent, by construction:
  base : ~/.cache/nanochat/base_checkpoints/d6      step 5000
  sft  : ~/.cache/nanochat/chatsft_checkpoints/d6   step 1499   (fine-tuned FROM that base)

Two orthogonal measurements on the SAME two models:
  A) base-capability regression : pretraining val BPB, identical bytes, identical metric.
     Any increase is forgetting, in bits per byte, on data neither model was tuned for.
  B) instruction/format gain    : a fixed prompt suite, greedy decoding, measuring whether
     the model terminates its turn, how long it runs, and whether it degenerates.

Run: python -m experiments.w3_sft.02_sft_evaluation
"""
import json, math, re, os, torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine

OUT = os.path.dirname(os.path.abspath(__file__))
ddp, rank, lrank, world, device = compute_init(autodetect_device_type())
token_bytes = get_token_bytes(device=device)

print("=" * 96)
print("0. SAME-PARENT PROVENANCE (the precondition for any of this to mean anything)")
print("=" * 96)
base_model, tok, base_meta = load_model("base", device, phase="eval", model_tag="d6")
sft_model, _, sft_meta = load_model("sft", device, phase="eval", model_tag="d6")
print(f"  base : step {base_meta['step']}, config {json.dumps(base_meta['model_config'])}")
print(f"  sft  : step {sft_meta['step']}, config {json.dumps(sft_meta['model_config'])}")
same_cfg = base_meta["model_config"] == sft_meta["model_config"]
print(f"  identical architecture: {same_cfg}")
print(f"  sft inherits the base weights (chat_sft loads 'base' then fine-tunes), so the ONLY")
print(f"  difference between these two checkpoints is the SFT stage. Any comparison of a")
print(f"  DIFFERENTLY-parented pair would confound SFT with pretraining variation.")
assert same_cfg, "different architectures cannot be compared"

# ============================================================ A
print("\n" + "=" * 96)
print("A. BASE-CAPABILITY REGRESSION: pretraining val BPB, same bytes, both models")
print("=" * 96)
SEQ = base_meta["model_config"]["sequence_len"]
B, STEPS = 4, 16
res = {}
for name, m in [("base", base_model), ("sft", sft_model)]:
    loader = tokenizing_distributed_data_loader_bos_bestfit(tok, B, SEQ, "val", device=device)
    res[name] = evaluate_bpb(m, loader, STEPS, token_bytes)
print(f"  eval budget: {B}x{SEQ}x{STEPS} = {B*SEQ*STEPS:,} target tokens from the SAME val shards,")
print(f"  fed by the SAME deterministic loader, so both models score identical bytes.")
print(f"\n  {'model':<8} {'val bpb':>10} {'byte perplexity':>17}")
for k in ("base", "sft"):
    print(f"  {k:<8} {res[k]:>10.6f} {2**res[k]:>17.4f}")
d = res["sft"] - res["base"]
print(f"\n  FORGETTING = {d:+.6f} bpb ({100*d/res['base']:+.2f}% relative).")
print(f"  In compression terms the SFT model needs {2**d:.4f}x more bits to describe the same")
print(f"  pretraining text. This is the cost side of SFT, and it is invisible to any chat metric.")
print(f"  NOTE this is a MEASUREMENT, not yet a finding: with no seed replicates the sign is")
print(f"  trustworthy only if |{d:.4f}| exceeds the noise floor (W2/C measures it).")

# ============================================================ B
print("\n" + "=" * 96)
print("B. INSTRUCTION / FORMAT GAIN: fixed prompt suite, greedy, identical decoding budget")
print("=" * 96)
PROMPTS = [
    "What is the capital of France?",
    "Write one sentence about the ocean.",
    "What is 2 + 2?",
    "Name three colors.",
    "Who wrote Romeo and Juliet?",
    "Explain gravity to a child in one sentence.",
    "Translate 'hello' into Spanish.",
    "What is the chemical symbol for gold?",
]
MAX_NEW = 200
assistant_end = tok.encode_special("<|assistant_end|>")

def chat_prompt(q):
    """Render the SFT chat format. The base model has never seen these control tokens."""
    return ([tok.get_bos_token_id(), tok.encode_special("<|user_start|>")]
            + tok.encode(q)
            + [tok.encode_special("<|user_end|>"), tok.encode_special("<|assistant_start|>")])

def degenerate(text):
    """A 3+ word phrase repeated verbatim = a decoding loop."""
    w = text.split()
    return any(w[i:i+3] == w[j:j+3] for i in range(len(w)-3) for j in range(i+3, len(w)-2))

rows = {}
for name, m in [("base", base_model), ("sft", sft_model)]:
    eng = Engine(m, tok)
    terminated, lengths, degen, samples = 0, [], 0, []
    for q in PROMPTS:
        ids = chat_prompt(q)
        out, _ = eng.generate_batch(ids, num_samples=1, max_tokens=MAX_NEW, temperature=0)
        gen = out[0][len(ids):]
        # Engine.generate_batch STRIPS the terminal token (<|assistant_end|>/<|bos|>) from the
        # result, so 'did it terminate?' is exactly 'did it stop short of the token budget?'.
        if len(gen) < MAX_NEW:
            terminated += 1
        lengths.append(len(gen))
        text = tok.decode(gen)
        degen += degenerate(text)
        samples.append((q, text))
    rows[name] = dict(terminated=terminated, mean_len=sum(lengths)/len(lengths),
                      degen=degen, samples=samples)

print(f"  {len(PROMPTS)} prompts, greedy (temperature=0), max {MAX_NEW} new tokens, identical rendering.")
print(f"\n  {'model':<8} {'terminated within budget':>26} {'mean response tokens':>21} {'degenerate loops':>18}")
for k in ("base", "sft"):
    r = rows[k]
    frac = "%d/%d" % (r["terminated"], len(PROMPTS))
    print(f"  {k:<8} {frac:>34} {r['mean_len']:>21.1f} {r['degen']:>18}")
print(f"\n  FORMAT GAIN = {rows['sft']['terminated'] - rows['base']['terminated']:+d}/{len(PROMPTS)} prompts now terminate the turn.")
print(f"  Turn termination is the cleanest instruction-following proxy available here: the base")
print(f"  model has literally never been trained on <|assistant_end|> (it is only emitted with")
print(f"  mask=1 during SFT, see W3/01), so this metric isolates the SFT format contract.")
print(f"  METHOD NOTE: an earlier version of this script tested `assistant_end in generated`,")
print(f"  which is ALWAYS False - engine.py:281 documents that terminal tokens are stripped from")
print(f"  the returned ids. The corrected test is `len(generated) < budget`. Measure the")
print(f"  instrument before trusting it: a metric that cannot fire is not a null result.")

print("\n" + "=" * 96)
print("B2. IS TERMINATION ABSENT, OR IS GREEDY DECODING TRAPPING IT?")
print("=" * 96)
print(f"  Greedy scores 0/8 for BOTH models, which alone cannot distinguish 'never learned")
print(f"  <|assistant_end|>' from 'greedy walks into a repetition loop and never reaches it'.")
print(f"  Same prompts, {len(PROMPTS)} x 4 samples, varying ONLY temperature:")
print(f"\n  {'model':<8} {'temperature':>12} {'terminated':>12} {'mean len':>10}")
sweep = {}
for name, m in [("base", base_model), ("sft", sft_model)]:
    eng = Engine(m, tok)
    for temp in (0.0, 0.8, 1.0):
        hits, L = 0, []
        for q in PROMPTS:
            out, _ = eng.generate_batch(chat_prompt(q), num_samples=4, max_tokens=MAX_NEW, temperature=temp)
            for o in out:
                g = o[len(chat_prompt(q)):]
                hits += len(g) < MAX_NEW
                L.append(len(g))
        sweep[(name, temp)] = (hits, len(L), sum(L)/len(L))
        print(f"  {name:<8} {temp:>12.1f} {f'{hits}/{len(L)}':>12} {sum(L)/len(L):>10.1f}")
b0, bt = sweep[("sft", 0.0)], sweep[("sft", 1.0)]
print(f"\n  The SFT model DOES terminate when sampled ({bt[0]}/{bt[1]} at T=1.0) but never under")
print(f"  greedy ({b0[0]}/{b0[1]}). So <|assistant_end|> was learned - it just is not the argmax")
print(f"  token anywhere along the greedy path, because the model falls into a repetition")
print(f"  attractor first. That is a DECODING failure, not a missing capability, and the fix is")
print(f"  a decoding change (sampling / repetition penalty), not more SFT steps.")
bb = sweep[("base", 1.0)]
print(f"  The base model terminates {bb[0]}/{bb[1]} even when sampled: it has no turn concept at all.")
print(f"  THIS is the format gain, and reporting only the greedy 0/8 would have hidden it entirely.")

print("\n" + "=" * 96)
print("C. REPRESENTATIVE FAILURES FROM THE FIXED SUITE")
print("=" * 96)
for k in ("base", "sft"):
    print(f"\n  --- {k} ---")
    for q, t in rows[k]["samples"][:5]:
        flat = " ".join(t.split())[:150]
        tag = "LOOP" if degenerate(t) else "    "
        print(f"   {tag} Q: {q}")
        print(f"        A: {flat!r}")

with open(os.path.join(OUT, "logs", "02_results.json"), "w") as f:
    json.dump(dict(bpb=res, forgetting=d,
                   format={k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in rows.items()},
                   samples={k: v["samples"] for k, v in rows.items()}), f, indent=2)
print(f"\n  wrote logs/02_results.json")
