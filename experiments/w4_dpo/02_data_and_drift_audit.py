"""
W4/B - Preference data card + drift audit, on preference pairs actually built here.

nanochat ships no preference dataset, so this constructs two candidates from GSM8K and
audits them BEFORE any DPO run:
  D1 "corrupted-answer"  chosen = reference solution, rejected = same solution with the
                         final number perturbed.  Length-controlled by construction.
  D2 "model-vs-reference" chosen = reference solution, rejected = the SFT model's own
                         greedy continuation.  Realistic, and length-confounded.

Then: a shortcut detector (can LENGTH ALONE predict the label?), the reference model's
starting preference accuracy, and the study design for beta / corrupted labels / retention.

Run: python -m experiments.w4_dpo.02_data_and_drift_audit
"""
import json, math, os, random, re, statistics as st
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.dpo import sequence_logprobs, dpo_loss
from tasks.gsm8k import GSM8K

OUT = os.path.dirname(os.path.abspath(__file__))
N_PAIRS = 40
MAX_NEW = 100
random.seed(0)

ddp, rank, lrank, world, device = compute_init(autodetect_device_type())
model, tok, meta = load_model("sft", device, phase="eval", model_tag="d6")
eng = Engine(model, tok)
SEQ = meta["model_config"]["sequence_len"]

def render_prompt(q):
    return ([tok.get_bos_token_id(), tok.encode_special("<|user_start|>")] + tok.encode(q)
            + [tok.encode_special("<|user_end|>"), tok.encode_special("<|assistant_start|>")])

def flatten(content):
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content if p["type"] in ("text",))

print("=" * 96)
print("1. BUILDING THE TWO CANDIDATE PREFERENCE DATASETS")
print("=" * 96)
ds = GSM8K(subset="main", split="train")
pairs_d1, pairs_d2 = [], []
i = 0
while len(pairs_d1) < N_PAIRS and i < len(ds):
    conv = ds[i]; i += 1
    q = conv["messages"][0]["content"]
    ref = flatten(conv["messages"][1]["content"])
    m = re.search(r"####\s*(-?\d+)", ref)
    if not m:
        continue
    gold = int(m.group(1))
    # D1: same text, wrong final answer -> isolates correctness from style and length
    bad = ref[:m.start()] + f"#### {gold + random.choice([1, 2, 3, -1, -2])}"
    pairs_d1.append((q, ref, bad))
    # D2: the SFT model's own greedy answer
    p = render_prompt(q)
    out, _ = eng.generate_batch(p, num_samples=1, max_tokens=MAX_NEW, temperature=0)
    pairs_d2.append((q, ref, tok.decode(out[0][len(p):])))
print(f"  built {len(pairs_d1)} pairs for each dataset from GSM8K train")

def card(name, pairs):
    cl = [len(tok.encode(c)) for _, c, _ in pairs]
    rl = [len(tok.encode(r)) for _, _, r in pairs]
    return dict(name=name, n=len(pairs),
                chosen_len=(st.mean(cl), st.pstdev(cl)), rejected_len=(st.mean(rl), st.pstdev(rl)),
                cl=cl, rl=rl)

cards = [card("D1 corrupted-answer", pairs_d1), card("D2 model-vs-reference", pairs_d2)]
print(f"\n  DATA CARD")
print(f"  {'dataset':<24} {'n':>4} {'chosen tokens':>17} {'rejected tokens':>17} {'mean diff':>11}")
for c in cards:
    print(f"  {c['name']:<24} {c['n']:>4} {c['chosen_len'][0]:>9.1f} +-{c['chosen_len'][1]:<5.1f} "
          f"{c['rejected_len'][0]:>9.1f} +-{c['rejected_len'][1]:<5.1f} "
          f"{c['chosen_len'][0]-c['rejected_len'][0]:>+11.1f}")

print("\n" + "=" * 96)
print("2. SHORTCUT DETECTOR: can LENGTH ALONE predict the preference label?")
print("=" * 96)
print("  Fit the single best length threshold on the pairs and report its accuracy. A")
print("  preference dataset that a 1-feature classifier solves is a dataset where DPO can")
print("  raise preference accuracy without learning anything about quality.")
print(f"\n  {'dataset':<24} {'best length-only accuracy':>27} {'verdict':>28}")
for c in cards:
    diffs = [a - b for a, b in zip(c["cl"], c["rl"])]
    n = len(diffs)
    ties = sum(d == 0 for d in diffs)
    # ties are chance for a length-only rule, so they score 0.5 - NOT a free win
    p_longer = (sum(d > 0 for d in diffs) + 0.5 * ties) / n
    acc = max(p_longer, 1 - p_longer)
    verdict = ("SHORTCUT-SOLVABLE" if acc >= 0.75 else
               "borderline" if acc >= 0.65 else "length-controlled")
    if ties == n:
        verdict = "length-controlled (all ties)"
    print(f"  {c['name']:<24} {acc:>26.1%} {verdict:>28}")
    c["length_only_acc"] = acc

print("\n" + "=" * 96)
print("3. DOES THE REFERENCE MODEL ALREADY RANK THESE PAIRS? (is there anything to learn?)")
print("=" * 96)

def logp_of(q, resp):
    """Sequence log-prob of `resp` given the chat-rendered prompt, response tokens only."""
    p = render_prompt(q)
    r = tok.encode(resp) + [tok.encode_special("<|assistant_end|>")]
    ids = (p + r)[:SEQ + 1]
    if len(ids) < 2:
        return None
    inp = torch.tensor([ids[:-1]], device=device)
    tgt = torch.tensor([ids[1:]], device=device)
    mask = torch.zeros_like(tgt)
    mask[0, len(p) - 1:] = 1                      # supervise only the response positions
    with torch.no_grad():
        return sequence_logprobs(model, inp, tgt, mask).item(), int(mask.sum())

rows = {}
for c, pairs in zip(cards, [pairs_d1, pairs_d2]):
    lc, lr, ok, ntok_c, ntok_r = [], [], 0, [], []
    for q, ch, rj in pairs:
        a, na = logp_of(q, ch)
        b, nb = logp_of(q, rj)
        lc.append(a); lr.append(b); ntok_c.append(na); ntok_r.append(nb)
        ok += a > b
    rows[c["name"]] = dict(acc=ok / len(pairs), lc=lc, lr=lr, ntok_c=ntok_c, ntok_r=ntok_r)
    print(f"  {c['name']:<24} pi_ref ranks chosen > rejected on {ok}/{len(pairs)} pairs = {ok/len(pairs):.1%}")
    print(f"      mean log pi_ref(chosen) = {st.mean(lc):+9.2f} over {st.mean(ntok_c):.1f} tokens "
          f"({st.mean(lc)/st.mean(ntok_c):+.3f}/token)")
    print(f"      mean log pi_ref(rejected)= {st.mean(lr):+9.2f} over {st.mean(ntok_r):.1f} tokens "
          f"({st.mean(lr)/st.mean(ntok_r):+.3f}/token)")
print(f"\n  A starting accuracy near 50% means the pairs carry signal DPO can actually move.")
print(f"  Near 100% means the reference already agrees and DPO will mostly saturate (loss ~0,")
print(f"  gradient ~0) - a dataset that looks great on preference accuracy and teaches nothing.")

print("\n" + "=" * 96)
print("4. LENGTH vs MARGIN CORRELATION (the drift audit that matters)")
print("=" * 96)
def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")

for c in cards:
    r = rows[c["name"]]
    dlen = [a - b for a, b in zip(r["ntok_c"], r["ntok_r"])]
    dlp = [a - b for a, b in zip(r["lc"], r["lr"])]
    if len(set(dlen)) == 1:
        rho = float("nan")
        print(f"  {c['name']:<24} corr(length diff, log-prob diff) = undefined "
              f"(length difference is identically {dlen[0]} by construction - that is the point of D1)")
    else:
        rho = pearson(dlen, dlp)
        print(f"  {c['name']:<24} corr(length diff, log-prob diff) = {rho:+.3f}")
    c["length_margin_corr"] = rho
print(f"\n  Because log pi is a SUM of negative terms, a response that is N tokens longer pays")
print(f"  roughly N * (mean per-token nats) extra. A strong correlation means the preference")
print(f"  margin DPO optimises is largely a length signal wearing a quality costume.")
print(f"  D1 exists precisely to give a length-controlled comparison against D2.")

print("\n" + "=" * 96)
print("5. THE STUDY (pre-registered, before any DPO run)")
print("=" * 96)
d1, d2 = cards[0], cards[1]
print(f"""  SPLIT: pairs are grouped by SOURCE PROBLEM before splitting. A random pair-level split
    would put the same GSM8K question in train and validation with different rejected
    responses, and validation preference accuracy would then measure memorisation.
    80/20 by problem id, frozen and hashed before the first run.

  ARMS (all from the same SFT parent used above, step {meta['step']}):
    beta in {{0.01, 0.1, 0.5}}       - the KL leash. W4/01 section 5 predicts near-zero
                                     learning at 0.01 and label-noise domination at 0.5.
    label corruption in {{0%, 15%}} - flip the chosen/rejected assignment on a random 15%.
                                     If validation preference accuracy barely moves, the
                                     metric is being driven by a shortcut, not the labels.
                                     This is the single cheapest test for {d2['name']},
                                     whose length-only accuracy is {d2['length_only_acc']:.0%}.
    dataset in {{D1, D2}}            - length-controlled vs length-confounded, same protocol.

  METRICS, read jointly and never singly:
    preference accuracy (train AND held-out)  - the thing being optimised
    reward margin                             - beta*(logratio), saturation detector
    KL PROXY: mean (log pi_theta - log pi_ref) on held-out RESPONSES. This is exactly the
      quantity DPO's derivation constrains, and dpo.py already returns it as
      chosen_reward/beta. If it grows without bound, the leash has snapped.
    mean response length                      - the primary drift channel
    CAPABILITY RETENTION: pretraining val BPB, base vs post-DPO, identical bytes - the same
      forgetting measurement used in W3/02, where SFT alone already cost +0.52 bpb.

  DECISION RULE: a beta is preferred only if it improves held-out preference accuracy AND
    keeps the KL proxy bounded AND does not shift mean length by more than the seed noise.
    An accuracy gain accompanied by a length shift is reported as a length effect, not a
    quality effect, unless it survives on D1.""")

with open(os.path.join(OUT, "logs", "02_audit.json"), "w") as f:
    json.dump({c["name"]: {k: v for k, v in c.items() if k not in ("cl", "rl")} for c in cards}, f, indent=2)
print(f"\n  wrote logs/02_audit.json")
