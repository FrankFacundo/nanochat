"""
W3/A - Trace the SFT learning signal end to end on REAL data.

Follows one GSM8K conversation (which exercises tool use) and one SmolTalk conversation
through: serialization -> token ids -> packing -> shifted targets -> response-only loss
masking -> which parameters the update touches. Then: why --load-optimizer is an
intervention, and what "changing the data mixture" means on three different budget axes.

Run: python -m experiments.w3_sft.01_sft_mechanism
"""
import json, math, torch
from collections import Counter
from nanochat.tokenizer import get_tokenizer
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.common import TaskMixture

tok = get_tokenizer()
SPECIAL = {tok.encode_special(s): s for s in tok.get_special_tokens()}
MAX_SEQ_LEN = 512          # d6 SFT config
ROW_CAP = MAX_SEQ_LEN + 1  # chat_sft: row_capacity = max_seq_len + 1

# ==================================================================== 1
print("=" * 100)
print("1. SERIALIZATION: one GSM8K conversation -> token ids + supervision mask")
print("=" * 100)
conv = GSM8K(subset="main", split="train")[0]
ids, mask = tok.render_conversation(conv, max_tokens=ROW_CAP)
print(f"  conversation has {len(conv['messages'])} messages; renders to {len(ids)} tokens")
print(f"\n  {'i':>4} {'id':>6} {'token':<26} {'mask':>4}  role of this position")
role = "BOS"
for i, (tid, mv) in enumerate(zip(ids, mask)):
    name = SPECIAL.get(tid)
    if name == "<|bos|>": role = "document delimiter"
    elif name == "<|user_start|>": role = "user prompt"
    elif name == "<|assistant_start|>": role = "assistant response"
    elif name == "<|python_start|>": role = "assistant tool CALL"
    elif name == "<|output_start|>": role = "tool OUTPUT (from python, not the model)"
    elif name == "<|output_end|>": role = "assistant response"
    if i < 26 or (30 <= i <= 46):
        s = name if name else repr(tok.decode([tid]))
        print(f"  {i:>4} {tid:>6} {s:<26} {mv:>4}  {role}")
    elif i == 26:
        print(f"  {'...':>4}")
print(f"  ... ({len(ids)} tokens total)")

groups = Counter()
cur = "bos"
for tid, mv in zip(ids, mask):
    n = SPECIAL.get(tid)
    if n == "<|user_start|>": cur = "user prompt"
    elif n == "<|assistant_start|>": cur = "assistant text"
    elif n == "<|python_start|>": cur = "assistant tool call"
    elif n == "<|output_start|>": cur = "tool output"
    elif n == "<|output_end|>": cur = "assistant text"
    groups[(cur, mv)] += 1
print(f"\n  {'content class':<24} {'mask=1 (trained)':>17} {'mask=0 (not trained)':>21}")
for k in ["bos", "user prompt", "assistant text", "assistant tool call", "tool output"]:
    print(f"  {k:<24} {groups[(k,1)]:>17} {groups[(k,0)]:>21}")
print(f"\n  RULES, read off nanochat/tokenizer.py render_conversation:")
print(f"    <|bos|>, user text, <|user_start|>/<|user_end|>  -> mask 0. Conditioning, not a target.")
print(f"    assistant text and <|assistant_end|>             -> mask 1. This is the learning signal.")
print(f"    <|python_start|> CODE <|python_end|>             -> mask 1. The model must learn to CALL the tool.")
print(f"    <|output_start|> RESULT <|output_end|>           -> mask 0. Produced by the Python interpreter")
print(f"      at test time, not by the model. Training on it would teach the model to hallucinate")
print(f"      execution results instead of calling the tool - the single most consequential mask in the file.")

# ==================================================================== 2
print("\n" + "=" * 100)
print("2. PACKING: many conversations per row, best-fit, padded not cropped")
print("=" * 100)
mixture = TaskMixture([SmolTalk(split="train"), MMLU(subset="all", split="auxiliary_train"),
                       GSM8K(subset="main", split="train")])
DEV_BS = 32                       # d6 SFT device_batch_size
buf, cursor = [], 0
def refill():
    global cursor
    while len(buf) < 100:
        buf.append(tok.render_conversation(mixture[cursor], max_tokens=ROW_CAP))
        cursor += 1

rows, mrows, lens, packed_counts = [], [], [], []
for _ in range(DEV_BS):
    row, mrow, npacked = [], [], 0
    content_len = ROW_CAP
    while len(row) < ROW_CAP:
        refill()
        rem = ROW_CAP - len(row)
        best, blen = -1, 0
        for i, (c, _) in enumerate(buf):
            if blen < len(c) <= rem:
                best, blen = i, len(c)
        if best < 0:
            content_len = len(row)
            row += [tok.get_bos_token_id()] * rem
            mrow += [0] * rem
            break
        c, cm = buf.pop(best)
        row += c; mrow += cm; npacked += 1
    rows.append(row); mrows.append(mrow); lens.append(content_len); packed_counts.append(npacked)

pad_tokens = sum(ROW_CAP - l for l in lens)
print(f"  row_capacity = max_seq_len + 1 = {ROW_CAP};  device_batch_size = {DEV_BS}")
print(f"  conversations packed per row: min={min(packed_counts)} median={sorted(packed_counts)[DEV_BS//2]} max={max(packed_counts)}")
print(f"  total conversations in this batch: {sum(packed_counts)}")
print(f"  padding: {pad_tokens:,} of {DEV_BS*ROW_CAP:,} positions ({100*pad_tokens/(DEV_BS*ROW_CAP):.1f}%)")
print(f"  Best-fit picks the LARGEST conversation that still fits, so padding is minimized but")
print(f"  never zero. Nothing is cropped: a conversation that does not fit waits in the buffer,")
print(f"  and render_conversation truncates at row_capacity so an over-long one can never starve it.")

# ==================================================================== 3
print("\n" + "=" * 100)
print("3. SHIFTED TARGETS AND THE THREE MASKS (exactly the arithmetic in chat_sft.py)")
print("=" * 100)
batch = torch.tensor(rows, dtype=torch.long)
inputs = batch[:, :-1].contiguous()
targets = batch[:, 1:].clone()
mask_t = torch.tensor(mrows, dtype=torch.int8)[:, 1:]
n_before = targets.numel()
targets[mask_t == 0] = -1                      # mask 1: response-only supervision
n_after_mask = int((targets >= 0).sum())
for i, cl in enumerate(lens):                  # mask 2: padding
    if cl < ROW_CAP:
        targets[i, cl-1:] = -1
n_after_pad = int((targets >= 0).sum())
print(f"  inputs  = batch[:, :-1]  shape {tuple(inputs.shape)}")
print(f"  targets = batch[:,  1:]  shape {tuple(targets.shape)}   <- position t predicts token t+1")
print(f"  {'stage':<48} {'supervised positions':>21}")
print(f"  {'all target positions':<48} {n_before:>21,}")
print(f"  {'after response-only mask (targets[mask==0] = -1)':<48} {n_after_mask:>21,}")
print(f"  {'after padding mask (targets[i, content_len-1:] = -1)':<48} {n_after_pad:>21,}")
print(f"  supervised fraction = {n_after_pad:,}/{n_before:,} = {100*n_after_pad/n_before:.1f}%")
rowsup = [(int((targets[i] >= 0).sum())) for i in range(DEV_BS)]
print(f"  per-row supervised tokens: min={min(rowsup)} median={sorted(rowsup)[DEV_BS//2]} max={max(rowsup)}")
print(f"  ({sum(1 for r in rowsup if r == 0)} of {DEV_BS} rows supervise NOTHING - legal, because")
print(f"   chat_sft asserts num_supervised > 0 per BATCH, not per row.)")
print(f"\n  -> about {100-100*n_after_pad/n_before:.0f}% of the FLOPs in an SFT step compute logits for")
print(f"     positions that contribute ZERO gradient. Not a bug - the prompt must be encoded to")
print(f"     condition the response - but it means 'SFT tokens' and 'SFT training signal' differ")
print(f"     by ~{n_before/max(n_after_pad,1):.1f}x, which matters for every budget claim in section 5.")
print(f"  -> the mask is applied via ignore_index=-1, which is why loss_eval.evaluate_bpb needs")
print(f"     a separate code path for negative targets (W1/02) and why SFT bpb is measured over")
print(f"     RESPONSE bytes only - it is not comparable to a pretraining bpb over all bytes.")
print(f"  -> chat_sft asserts num_supervised > 0 per batch, because cross_entropy over zero")
print(f"     elements returns nan with zero grad: training would silently no-op while the")
print(f"     optimizer kept stepping on stale momentum.")

# ==================================================================== 4
print("\n" + "=" * 100)
print("4. WHY --load-optimizer IS AN INTERVENTION, NOT A DETAIL")
print("=" * 100)
print(f"""  chat_sft.py defaults to --load-optimizer=1, which calls load_optimizer_state("base", ...)
  and then load_state_dict on the optimizer. What that inherits:
    - Muon momentum buffers for every transformer matrix (momentum 0.95 -> ~20-step memory)
    - AdamW exp_avg / exp_avg_sq for wte, lm_head, value_embeds, scalars
    - and, until the code explicitly restores them, the pretrained param_group LRs
  The last one is a real trap the file documents: pretraining warmdown drives LR to ~0, so a
  naive load would start SFT at LR~0. chat_sft saves base_lrs before the load and restores
  them after (lines 156-161). If that restore were missing, the ablation "SFT with vs without
  optimizer warm start" would silently become "SFT with vs without any learning at all".

  It is an INTERVENTION because it changes the optimizer's state at step 0, which changes the
  first ~20 updates' direction and magnitude - a different dependent variable trajectory from
  the same data and the same weights.

  THE CONTROL:  run the identical SFT twice, --load-optimizer=1 and --load-optimizer=0, same
  parent checkpoint, same mixture, same steps, same seed. Everything else is already frozen by
  the inheritance block (max_seq_len, batch sizes, and all three LRs are inherited from the
  parent's meta.json, so the two arms cannot drift apart on hyperparameters).
    python -m scripts.chat_sft --model-tag=d6 --load-optimizer=1 --num-iterations=1500 ... --run=sft_warm
    python -m scripts.chat_sft --model-tag=d6 --load-optimizer=0 --num-iterations=1500 ... --run=sft_cold
  Read out val bpb AND ChatCORE. Expect the effect to be largest in the first 100 steps and to
  shrink by the end; if it does not shrink, the warm start is doing something other than
  smoothing the transient and deserves its own investigation.""")

# ==================================================================== 5
print("\n" + "=" * 100)
print("5. DATA MIXTURE AT FIXED EXAMPLES vs FIXED TOKENS vs FIXED STEPS")
print("=" * 100)
def stats(task, n=300):
    L, sup = [], []
    for i in range(min(n, len(task))):
        ii, mm = tok.render_conversation(task[i], max_tokens=ROW_CAP)
        L.append(len(ii)); sup.append(sum(mm))
    return sum(L)/len(L), sum(sup)/len(sup)

tasks = {"SmolTalk": SmolTalk(split="train"),
         "MMLU aux": MMLU(subset="all", split="auxiliary_train"),
         "GSM8K": GSM8K(subset="main", split="train")}
print(f"  {'source':<10} {'rows':>9} {'avg tokens/row':>15} {'avg supervised':>15} {'supervised %':>13}")
info = {}
for name, t in tasks.items():
    a, s = stats(t)
    info[name] = (len(t), a, s)
    print(f"  {name:<10} {len(t):>9,} {a:>15.1f} {s:>15.1f} {100*s/a:>12.1f}%")

BASE = dict(mmlu=1, gsm=8)   # w3-sft-math-heavy
CTRL = dict(mmlu=3, gsm=4)   # w3-sft-baseline (chat_sft defaults)
print(f"\n  The catalog's two arms differ ONLY in --mmlu-epochs / --gsm8k-epochs:")
print(f"    w3-sft-baseline   : mmlu x{CTRL['mmlu']}, gsm8k x{CTRL['gsm']}")
print(f"    w3-sft-math-heavy : mmlu x{BASE['mmlu']}, gsm8k x{BASE['gsm']}")
print(f"\n  {'arm':<20} {'rows':>10} {'total tokens':>14} {'supervised tokens':>18} {'GSM8K share of supervision':>27}")
for label, cfg in [("w3-sft-baseline", CTRL), ("w3-sft-math-heavy", BASE)]:
    rows = info["SmolTalk"][0] + cfg["mmlu"]*info["MMLU aux"][0] + cfg["gsm"]*info["GSM8K"][0]
    toks = info["SmolTalk"][0]*info["SmolTalk"][1] + cfg["mmlu"]*info["MMLU aux"][0]*info["MMLU aux"][1] + cfg["gsm"]*info["GSM8K"][0]*info["GSM8K"][1]
    sup  = info["SmolTalk"][0]*info["SmolTalk"][2] + cfg["mmlu"]*info["MMLU aux"][0]*info["MMLU aux"][2] + cfg["gsm"]*info["GSM8K"][0]*info["GSM8K"][2]
    gsm  = cfg["gsm"]*info["GSM8K"][0]*info["GSM8K"][2]
    print(f"  {label:<20} {rows:>10,} {toks:>14,.0f} {sup:>18,.0f} {100*gsm/sup:>26.1f}%")
print(f"""
  FIXED EXAMPLES  - "each arm sees N conversations". Different sources have different
    lengths, so equal rows means UNEQUAL tokens and unequal gradient signal. Use only when
    the claim is about example efficiency (e.g. annotation cost per conversation).
  FIXED TOKENS    - "each arm sees N tokens". Closer to equal compute, but still unequal
    SUPERVISION because supervised% differs per source (table above). Use for data-efficiency
    claims, and report supervised tokens too.
  FIXED STEPS     - what the catalog actually does: both arms run --num-iterations=1500 at the
    same batch size, so both see EXACTLY the same number of tokens and the same FLOPs. The
    mixture only changes WHICH tokens. This is the right axis for "does the mixture help",
    and it is the only one of the three that is automatically compute-matched.
  The trap: at fixed steps, more GSM8K epochs does NOT mean more GSM8K exposure per step
    unless the sampler is proportional - TaskMixture concatenates, so repeating GSM8K x8
    raises its share of rows from {100*CTRL['gsm']*info['GSM8K'][0]/(info['SmolTalk'][0]+CTRL['mmlu']*info['MMLU aux'][0]+CTRL['gsm']*info['GSM8K'][0]):.1f}% to {100*BASE['gsm']*info['GSM8K'][0]/(info['SmolTalk'][0]+BASE['mmlu']*info['MMLU aux'][0]+BASE['gsm']*info['GSM8K'][0]):.1f}% - and simultaneously
    DROPS MMLU from x3 to x1. The catalog's two arms are a JOINT intervention on two sources,
    not a clean GSM8K dose-response. A clean version holds mmlu-epochs fixed at 3 and varies
    only gsm8k-epochs.""")
