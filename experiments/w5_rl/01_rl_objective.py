"""
W5/A - Trace nanochat's policy-gradient update on ONE real GSM8K item, end to end,
with the objective verified numerically against the formula.

Run: python -m experiments.w5_rl.01_rl_objective
"""
import json, math, os, torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

NUM_SAMPLES, MAX_NEW, TEMP, TOPK = 8, 128, 1.0, 50
ddp, rank, lrank, world, device = compute_init(autodetect_device_type())
model, tok, meta = load_model("sft", device, phase="eval", model_tag="d6")
engine = Engine(model, tok)
task = GSM8K(subset="main", split="train")
assistant_end = tok.encode_special("<|assistant_end|>")

print("=" * 98)
print("1. ONE ITEM -> MANY ROLLOUTS")
print("=" * 98)
SCAN = 12
print(f"  Scanning up to {SCAN} training items for one with NONZERO advantage variance, i.e. a")
print(f"  group whose rewards are not unanimous. This scan is itself the measurement that")
print(f"  W5/B needs: the fraction of items that produce any gradient at all.")
scan_rows, chosen = [], None
for j in range(SCAN):
    c = task[j]
    pids = tok.render_for_completion(c)
    ss, mm = engine.generate_batch(pids, num_samples=NUM_SAMPLES, max_tokens=MAX_NEW,
                                   temperature=TEMP, top_k=TOPK, seed=1234 + j)
    rr = [task.reward(c, tok.decode(x[len(pids):])) for x in ss]
    scan_rows.append((j, sum(rr), len(rr)))
    if chosen is None and 0 < sum(rr) < len(rr):
        chosen = (c, pids, ss, mm, rr, j)
solved = sum(1 for _, s_, n_ in scan_rows if 0 < s_ < n_)
allzero = sum(1 for _, s_, n_ in scan_rows if s_ == 0)
allone = sum(1 for _, s_, n_ in scan_rows if s_ == n_)
print(f"\n  {'item':>5} {'rewards summed':>15} {'group':>7} {'advantage variance':>20}")
for j, s_, n_ in scan_rows:
    print(f"  {j:>5} {s_:>15.0f} {n_:>7} {'NONZERO' if 0 < s_ < n_ else 'zero (unanimous)':>20}")
print(f"\n  informative items: {solved}/{SCAN}   all-wrong: {allzero}   all-right: {allone}")
print(f"  -> {100*(SCAN-solved)/SCAN:.0f}% of rollout compute produced EXACTLY ZERO gradient.")

if chosen is None:
    print(f"\n  No item in the scan had mixed rewards. The d6 SFT model solves essentially no")
    print(f"  GSM8K at k={NUM_SAMPLES}, so REAL advantages are all zero and cannot exercise the")
    print(f"  objective arithmetic. Falling back to SYNTHETIC advantages for the identity check")
    print(f"  in section 3 - the arithmetic is what is being verified there, not the rewards.")
    conv = task[0]
    prompt_ids = tok.render_for_completion(conv)
    seqs, masks = engine.generate_batch(prompt_ids, num_samples=NUM_SAMPLES, max_tokens=MAX_NEW,
                                        temperature=TEMP, top_k=TOPK, seed=1234)
    rewards = [task.reward(conv, tok.decode(x[len(prompt_ids):])) for x in seqs]
    SYNTHETIC = True
else:
    conv, prompt_ids, seqs, masks, rewards, jsel = chosen
    SYNTHETIC = False
    print(f"\n  Using item {jsel}, which has mixed rewards.")
q = conv["messages"][0]["content"]
print(f"\n  question: {q[:150]}...")
print(f"  render_for_completion pops the reference answer and appends <|assistant_start|>,")
print(f"  so the model is primed to complete. prefix_length = {len(prompt_ids)} tokens.")
print(f"\n  {NUM_SAMPLES} rollouts at temperature={TEMP}, top_k={TOPK}:")
print(f"  {'i':>3} {'gen tokens':>11} {'reward':>7}  response tail")
for i, (s, r) in enumerate(zip(seqs, rewards)):
    t = " ".join(tok.decode(s[len(prompt_ids):]).split())
    print(f"  {i:>3} {len(s)-len(prompt_ids):>11} {r:>7.1f}  ...{t[-70:]!r}")

# ============================================================ 2
print("\n" + "=" * 98)
print("2. GROUP-MEAN ADVANTAGE (this is the entire 'GRPO' of nanochat's GRPO)")
print("=" * 98)
R = torch.tensor(rewards, dtype=torch.float, device=device)
if SYNTHETIC:
    R = torch.tensor([1., 0., 1., 0., 0., 1., 0., 0.][:NUM_SAMPLES], dtype=torch.float, device=device)
    print("  [SYNTHETIC REWARDS - real rollouts were unanimous; verifying arithmetic only]")
mu = R.mean()
A = R - mu
print(f"  rewards    = {[f'{r:.0f}' for r in rewards]}")
print(f"  mu         = mean(rewards) = {mu.item():.4f}")
print(f"  advantages = rewards - mu  = {[f'{a:+.3f}' for a in A.tolist()]}")
print(f"  sum(advantages) = {A.sum().item():+.6f}  <- exactly zero by construction")
print(f"""
  WHY THE GROUP MEAN IS A VALID BASELINE: the REINFORCE gradient is
      E[ (R - b) * grad log pi ]  for ANY b that does not depend on the sampled action,
  because E[ b * grad log pi ] = b * grad E[1] = 0. The group mean depends on the OTHER
  samples for the same prompt, not on the action being credited, so it is unbiased and it
  removes the per-prompt difficulty offset: an easy question where everything scores 1 and
  a hard one where everything scores 0 both contribute zero, instead of flooding the update
  with a constant that only reflects question difficulty.

  WHEN ALL REWARDS ARE EQUAL: mu == r, so every advantage is 0, pg_obj is 0, and the
  gradient is EXACTLY zero - the item contributes nothing but still costs a full set of
  rollouts plus a forward/backward. That is the central efficiency problem of this method:
  fully-solved and fully-failed items are pure waste.""")
degenerate = (A.abs().max().item() == 0)
print(f"  this item: all rewards equal? {degenerate}  -> "
      f"{'ZERO gradient, rollouts wasted' if degenerate else 'nonzero gradient, item is informative'}")

# ============================================================ 3
print("\n" + "=" * 98)
print("3. THE OBJECTIVE, RECOMPUTED TWO WAYS")
print("=" * 98)
max_len = max(len(s) for s in seqs)
padded = [s + [assistant_end] * (max_len - len(s)) for s in seqs]
pmask = [m + [0] * (max_len - len(m)) for m in masks]
ids = torch.tensor(padded, dtype=torch.long, device=device)
mask_ids = torch.tensor(pmask, dtype=torch.long, device=device)
inputs = ids[:, :-1]
targets = ids[:, 1:].clone()
targets[mask_ids[:, 1:] == 0] = -1
print(f"  ids {tuple(ids.shape)} -> inputs {tuple(inputs.shape)}, targets {tuple(targets.shape)}")
print(f"  Engine returns mask=0 for prompt tokens AND for forced tool-use tokens, so targets")
print(f"  is -1 there and those positions contribute nothing (chat_rl.py comment).")
print(f"  supervised positions per rollout: {[int(x) for x in (targets >= 0).sum(1).tolist()]}")

logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)   # (B, T)
num_valid = (targets >= 0).sum().clamp(min=1)
pg_code = (logp * A.unsqueeze(-1)).sum() / num_valid                    # num_passes = examples = 1
# recompute from the formula: sum_i A_i * sum_t logp_it, over VALID t only
valid = (targets >= 0)
per_seq = (logp * valid).sum(dim=1)                                     # (B,)
pg_formula = (A * per_seq).sum() / num_valid
print(f"\n  {'rollout':>8} {'advantage':>11} {'sum_t log p':>14} {'A * sum_t log p':>18}")
for i in range(len(seqs)):
    print(f"  {i:>8} {A[i].item():>+11.3f} {per_seq[i].item():>14.3f} {(A[i]*per_seq[i]).item():>+18.3f}")
print(f"\n  num_valid (all rollouts) = {int(num_valid)}")
print(f"  pg_obj from chat_rl.py code path : {pg_code.item():+.9f}")
print(f"  pg_obj from  sum_i A_i*sum_t logp / num_valid : {pg_formula.item():+.9f}")
print(f"  |difference| = {abs(pg_code.item()-pg_formula.item()):.2e}")
print(f"  loss = -pg_obj = {-pg_code.item():+.9f}")
print(f"""
  DERIVATION of -A * sum_t log p:
    maximise J = E_y~pi [ A(y) ].  REINFORCE: grad J = E[ A(y) * grad log pi(y|x) ]
    and log pi(y|x) = sum_t log p(y_t | y_<t, x)  (chain rule).
    So the surrogate whose gradient equals grad J is   sum_i A_i * sum_t log p_it,
    and since optimizers MINIMISE, the loss is its negative:  -A * sum_t log p.
    logp is obtained as -model(..., loss_reduction='none') because cross_entropy returns
    NLL; ignore_index=-1 already zeroes the masked positions, so no extra masking is needed.""")

# ============================================================ 4
print("\n" + "=" * 98)
print("4. NORMALIZATION: token-level (DAPO style), not sequence-level")
print("=" * 98)
seq_level = (A * (per_seq / valid.sum(1).clamp(min=1))).sum() / len(seqs)
print(f"  nanochat divides ONE total by num_valid summed over the whole group:")
print(f"     pg = sum_i A_i * sum_t logp_it  /  sum_i T_i        = {pg_code.item():+.9f}")
print(f"  sequence-level normalization would instead average per-sequence means:")
print(f"     pg = (1/N) sum_i A_i * (sum_t logp_it / T_i)        = {seq_level.item():+.9f}")
print(f"  lengths this batch: {[int(x) for x in valid.sum(1).tolist()]}")
print(f"  Token-level weighting means a LONG rollout contributes proportionally more gradient")
print(f"  than a short one with the same advantage. If correct answers are systematically")
print(f"  longer (they usually are - more reasoning steps), token-level normalization")
print(f"  quietly amplifies a length preference on top of the correctness signal. This is the")
print(f"  same structural bias as DPO's summed log-probs (W4/01 section 6), arriving by a")
print(f"  different route, and it is why response length must be logged next to reward.")

# ============================================================ 5
print("\n" + "=" * 98)
print("5. WHAT nanochat OMITS FROM GRPO / PPO (from the file's own docstring, verified in code)")
print("=" * 98)
print(f"""  {'component':<34} {'PPO/GRPO':<26} {'nanochat':<34}
  {'-'*34} {'-'*26} {'-'*34}
  {'KL to a reference policy':<34} {'trust region term':<26} {'ABSENT - no reference model loaded':<34}
  {'importance ratio + clip':<34} {'pi_new/pi_old, clipped':<26} {'ABSENT - strictly on-policy':<34}
  {'advantage normalization':<34} {'(r-mu)/sigma  z-score':<26} {'(r-mu) only, no sigma divide':<34}
  {'normalization level':<34} {'sequence-level':<26} {'token-level (DAPO style)':<34}
  {'value network / GAE':<34} {'critic estimates V(s)':<26} {'ABSENT - group mean is the baseline':<34}
  {'multiple epochs per batch':<34} {'yes, hence the ratio':<26} {'one pass, so no ratio is needed':<34}

  CONSEQUENCES, not just differences:
   - No KL and no clip means NOTHING bounds how far the policy moves from the SFT model. The
     only brakes are the LR and the horizon. Capability retention (W3/02-style pretraining
     bpb) must therefore be measured explicitly; RL here can silently destroy the base model.
   - Dropping the sigma divide keeps the update magnitude proportional to how much reward
     actually varies. With binary rewards and group size k, sigma is tiny whenever the group
     is nearly unanimous, and dividing by it would explode those updates. This is a
     deliberate stability trade: less normalized, but far less spiky.
   - Strictly on-policy means every gradient step needs fresh rollouts, which is why the RL
     stage is dominated by GENERATION time, not backward time.""")
