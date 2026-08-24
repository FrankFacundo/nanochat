"""
W4/A - Derive DPO from response log ratios, then CHECK the derivation numerically
against nanochat/dpo.py. Every hand number below is recomputed with math.* and asserted
equal to the implementation, so the derivation cannot silently drift from the code.

Run: python -m experiments.w4_dpo.01_derivation
"""
import math
import torch
from nanochat.dpo import dpo_loss, sequence_logprobs

sig = lambda z: 1.0 / (1.0 + math.exp(-z))
T = lambda *v: torch.tensor(v, dtype=torch.float64)

print("=" * 96)
print("1. THE DERIVATION")
print("=" * 96)
print("""  Start from the RLHF objective: maximise E[r(x,y)] - beta * KL( pi_theta || pi_ref ).
  Its exact solution is the Gibbs distribution
        pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp( r(x,y) / beta )
  Take logs and solve for the reward:
        r(x,y) = beta * log( pi*(y|x) / pi_ref(y|x) ) + beta * log Z(x)
  DPO's move: treat the POLICY as the reward model, i.e. define the implicit reward
        r_theta(x,y) = beta * ( log pi_theta(y|x) - log pi_ref(y|x) )                    (*)
  Now fit r_theta to preferences with the Bradley-Terry likelihood
        P(y_w > y_l | x) = sigmoid( r(x,y_w) - r(x,y_l) )
  The intractable partition term beta*log Z(x) is IDENTICAL for y_w and y_l (it depends
  only on x), so it CANCELS in the difference. That cancellation is the whole trick - it
  is why DPO needs no reward model, no sampling, and no RL loop. Substituting (*):

        L_DPO = - log sigmoid( beta * [ (log pi_th(y_w|x) - log pi_ref(y_w|x))
                                      - (log pi_th(y_l|x) - log pi_ref(y_l|x)) ] )

  and log pi(y|x) = SUM_t log p(y_t | x, y_<t) by the chain rule. A sum, never a mean:
  averaging would divide by |y|, which is not part of any probability and would make the
  loss depend on an arbitrary normalization choice.""")

print("\n" + "=" * 96)
print("2. THE GRADIENT, AND WHY THE REFERENCE IS FROZEN")
print("=" * 96)
print("""  d/d_theta L = - beta * sigmoid( -beta * margin ) * [ grad log pi_th(y_w) - grad log pi_th(y_l) ]

  Read it as: push UP the chosen response, push DOWN the rejected one, with a per-pair
  weight sigmoid(-beta*margin) that is LARGE exactly when the model currently has the pair
  ranked wrong. It is a self-weighting hard-example miner.

  pi_ref appears only inside `margin`, as a constant offset. If it were trainable, the pair
  could be satisfied by DRAGGING THE REFERENCE DOWN instead of improving the policy, and the
  KL anchor - the only thing preventing the policy from collapsing onto a degenerate
  high-reward mode - would disappear. nanochat/dpo.py asserts requires_grad is False on both
  reference tensors so this cannot happen by accident.""")

pc = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
pr = torch.tensor([-5.0], dtype=torch.float64, requires_grad=True)
beta, margin = 0.1, (-3.0 - -5.0) - (-4.0 - -4.5)
loss, _ = dpo_loss(pc, pr, T(-4.0), T(-4.5), beta=beta)
loss.backward()
hand_g = -beta * sig(-beta * margin)
print(f"\n  numeric check of the gradient formula (beta={beta}, margin={margin}):")
print(f"    hand   dL/d log_pi(chosen)   = -beta*sigmoid(-beta*margin) = {hand_g:+.12f}")
print(f"    torch  pc.grad               = {pc.grad.item():+.12f}   match={abs(hand_g-pc.grad.item())<1e-12}")
print(f"    hand   dL/d log_pi(rejected) = {-hand_g:+.12f}")
print(f"    torch  pr.grad               = {pr.grad.item():+.12f}   match={abs(hand_g+pr.grad.item())<1e-12}")
assert abs(hand_g - pc.grad.item()) < 1e-12 and abs(hand_g + pr.grad.item()) < 1e-12

print("\n" + "=" * 96)
print("3. WORKED EXAMPLE, COMPUTED BY HAND THEN VERIFIED")
print("=" * 96)
print("""  Prompt x = "<|user_start|>What is 2+2?<|user_end|><|assistant_start|>"
  chosen   y_w = "4"          rejected y_l = "I think it might be 5, but I am not sure."
  Suppose the token log-probs on the RESPONSE positions only (prompt masked out) are:""")
chosen_toks = [("4", -0.30), ("<|assistant_end|>", -0.20)]
rej_toks = [("I", -1.10), (" think", -0.90), (" it", -0.80), (" might", -1.20),
            (" be", -0.60), (" 5", -2.00), (",", -0.70), ("...", -1.50), ("<|assistant_end|>", -0.40)]
print(f"\n    {'chosen y_w':<22}{'log p':>8}      {'rejected y_l':<22}{'log p':>8}")
for i in range(max(len(chosen_toks), len(rej_toks))):
    a = f"{chosen_toks[i][0]:<22}{chosen_toks[i][1]:>8.2f}" if i < len(chosen_toks) else " " * 30
    b = f"{rej_toks[i][0]:<22}{rej_toks[i][1]:>8.2f}" if i < len(rej_toks) else ""
    print(f"    {a}      {b}")
lpc = sum(v for _, v in chosen_toks)
lpr = sum(v for _, v in rej_toks)
print(f"\n    log pi_theta(y_w|x) = sum = {lpc:+.2f}   ({len(chosen_toks)} tokens)")
print(f"    log pi_theta(y_l|x) = sum = {lpr:+.2f}   ({len(rej_toks)} tokens)")
lrc, lrr = -0.60, -8.00
print(f"    reference (frozen SFT model): log pi_ref(y_w|x) = {lrc:+.2f}, log pi_ref(y_l|x) = {lrr:+.2f}")

BETA = 0.1
pol_ratio = lpc - lpr
ref_ratio = lrc - lrr
marg = pol_ratio - ref_ratio
z = BETA * marg
hand_loss = -math.log(sig(z))
print(f"\n  BY HAND:")
print(f"    policy log-ratio    = {lpc:+.2f} - ({lpr:+.2f}) = {pol_ratio:+.2f}")
print(f"    reference log-ratio = {lrc:+.2f} - ({lrr:+.2f}) = {ref_ratio:+.2f}")
print(f"    margin              = {pol_ratio:+.2f} - ({ref_ratio:+.2f}) = {marg:+.2f}")
print(f"    z = beta * margin   = {BETA} * {marg:+.2f} = {z:+.3f}")
print(f"    sigmoid(z)          = {sig(z):.9f}")
print(f"    L = -log sigmoid(z) = {hand_loss:.9f}")
print(f"    implicit rewards: r(y_w) = beta*({lpc:+.2f} - {lrc:+.2f}) = {BETA*(lpc-lrc):+.3f}")
print(f"                      r(y_l) = beta*({lpr:+.2f} - {lrr:+.2f}) = {BETA*(lpr-lrr):+.3f}")

loss2, m2 = dpo_loss(T(lpc), T(lpr), T(lrc), T(lrr), beta=BETA)
print(f"\n  FROM nanochat/dpo.py:")
print(f"    loss            = {loss2.item():.9f}   |diff| = {abs(loss2.item()-hand_loss):.2e}")
print(f"    chosen_reward   = {m2['chosen_reward'].item():+.3f}")
print(f"    rejected_reward = {m2['rejected_reward'].item():+.3f}")
print(f"    reward_margin   = {m2['reward_margin'].item():+.3f}")
print(f"    pref_accuracy   = {m2['preference_accuracy'].item():.1f}")
assert abs(loss2.item() - hand_loss) < 1e-12

print("\n" + "=" * 96)
print("4. WHAT MASKING MUST EXCLUDE, AND WHY EACH ONE MATTERS")
print("=" * 96)
print("""  Only the ASSISTANT RESPONSE tokens enter log pi(y|x). Excluded:
    prompt tokens   - they are the conditioning x. They are IDENTICAL in the chosen and
                      rejected branches, so including them adds the same constant to both
                      log-probs. In the margin it cancels... but ONLY if both branches
                      were padded identically. Include the prompt and the loss silently
                      becomes padding-dependent. Exclude it and the question never arises.
    padding         - contributes log p of a pad token, pure noise scaled by how much
                      padding a batch happened to need.
    tool output     - generated by the interpreter at test time, not by the model (W3/01).
    <|assistant_end|> - INCLUDED. Deciding to stop is part of the response being preferred.
  nanochat/dpo.py takes response_mask explicitly rather than deriving it, so the caller
  cannot accidentally inherit the pretraining convention where every token is a target.""")
print(f"  Verified by test_masking_ignores_prompt_and_padding: two sequences with the same")
print(f"  response tokens but different prompts and different padding give byte-identical")
print(f"  response log-probs.")

print("\n" + "=" * 96)
print("5. THE BETA LIMITS, NUMERICALLY")
print("=" * 96)
print(f"  Fixed pair with margin = {marg:+.2f} (policy already prefers the chosen response).")
print(f"  {'beta':>10} {'z=beta*margin':>15} {'sigmoid(z)':>12} {'loss':>12} {'|dL/dlogpi_w|':>15}")
for b in [1e-6, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]:
    zz = b * marg
    print(f"  {b:>10} {zz:>15.4f} {sig(zz):>12.6f} {-math.log(sig(zz)):>12.6f} {b*sig(-zz):>15.6g}")
print(f"""
  beta -> 0   : z -> 0, loss -> log 2 = {math.log(2):.6f} for EVERY pair, and the gradient
                magnitude beta*sigmoid(-z) -> 0. The KL penalty dominates completely: the
                policy is pinned to the reference and learns nothing. DPO degenerates to a
                constant.
  beta large  : sigmoid saturates. Correctly-ranked pairs contribute ~0 loss and ~0 gradient;
                wrongly-ranked pairs contribute a loss that grows LINEARLY in beta
                (-log sigmoid(z) -> -z for z << 0). Training is then driven almost entirely
                by the mislabeled and ambiguous tail of the dataset - which is exactly the
                subset most likely to be noise. High beta does not mean 'trust preferences
                more', it means 'trust your worst labels more'.
  In between  : beta is the KL leash length. Report it with every DPO number; a DPO result
                without its beta is uninterpretable.""")

print("\n" + "=" * 96)
print("6. LENGTH BIAS - the failure mode built into the objective")
print("=" * 96)
print(f"  log pi(y|x) is a SUM over tokens, and every token log-prob is negative. So longer")
print(f"  responses have systematically MORE NEGATIVE log-probs, purely by length:")
print(f"    chosen  : {len(chosen_toks)} tokens -> {lpc:+.2f}   ({lpc/len(chosen_toks):+.3f} per token)")
print(f"    rejected: {len(rej_toks)} tokens -> {lpr:+.2f}   ({lpr/len(rej_toks):+.3f} per token)")
print(f"  Per token the rejected response is actually only {abs(lpr/len(rej_toks)) - abs(lpc/len(chosen_toks)):+.3f} nats worse, but in")
print(f"  SUM it is {lpr-lpc:+.2f} nats worse - {abs(lpr-lpc)/abs(lpr/len(rej_toks) - lpc/len(chosen_toks)):.1f}x amplified by the length difference alone.")
print(f"""
  If chosen responses in the dataset are systematically SHORTER (or longer) than rejected
  ones, the fastest way to lower the loss is to shift the model's length distribution -
  a change that raises preference accuracy while teaching nothing about quality.
  The reference log-ratio absorbs part of this (pi_ref pays the same length cost), which
  is exactly why the pi_ref term is not optional bookkeeping. It does NOT absorb all of it,
  because the policy can still change lengths relative to the reference.
  MEASURE IT: report mean token counts for chosen and rejected, and correlate the
  per-pair margin with the length difference. If |correlation| is large, the preference
  signal and the length signal are not separable in that dataset.""")
