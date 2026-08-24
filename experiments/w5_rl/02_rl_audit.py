"""
W5/B - Interrogate reward improvement: is it reasoning, or is it the measuring device?

Three runnable audits, no RL training required:
  A) the VERIFIER, attacked directly with adversarial strings
  B) GROUP SIZE: information vs compute, from the real measured solve rate
  C) the JOINT metric table and the claim-calibration rule

Run: python -m experiments.w5_rl.02_rl_audit
"""
import math, re, json, os
from tasks.gsm8k import GSM8K, extract_answer, GSM_RE

task = GSM8K(subset="main", split="train")
conv = task[0]                      # gold answer is "#### 16"
gold = extract_answer(conv["messages"][-1]["content"][-1]["text"])

print("=" * 100)
print("A. VERIFIER AUDIT - attacking tasks/gsm8k.py directly")
print("=" * 100)
print(f"  The whole reward is: extract_answer(response) == extract_answer(reference), string compare.")
print(f"    GSM_RE = {GSM_RE.pattern!r}")
print(f"    gold answer for item 0 = {gold!r}")
print(f"\n  {'response fed to reward()':<62} {'extracted':>12} {'reward':>7}")
cases = [
    ("#### 16",                                                   "exact"),
    ("The answer is #### 16",                                     "prefixed prose"),
    ("#### 16\n#### 99",                                          "two markers, first wins"),
    ("#### 99\n#### 16",                                          "two markers, first wins"),
    ("#### 16.0",                                                 "float form of a correct answer"),
    ("#### 016",                                                  "leading zero"),
    ("#### 16,",                                                  "trailing comma inside the class"),
    ("####16",                                                    "no space - regex needs one"),
    ("#### -16",                                                  "sign flipped"),
    ("I don't know.",                                             "no marker at all"),
    ("#### 1 #### 6",                                             "split digits"),
    ("Reasoning is wrong but #### 16",                            "right answer, broken reasoning"),
    ("#### 16" + " garbage" * 20,                                 "correct then padding"),
]
rows = []
for text, label in cases:
    r = task.reward(conv, text)
    rows.append((label, text, extract_answer(text), r))
    disp = (text[:56] + "...") if len(text) > 59 else text
    print(f"  {disp!r:<62} {str(extract_answer(text)):>12} {r:>7.1f}   {label}")

print(f"""
  FINDINGS:
   1. The verifier reads the FIRST '#### <num>' and ignores everything else. Reasoning is
      never checked. '#### 16' preceded by completely wrong arithmetic scores a full 1.0.
      Reward here measures ANSWER MATCHING, not reasoning - any claim about "better
      reasoning" from reward alone is unsupported by construction.
   2. Formatting is brittle in ways that are NOT about correctness: '####16' (no space)
      and '#### 16.0' both fail while being right. Some of what RL learns is the emission
      of a literal '#### ' marker - a pure format skill that inflates reward with zero
      capability change.
   3. extract_answer returns None when nothing matches. Reward is `pred == ref`, so a
      response with no marker is only safe because the GSM8K reference ALWAYS has one
      (checked below). If a reference ever lacked '####', None == None would award 1.0
      to a model that said nothing.""")
missing = sum(1 for i in range(200)
              if extract_answer(task[i]["messages"][-1]["content"][-1]["text"]) is None)
print(f"      -> references without a '####' marker in the first 200 train items: {missing}")
print(f"      The None==None hazard is currently unreachable, but it is one dataset swap away.")
print(f"      A verifier should return a distinguished FAILURE, never a value that can match.")

print("\n" + "=" * 100)
print("B. GROUP SIZE: information per rollout vs compute per rollout")
print("=" * 100)
print("""  A group of k rollouts on one prompt yields a NONZERO gradient only if the rewards are
  not unanimous. With binary reward and per-sample success probability p:
        P(informative) = 1 - (1-p)^k - p^k
  Cost grows linearly in k; information grows and then saturates.""")
print(f"\n  {'p':>6} " + " ".join(f"{'k='+str(k):>9}" for k in (2, 4, 8, 16)) +
      "     informative-per-rollout (k=2..16)")
for p in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75):
    inf = {k: 1 - (1 - p) ** k - p ** k for k in (2, 4, 8, 16)}
    eff = " ".join(f"{inf[k]/k:>7.3f}" for k in (2, 4, 8, 16))
    print(f"  {p:>6.2f} " + " ".join(f"{inf[k]:>9.3f}" for k in (2, 4, 8, 16)) + f"     {eff}")
print(f"""
  READ IT AS: P(informative) is what you buy; P(informative)/k is what you buy per unit of
  compute. Efficiency per rollout is ALWAYS highest at k=2 - large groups buy a higher hit
  rate at worse marginal cost. Large k is justified only when p is far from 0.5, because
  that is where a small group is almost certainly unanimous and returns nothing at all.
  The catalog's w5-rl-baseline (k=4) vs w5-rl-samples8 (k=8) is exactly this trade, and it
  is NOT compute-matched: k=8 costs 2x the generation for the same number of optimizer
  steps. Comparing them at fixed steps compares 2x the compute; comparing at fixed wall
  time is the honest axis for a "which k should I use" claim.""")

print(f"\n  MEASURED ON THIS MODEL (experiment 01): 0 of 12 items were informative at k=8,")
print(f"  i.e. 0 successes in 96 samples. A 95% upper bound on p is about {1-0.05**(1/96):.3f}.")
for p in (0.005, 0.01, 0.031):
    print(f"    if p = {p:<5} -> P(informative | k=8) = {1-(1-p)**8-p**8:.4f}, "
          f"so ~{1/(1-(1-p)**8-p**8):.0f} items of rollouts per single gradient-carrying item")
print(f"  CONCLUSION FOR THIS MODEL: RL on GSM8K at d6 is not a weak intervention, it is an")
print(f"  approximately EMPTY one. The correct next move is not tuning k, it is raising p")
print(f"  (better SFT, easier task subset, or more sampling budget per item) until the group")
print(f"  is not unanimous. Any reward curve produced here would be noise around zero.")

print("\n" + "=" * 100)
print("C. THE JOINT METRIC TABLE (never read reward alone)")
print("=" * 100)
print("""  metric                     what it detects                  failure it exposes
  -------------------------- -------------------------------- ---------------------------------
  mean reward                the optimised quantity           nothing on its own
  pass@1 (greedy, held-out)  deployable single-shot skill     reward up, pass@1 flat -> the
                                                              model is exploiting sampling,
                                                              not improving
  pass@k (same k as train)   breadth of the solution set      pass@k DOWN while pass@1 up ->
                                                              entropy collapse: RL sharpened
                                                              onto one mode and lost coverage
  response length            the token-level normalization
                             bias (W5/01 section 4)           length up with reward -> length
                                                              is doing the work
  nonzero-advantage fraction  whether ANY gradient flowed     near 0 -> the run is a no-op
                                                              regardless of the reward curve
  wall time / rollout         where compute actually goes     on-policy RL is generation-bound
  pretraining val bpb        capability retention             no KL term exists in this
                             (base vs post-RL, W3/02 method)  implementation, so nothing else
                                                              bounds the drift

  IF REWARD RISES WHILE pass@1 IS FLAT, the admissible conclusions are:
    (a) the model got better at the SAMPLING distribution used in training but not at the
        greedy policy that pass@1 measures - a temperature/entropy effect, not a capability;
    (b) the model learned verifier-satisfying FORMAT (emitting '#### <num>' reliably), which
        raises reward on both correct and incorrect reasoning - audit A shows the verifier
        cannot tell these apart;
    (c) the training and evaluation prompts differ enough that the gain does not transfer.
  What is NOT admissible: "the model reasons better". That claim requires pass@1 to move on
  a held-out set, with response length and format-compliance rate reported alongside so the
  reader can rule out (a) and (b).

  PRE-REGISTERED DECISION RULE: report the k=4 vs k=8 comparison at FIXED WALL TIME, with
  pass@1 as the primary endpoint, reward as a secondary, and the nonzero-advantage fraction
  as a validity gate. If that fraction is below ~0.1, report the run as UNINFORMATIVE and
  do not compare arms at all - which, on the evidence above, is the expected outcome at d6.""")
