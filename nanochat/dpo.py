"""
Direct Preference Optimization (DPO) for nanochat.

DPO replaces the RL stage of RLHF with a single classification loss on preference pairs.
For a prompt x with a chosen response y_w and a rejected response y_l:

    r_theta(x, y) = beta * ( log pi_theta(y|x) - log pi_ref(y|x) )        "implicit reward"
    L_DPO         = - log sigmoid( r_theta(x, y_w) - r_theta(x, y_l) )

where log pi(y|x) = SUM over response tokens of log p(token | prefix)  (chain rule, so a
sum and never a mean), pi_ref is a FROZEN copy of the starting policy, and prompt tokens,
padding and tool output contribute nothing.

Ref: Rafailov et al., https://arxiv.org/abs/2305.18290
"""
import torch
import torch.nn.functional as F


def sequence_logprobs(model, input_ids, target_ids, response_mask):
    """
    Sum of log p(target | prefix) over the positions where response_mask == 1.

    input_ids     (B, T) int   - what the model reads
    target_ids    (B, T) int   - input_ids shifted by one; -1 (or any negative) where ignored
    response_mask (B, T) {0,1} - 1 exactly on the response tokens that count

    Returns (B,) float. This is log pi(y|x), NOT a per-token average: DPO compares
    probabilities of whole sequences, and averaging would silently divide out length.
    """
    logits = model(input_ids)                                   # (B, T, V)
    # upcast reduced precision, but never DOWNcast (float64 callers rely on this)
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    logprobs = F.log_softmax(logits, dim=-1)                    # (B, T, V)
    safe = target_ids.clamp(min=0).unsqueeze(-1)                # never index with -1
    token_logprobs = logprobs.gather(-1, safe).squeeze(-1)      # (B, T)
    mask = response_mask.to(token_logprobs.dtype)
    return (token_logprobs * mask).sum(dim=-1)                  # (B,)


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    All four arguments are (B,) sequence log-probability SUMS. The two reference tensors
    must already be detached - the reference policy is frozen and never receives gradient.

    Returns (loss, metrics).
    """
    assert not ref_chosen_logps.requires_grad and not ref_rejected_logps.requires_grad, \
        "reference log-probs must be detached: pi_ref is frozen by definition"

    policy_logratio = policy_chosen_logps - policy_rejected_logps
    ref_logratio = ref_chosen_logps - ref_rejected_logps
    margin = policy_logratio - ref_logratio          # the "logits" of the preference classifier
    loss = -F.logsigmoid(beta * margin).mean()

    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    metrics = {
        "loss": loss.detach(),
        "chosen_reward": chosen_reward.mean(),
        "rejected_reward": rejected_reward.mean(),
        "reward_margin": (chosen_reward - rejected_reward).mean(),
        # fraction of pairs the implicit reward model already ranks correctly
        "preference_accuracy": (chosen_reward > rejected_reward).float().mean(),
    }
    return loss, metrics


def dpo_loss_from_batch(policy, reference, batch, beta=0.1):
    """
    batch: dict with chosen_input/chosen_target/chosen_mask and rejected_* of shape (B, T).
    Runs four forward passes (2 policy with grad, 2 reference without).
    """
    pc = sequence_logprobs(policy, batch["chosen_input"], batch["chosen_target"], batch["chosen_mask"])
    pr = sequence_logprobs(policy, batch["rejected_input"], batch["rejected_target"], batch["rejected_mask"])
    with torch.no_grad():
        rc = sequence_logprobs(reference, batch["chosen_input"], batch["chosen_target"], batch["chosen_mask"])
        rr = sequence_logprobs(reference, batch["rejected_input"], batch["rejected_target"], batch["rejected_mask"])
    return dpo_loss(pc, pr, rc, rr, beta=beta)
