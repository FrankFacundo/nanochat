"""
W4/B - Unit tests for nanochat/dpo.py, written before any DPO training run.

Covers exactly the checks the rubric names:
  hand-calculated batch match, swap test, neutral pair, frozen-reference gradients,
  masking test - plus length sensitivity and the beta limits.

Run: python -m pytest experiments/w4_dpo/test_dpo.py -v
"""
import math
import pytest
import torch
import torch.nn.functional as F

from nanochat.dpo import dpo_loss, sequence_logprobs, dpo_loss_from_batch


def L(*vals, grad=False):
    return torch.tensor(vals, dtype=torch.float64, requires_grad=grad)


# ------------------------------------------------------------------ 1
def test_matches_hand_calculation():
    """One pair, every number computed by hand with the standard library."""
    pc, pr, rc, rr, beta = -3.0, -5.0, -4.0, -4.5, 0.1
    # by hand:
    policy_logratio = pc - pr            # -3 - (-5)  = 2.0
    ref_logratio = rc - rr               # -4 - (-4.5) = 0.5
    margin = policy_logratio - ref_logratio                     # 1.5
    expected = -math.log(1.0 / (1.0 + math.exp(-beta * margin)))  # -log sigmoid(0.15)
    assert policy_logratio == 2.0 and ref_logratio == 0.5 and margin == 1.5
    assert expected == pytest.approx(0.6209570, abs=1e-6)

    loss, m = dpo_loss(L(pc, grad=True), L(pr), L(rc), L(rr), beta=beta)
    assert loss.item() == pytest.approx(expected, abs=1e-12)
    assert m["chosen_reward"].item() == pytest.approx(beta * (pc - rc))    # 0.1 * 1.0
    assert m["rejected_reward"].item() == pytest.approx(beta * (pr - rr))  # 0.1 * -0.5
    assert m["reward_margin"].item() == pytest.approx(beta * margin)       # 0.15
    assert m["preference_accuracy"].item() == 1.0


# ------------------------------------------------------------------ 2
def test_swap_test():
    """Swapping chosen and rejected must flip the sign of the margin, not just perturb it."""
    pc, pr, rc, rr = L(-3.0), L(-5.0), L(-4.0), L(-4.5)
    beta = 0.1
    a, ma = dpo_loss(pc, pr, rc, rr, beta=beta)
    b, mb = dpo_loss(pr, pc, rr, rc, beta=beta)
    assert ma["reward_margin"].item() == pytest.approx(-mb["reward_margin"].item())
    # -log sigmoid(z) + -log sigmoid(-z) = -log(sigmoid(z)*sigmoid(-z)); check via identity
    z = beta * 1.5
    assert a.item() == pytest.approx(-math.log(1 / (1 + math.exp(-z))))
    assert b.item() == pytest.approx(-math.log(1 / (1 + math.exp(z))))
    assert b.item() > a.item(), "the wrong ordering must cost more"
    assert ma["preference_accuracy"].item() == 1.0
    assert mb["preference_accuracy"].item() == 0.0


# ------------------------------------------------------------------ 3
def test_neutral_pair_gives_log2_and_zero_gradient_direction():
    """If the policy has not moved relative to the reference, the loss is exactly log 2."""
    p = L(-3.0, grad=True)
    loss, m = dpo_loss(p, L(-5.0, grad=True), L(-3.0), L(-5.0), beta=0.1)
    assert loss.item() == pytest.approx(math.log(2.0), abs=1e-12)
    assert m["reward_margin"].item() == pytest.approx(0.0)
    # the pair is "already tied": the implicit rewards are equal
    assert m["chosen_reward"].item() == pytest.approx(m["rejected_reward"].item())
    # gradient is non-zero (it is sigmoid(0)=1/2 scaled by beta) and pushes chosen UP
    loss.backward()
    assert p.grad.item() < 0, "increasing log p(chosen) must decrease the loss"
    assert abs(p.grad.item()) == pytest.approx(0.1 * 0.5, abs=1e-12)


# ------------------------------------------------------------------ 4
def test_reference_is_frozen():
    """Reference log-probs must carry no gradient, and passing an attached one must fail."""
    pc, pr = L(-3.0, grad=True), L(-5.0, grad=True)
    rc, rr = L(-4.0), L(-4.5)
    loss, _ = dpo_loss(pc, pr, rc, rr, beta=0.1)
    loss.backward()
    assert pc.grad is not None and pr.grad is not None
    assert rc.grad is None and rr.grad is None
    # gradients are equal and opposite: the loss depends only on the DIFFERENCE
    assert pc.grad.item() == pytest.approx(-pr.grad.item())
    with pytest.raises(AssertionError):
        dpo_loss(pc, pr, L(-4.0, grad=True), rr, beta=0.1)


# ------------------------------------------------------------------ 5
def test_masking_ignores_prompt_and_padding():
    """
    Two sequences with IDENTICAL response tokens but different prompts and padding must
    produce identical response log-probs.
    """
    torch.manual_seed(0)
    B, T, V = 2, 8, 16

    class FixedLM(torch.nn.Module):
        """A deterministic 'model': logits depend only on the input token id."""
        def __init__(self):
            super().__init__()
            self.table = torch.randn(V, V, dtype=torch.float64)
        def forward(self, idx):
            return self.table[idx]

    m = FixedLM()
    resp = [5, 6, 7]
    a_in = torch.tensor([[1, 2, 3] + resp + [0, 0], [9, 9, 9] + resp + [4, 4]])
    a_tg = torch.tensor([[2, 3, 5, 6, 7, 8, -1, -1], [9, 9, 5, 6, 7, 8, 4, 4]])
    # mask the positions whose (input, target) pair is the response: inputs 5,6,7 -> targets 6,7,8
    # in BOTH rows. Positions 0-2 read different prompts and positions 6-7 differ in padding.
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0]])
    lp = sequence_logprobs(m, a_in, a_tg, mask)
    assert lp[0].item() == pytest.approx(lp[1].item(), abs=1e-12), \
        "different prompt/padding changed the response log-prob"

    # a mask of all zeros gives exactly 0.0 (an empty product of probabilities)
    zero = sequence_logprobs(m, a_in, a_tg, torch.zeros_like(mask))
    assert torch.allclose(zero, torch.zeros(B, dtype=torch.float64))
    # negative targets are never used as indices (would raise or silently wrap otherwise)
    assert torch.isfinite(lp).all()


# ------------------------------------------------------------------ 6
def test_logprobs_are_sums_not_means_and_are_length_sensitive():
    torch.manual_seed(0)
    V = 16

    class FixedLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.zeros(V, V, dtype=torch.float64)  # uniform -> log p = -log V
        def forward(self, idx):
            return self.table[idx]

    m = FixedLM()
    idx = torch.tensor([[1, 2, 3, 4, 5, 6]])
    tgt = torch.tensor([[2, 3, 4, 5, 6, 7]])
    for n in (1, 3, 6):
        mask = torch.tensor([[1] * n + [0] * (6 - n)])
        lp = sequence_logprobs(m, idx, tgt, mask)
        assert lp.item() == pytest.approx(-n * math.log(V), abs=1e-10), \
            "log pi(y|x) must scale linearly with response length (it is a sum)"


# ------------------------------------------------------------------ 7
def test_beta_limits():
    pc, pr, rc, rr = L(-3.0), L(-5.0), L(-4.0), L(-4.5)   # margin = +1.5
    small, _ = dpo_loss(pc, pr, rc, rr, beta=1e-8)
    assert small.item() == pytest.approx(math.log(2.0), abs=1e-7), \
        "beta -> 0 makes every pair look tied: loss -> log 2, gradient -> 0"
    big, _ = dpo_loss(pc, pr, rc, rr, beta=100.0)
    assert big.item() < 1e-30, "beta large saturates a correctly-ordered pair"
    wrong, _ = dpo_loss(pr, pc, rr, rc, beta=100.0)   # margin = -1.5
    assert wrong.item() == pytest.approx(100.0 * 1.5, rel=1e-6), \
        "for a large negative margin, -log sigmoid(z) -> -z, i.e. loss grows linearly in beta"


# ------------------------------------------------------------------ 8
def test_end_to_end_on_a_real_gpt():
    """The full four-forward-pass path on an actual (tiny) nanochat GPT."""
    from nanochat.gpt import GPT, GPTConfig
    cfg = GPTConfig(sequence_len=32, vocab_size=256, n_layer=2, n_head=2, n_kv_head=2,
                    n_embd=64, window_pattern="L")
    torch.manual_seed(0)
    policy = GPT(cfg); policy.init_weights()
    torch.manual_seed(0)
    reference = GPT(cfg); reference.init_weights()
    for p in reference.parameters():
        p.requires_grad_(False)

    B, T = 2, 16
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 256, (B, T + 1), generator=g)
    mask = torch.zeros(B, T, dtype=torch.long); mask[:, 8:] = 1
    batch = dict(chosen_input=ids[:, :-1], chosen_target=ids[:, 1:], chosen_mask=mask,
                 rejected_input=ids[:, :-1].flip(0), rejected_target=ids[:, 1:].flip(0), rejected_mask=mask)

    loss, m = dpo_loss_from_batch(policy, reference, batch, beta=0.1)
    # policy == reference at init, so every pair is exactly tied
    assert loss.item() == pytest.approx(math.log(2.0), abs=1e-4)
    assert abs(m["reward_margin"].item()) < 1e-4
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in policy.parameters()), "policy received no gradient"
    assert all(p.grad is None for p in reference.parameters()), "reference must stay frozen"
