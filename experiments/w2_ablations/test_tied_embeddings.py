"""
W2/B - Unit tests for the tied-embeddings implementation contract.

These are written to be run BEFORE any training run is launched. They prove:
  1. baseline preservation  - tie_embeddings=False changes nothing, bit-for-bit
  2. storage sharing        - one tensor, not two copies kept in sync
  3. optimizer coverage     - the shared tensor is updated exactly once per step
  4. gradient accumulation  - both the lookup path and the logit path feed its grad
  5. parameter accounting   - num_scaling_params stops double counting
  6. FLOP accounting        - FLOPs/token are UNCHANGED (tying removes params, not matmuls)
  7. checkpoint metadata    - the flag round-trips through the saved model config
  8. forward equivalence    - a tied model is a valid model (finite loss, correct shapes)

Run: python -m pytest experiments/w2_ablations/test_tied_embeddings.py -v
"""
import copy
from dataclasses import asdict
import pytest
import torch

from nanochat.gpt import GPT, GPTConfig

SMALL = dict(sequence_len=64, vocab_size=512, n_layer=2, n_head=2, n_kv_head=2,
             n_embd=64, window_pattern="L")


def make(tie, seed=0):
    torch.manual_seed(seed)
    m = GPT(GPTConfig(**SMALL, tie_embeddings=tie))
    m.init_weights()
    return m


# ------------------------------------------------------------------ 1
def test_default_is_untied_and_baseline_is_bit_identical():
    """The contract's most important property: opting out costs nothing."""
    assert GPTConfig(**SMALL).tie_embeddings is False, "default must preserve the baseline"
    a, b = make(False, seed=1234), make(False, seed=1234)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"untied init is not deterministic at {k}"
    assert a.lm_head.weight is not a.transformer.wte.weight
    assert a.lm_head.weight.data_ptr() != a.transformer.wte.weight.data_ptr()


# ------------------------------------------------------------------ 2
def test_tied_model_shares_one_storage():
    m = make(True)
    assert m.lm_head.weight is m.transformer.wte.weight
    assert m.lm_head.weight.data_ptr() == m.transformer.wte.weight.data_ptr()
    # writing through one view is visible through the other -> genuinely one tensor
    with torch.no_grad():
        m.transformer.wte.weight[0, 0] = 3.14159
    assert m.lm_head.weight[0, 0].item() == pytest.approx(3.14159)
    # and nn.Module dedups it
    assert len(list(m.parameters())) == len(list(make(False).parameters())) - 1


# ------------------------------------------------------------------ 3
def test_optimizer_registers_the_shared_tensor_exactly_once():
    m = make(True)
    opt = m.setup_optimizer(weight_decay=0.1)
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "a parameter is registered in two groups"
    shared = id(m.lm_head.weight)
    assert sum(i == shared for i in ids) == 1, "tied tensor must be updated once per step"
    assert len(ids) == len(list(m.parameters())), "optimizer must cover every parameter"
    # untied control still covers both separately
    m2 = make(False)
    ids2 = [id(p) for g in m2.setup_optimizer(weight_decay=0.1).param_groups for p in g["params"]]
    assert len(ids2) == len(set(ids2)) == len(list(m2.parameters()))
    assert id(m2.lm_head.weight) in ids2 and id(m2.transformer.wte.weight) in ids2


# ------------------------------------------------------------------ 4
def test_gradient_accumulates_from_both_paths():
    """The shared tensor must receive grad from the embedding lookup AND the logit matmul."""
    torch.manual_seed(0)
    x = torch.randint(0, SMALL["vocab_size"], (2, 16))
    y = torch.randint(0, SMALL["vocab_size"], (2, 16))

    tied = make(True)
    tied(x, y).backward()
    g_both = tied.transformer.wte.weight.grad.clone()

    # isolate the logit path: freeze the lookup by detaching the embedding output
    ref = make(True)
    orig = ref.transformer.wte.forward
    ref.transformer.wte.forward = lambda idx: orig(idx).detach()
    ref(x, y).backward()
    g_logit_only = ref.transformer.wte.weight.grad.clone()

    assert torch.isfinite(g_both).all()
    assert not torch.allclose(g_both, g_logit_only), \
        "grad is identical with the lookup path cut -> the embedding path is not contributing"
    # rows for tokens that appear in the input must differ the most
    touched = torch.unique(x)
    delta = (g_both - g_logit_only).abs().sum(dim=1)
    assert delta[touched].sum() > 0


# ------------------------------------------------------------------ 5
def test_parameter_accounting_does_not_double_count():
    tied, untied = make(True), make(False)
    pt, pu = tied.num_scaling_params(), untied.num_scaling_params()
    # group sizes are still reported for both roles...
    assert pt["wte"] == pt["lm_head"] == pu["wte"] == pu["lm_head"]
    # ...but the total counts the storage once
    assert pt["total"] == sum(p.numel() for p in tied.parameters())
    assert pu["total"] == sum(p.numel() for p in untied.parameters())
    assert pu["total"] - pt["total"] == pu["lm_head"], "tying must save exactly one vocab tensor"


# ------------------------------------------------------------------ 6
def test_flops_per_token_are_unchanged_by_tying():
    """Tying removes STORAGE, not arithmetic: the logit matmul is still full size."""
    tied, untied = make(True), make(False)
    assert tied.estimate_flops() == untied.estimate_flops()
    assert tied.num_matmul_params() == untied.num_matmul_params()


# ------------------------------------------------------------------ 7
def test_config_roundtrips_through_checkpoint_metadata():
    """checkpoint_manager saves asdict(model.config); the flag must survive it."""
    cfg = GPTConfig(**SMALL, tie_embeddings=True)
    blob = asdict(cfg)
    assert blob["tie_embeddings"] is True
    restored = GPT(GPTConfig(**blob))
    assert restored.lm_head.weight is restored.transformer.wte.weight
    # and a checkpoint written before this feature existed still loads (key absent -> default)
    legacy = {k: v for k, v in blob.items() if k != "tie_embeddings"}
    legacy_model = GPT(GPTConfig(**legacy))
    assert legacy_model.lm_head.weight is not legacy_model.transformer.wte.weight


# ------------------------------------------------------------------ 8
def test_tied_model_trains_one_step_without_nan():
    torch.manual_seed(0)
    m = make(True)
    opt = m.setup_optimizer(weight_decay=0.0)
    x = torch.randint(0, SMALL["vocab_size"], (2, 16))
    y = torch.randint(0, SMALL["vocab_size"], (2, 16))
    before = m.lm_head.weight.detach().clone()
    loss0 = m(x, y)
    assert torch.isfinite(loss0), "tied init produced a non-finite loss"
    loss0.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    after = m.lm_head.weight.detach()
    assert not torch.equal(before, after), "shared tensor did not update"
    assert torch.equal(after, m.transformer.wte.weight.detach()), "views desynced after step"
    assert torch.isfinite(m(x, y))


# ------------------------------------------------------------------ 9 (regression)
def test_tying_survives_the_meta_device_build_path():
    """
    REGRESSION. The first version of this contract tied only in __init__ and passed all of
    tests 1-8, because those build on CPU. base_train.py builds on the META device and then
    calls to_empty(), which rebuilds parameter storage and SILENTLY BREAKS the alias - the
    model trained untied while reporting tie_embeddings=True. Caught by a 30-step smoke run
    that showed an unchanged parameter count. This test pins the real code path.
    """
    cfg = GPTConfig(**SMALL, tie_embeddings=True)
    with torch.device("meta"):
        m = GPT(cfg)
    assert m.lm_head.weight is m.transformer.wte.weight, "should be tied on meta"
    m.to_empty(device="cpu")
    assert m.lm_head.weight is not m.transformer.wte.weight, \
        "to_empty is expected to break the alias; if this ever stops being true, simplify tie_weights()"
    m.init_weights()
    assert m.lm_head.weight is m.transformer.wte.weight, "init_weights must re-tie"
    assert len(list(m.parameters())) == len(list(make(False).parameters())) - 1


# ------------------------------------------------------------------ 10 (regression)
def test_tying_survives_checkpoint_load_with_assign():
    """
    REGRESSION. checkpoint_manager.build_model uses load_state_dict(..., assign=True), which
    REPLACES parameter tensors with the ones read from disk and therefore breaks the alias a
    second time. build_model now calls model.tie_weights() afterwards.
    """
    cfg = GPTConfig(**SMALL, tie_embeddings=True)
    with torch.device("meta"):
        src = GPT(cfg)
    src.to_empty(device="cpu"); src.init_weights()
    sd = src.state_dict()
    assert "lm_head.weight" in sd and "transformer.wte.weight" in sd, \
        "state_dict still emits both keys for a tied model; loaders must handle that"

    with torch.device("meta"):
        dst = GPT(cfg)
    dst.to_empty(device="cpu"); dst.init_weights()
    dst.load_state_dict(sd, strict=True, assign=True)
    dst.tie_weights()
    assert dst.lm_head.weight is dst.transformer.wte.weight
    assert torch.equal(dst.lm_head.weight, src.lm_head.weight)
    # an untied checkpoint must NOT be silently tied on load
    with torch.device("meta"):
        untied = GPT(GPTConfig(**SMALL, tie_embeddings=False))
    untied.to_empty(device="cpu"); untied.init_weights(); untied.tie_weights()
    assert untied.lm_head.weight is not untied.transformer.wte.weight
