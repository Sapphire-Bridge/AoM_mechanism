import numpy as np
import torch
import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from aom.mechanistic.attention_recorder import AttentionPatternRecorder
from aom.mechanistic.induction import (
    bootstrap_ci_matrix,
    calculate_first_half_mass,
    calculate_induction_control,
    calculate_induction_diag_fraction,
    calculate_induction_score,
)
from aom.mechanistic.synthetic import generate_induction_batch


def _make_attn_tensor(base_len: int) -> torch.Tensor:
    # Build a simple attention tensor with perfect induction diagonal at offset +1.
    seq_len = base_len * 2
    attn = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    for q in range(seq_len):
        if base_len <= q <= (2 * base_len - 2):
            k = q - base_len + 1
            attn[0, 0, q, k] = 1.0
        else:
            attn[0, 0, q, q] = 1.0
    return attn


def test_induction_score_offset_diagonal():
    base_len = 4
    attn = _make_attn_tensor(base_len)
    score = calculate_induction_score(attn, base_len=base_len)
    baseline = calculate_induction_control(attn, base_len=base_len, mode="offset0")
    assert score.shape == (1, 1)
    assert baseline.shape == (1, 1)
    assert float(score.item()) == 1.0
    assert float(baseline.item()) == 0.0


def test_first_half_mass_and_fraction():
    base_len = 4
    attn = _make_attn_tensor(base_len)
    mass = calculate_first_half_mass(attn, base_len=base_len)
    frac = calculate_induction_diag_fraction(attn, base_len=base_len)
    assert float(mass.item()) == 1.0
    assert float(frac.item()) == 1.0


def test_generate_induction_batch_deterministic():
    batch1 = generate_induction_batch(vocab_size=50, base_len=8, repeats=2, batch_size=3, seed=123)
    batch2 = generate_induction_batch(vocab_size=50, base_len=8, repeats=2, batch_size=3, seed=123)
    assert torch.equal(batch1, batch2)


def test_recorder_gpt2_stats_shape():
    config = GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=50, n_positions=32)
    model = GPT2LMHeadModel(config)
    model.eval()

    base_len = 6
    input_ids = torch.randint(0, 50, (2, base_len * 2), dtype=torch.long)

    recorder = AttentionPatternRecorder(model, layers=[0], base_len=base_len, repeats=2, baseline_mode="offset0")
    with torch.no_grad(), recorder:
        _ = model(input_ids=input_ids, use_cache=False)

    samples = recorder.get_samples(0, "repeat")
    assert len(samples) == 1
    s0 = samples[0]
    assert s0.shape == (2, config.n_head)
    assert torch.isfinite(s0).all()


def test_recorder_noop_logits():
    config = GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=50, n_positions=32)
    model = GPT2LMHeadModel(config)
    model.eval()

    base_len = 5
    input_ids = torch.randint(0, 50, (1, base_len * 2), dtype=torch.long)
    base_logits = model(input_ids=input_ids, use_cache=False).logits

    recorder = AttentionPatternRecorder(model, layers=[0], base_len=base_len, repeats=2, baseline_mode="offset0")
    with torch.no_grad(), recorder:
        rec_logits = model(input_ids=input_ids, use_cache=False).logits

    assert torch.allclose(base_logits, rec_logits, atol=1e-5)


def _make_llama_tiny():
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except Exception:
        pytest.skip("LlamaForCausalLM not available")
    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=50,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def test_recorder_llama_stats_shape():
    model = _make_llama_tiny()
    base_len = 5
    input_ids = torch.randint(0, 50, (2, base_len * 2), dtype=torch.long)

    recorder = AttentionPatternRecorder(model, layers=[0], base_len=base_len, repeats=2, baseline_mode="offset0")
    with torch.no_grad(), recorder:
        _ = model(input_ids=input_ids, use_cache=False)

    samples = recorder.get_samples(0, "repeat")
    assert len(samples) == 1
    s0 = samples[0]
    assert s0.shape[0] == 2
    assert s0.shape[1] == model.config.num_attention_heads
    assert torch.isfinite(s0).all()


def test_recorder_llama_noop_logits():
    model = _make_llama_tiny()
    base_len = 4
    input_ids = torch.randint(0, 50, (1, base_len * 2), dtype=torch.long)
    base_logits = model(input_ids=input_ids, use_cache=False).logits

    recorder = AttentionPatternRecorder(model, layers=[0], base_len=base_len, repeats=2, baseline_mode="offset0")
    with torch.no_grad(), recorder:
        rec_logits = model(input_ids=input_ids, use_cache=False).logits

    assert torch.allclose(base_logits, rec_logits, atol=1e-5)


def test_shuffle_baseline_pairs():
    config = GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=50, n_positions=32)
    model = GPT2LMHeadModel(config)
    model.eval()

    base_len = 6
    input_ids = torch.randint(0, 50, (2, base_len * 2), dtype=torch.long)

    recorder = AttentionPatternRecorder(model, layers=[0], base_len=base_len, repeats=2, baseline_mode="shuffle")
    with torch.no_grad(), recorder:
        recorder.set_mode("repeat")
        _ = model(input_ids=input_ids, use_cache=False)
        recorder.set_mode("control")
        _ = model(input_ids=input_ids, use_cache=False)

    rep = recorder.get_samples(0, "repeat")
    ctl = recorder.get_samples(0, "control")
    assert len(rep) == len(ctl) == 1
    assert rep[0].shape == ctl[0].shape
    adv = rep[0] - ctl[0]
    assert torch.isfinite(adv).all()


def test_offset0_baseline_cropping_length():
    base_len = 3
    seq_len = base_len * 2
    attn = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    # Set offset0 diagonal values for queries 3,4,5 to [1,0,1].
    attn[0, 0, 3, 0] = 1.0
    attn[0, 0, 4, 1] = 0.0
    attn[0, 0, 5, 2] = 1.0
    # Fill remaining queries with self-attention to keep row sums = 1.
    for q in range(0, seq_len):
        if attn[0, 0, q, :].sum().item() == 0.0:
            attn[0, 0, q, q] = 1.0

    baseline = calculate_induction_control(attn, base_len=base_len, mode="offset0")
    # Cropped baseline uses only queries 3 and 4 => mean = (1 + 0) / 2.
    assert float(baseline.item()) == 0.5


def test_bootstrap_ci_matrix_empty_returns_nan():
    samples = np.empty((0, 3), dtype=float)
    mean, lo, hi = bootstrap_ci_matrix(samples, n_bootstrap=10, ci=0.95, seed=0)
    assert mean.shape == (3,)
    assert lo.shape == (3,)
    assert hi.shape == (3,)
    assert np.isnan(mean).all()
    assert np.isnan(lo).all()
    assert np.isnan(hi).all()


def test_bootstrap_ci_matrix_non_finite_returns_nan():
    samples = np.asarray([[0.0, np.nan, 1.0]], dtype=float)
    mean, lo, hi = bootstrap_ci_matrix(samples, n_bootstrap=10, ci=0.95, seed=0)
    assert np.isnan(mean).all()
    assert np.isnan(lo).all()
    assert np.isnan(hi).all()
