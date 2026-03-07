import torch
import pytest

from aom.mechanistic.backends.base import BackendLoadResult
from aom.mechanistic.backends.hf_eager import HFEagerBackend


def _make_loaded_gpt2_tiny() -> BackendLoadResult:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=50, n_positions=32)
    model = GPT2LMHeadModel(config)
    model.eval()
    param0 = next(model.parameters())

    return BackendLoadResult(
        model=model,
        tokenizer=None,
        architecture="gpt2",
        device=torch.device("cpu"),
        n_layers=int(config.n_layer),
        vocab_size=int(config.vocab_size),
        exclude_token_ids=frozenset(),
        max_seq_len=int(config.n_positions),
        model_param_dtype=str(param0.dtype),
        model_param_device=str(param0.device),
        num_attention_heads=int(config.n_head),
        num_key_value_heads=None,
        backend_version="test",
    )


def test_hf_backend_run_batches_shuffle_collects_samples():
    loaded = _make_loaded_gpt2_tiny()
    backend = HFEagerBackend()
    result = backend.run_batches(
        loaded=loaded,
        layers=[0],
        base_len=6,
        repeats=2,
        batch_size=2,
        n_batches=1,
        seed=0,
        baseline="shuffle",
        debug_store_attn=False,
        validate_attn="none",
        tl_crosscheck=False,
        bootstrap_n=50,
        bootstrap_seed=0,
        ci=0.95,
    )
    rep = result.collector.get_samples(0, "repeat")
    ctl = result.collector.get_samples(0, "control")
    assert len(rep) == len(ctl) == 1
    assert rep[0].shape == ctl[0].shape == (2, 2)
    assert torch.isfinite(rep[0]).all()
    assert torch.isfinite(ctl[0]).all()


def test_hf_backend_run_batches_offset0_populates_control():
    loaded = _make_loaded_gpt2_tiny()
    backend = HFEagerBackend()
    result = backend.run_batches(
        loaded=loaded,
        layers=[0],
        base_len=6,
        repeats=2,
        batch_size=2,
        n_batches=1,
        seed=0,
        baseline="offset0",
        debug_store_attn=False,
        validate_attn="none",
        tl_crosscheck=False,
        bootstrap_n=50,
        bootstrap_seed=0,
        ci=0.95,
    )
    rep = result.collector.get_samples(0, "repeat")
    ctl = result.collector.get_samples(0, "control")
    assert len(rep) == len(ctl) == 1
    assert rep[0].shape == ctl[0].shape == (2, 2)


def test_transformer_lens_backend_dependency_guard():
    import importlib.util

    if importlib.util.find_spec("transformer_lens") is not None:
        pytest.skip("transformer_lens installed; dependency guard not applicable")

    from aom.mechanistic.backends.transformer_lens import TransformerLensBackend

    backend = TransformerLensBackend()
    with pytest.raises(RuntimeError):
        _ = backend.load(
            model_name_or_path="gpt2",
            device=torch.device("cpu"),
            torch_dtype=None,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="eager",
        )
