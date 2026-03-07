from __future__ import annotations

import importlib.util

import pytest
import torch

from aom.mechanistic.backends.hooks import HookSpec
from aom.mechanistic.backends.transformer_lens import (
    TransformerLensHookableBackend,
    make_patch_pattern_hook,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("transformer_lens") is None,
    reason="transformer_lens not installed",
)


def _tiny_tl_model():
    from transformer_lens import HookedTransformer, HookedTransformerConfig  # type: ignore[import-not-found]

    cfg = HookedTransformerConfig(
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_head=16,
        d_mlp=64,
        n_ctx=32,
        d_vocab=128,
        act_fn="relu",
    )
    model = HookedTransformer(cfg)
    model.eval()
    return model


def test_tlens_hookable_run_with_cache_shapes():
    model = _tiny_tl_model()
    backend = TransformerLensHookableBackend(model=model, tokenizer=getattr(model, "tokenizer", None))
    tokens = torch.randint(0, int(model.cfg.d_vocab), (1, 8), dtype=torch.long)

    out = backend.run_with_cache(
        prompt="",
        capture=["pattern", "value", "resid_post", "resid_pre", "resid_final"],
        tokens=tokens,
    )

    assert isinstance(out.logits, torch.Tensor)
    assert out.logits.shape[:2] == tokens.shape
    assert "pattern.0" in out.cache and out.cache["pattern.0"].ndim == 4
    assert "value.0" in out.cache and out.cache["value.0"].ndim == 4
    assert "resid_post.1" in out.cache and out.cache["resid_post.1"].ndim == 3
    assert "resid_final" in out.cache and out.cache["resid_final"].ndim == 3


def test_tlens_pattern_hook_identity_and_uniform_change():
    model = _tiny_tl_model()
    backend = TransformerLensHookableBackend(model=model, tokenizer=getattr(model, "tokenizer", None))
    tokens = torch.randint(0, int(model.cfg.d_vocab), (1, 8), dtype=torch.long)

    cached = backend.run_with_cache(prompt="", capture=["pattern"], tokens=tokens)
    base_logits = cached.logits.detach()

    identity_hook = make_patch_pattern_hook(
        layer=0,
        head=0,
        source_cache=cached.cache,
        q_pos=0,
    )
    id_logits = backend.run_with_hooks(prompt="", hooks=[identity_hook], tokens=tokens).detach()
    assert torch.allclose(id_logits, base_logits, atol=1e-6, rtol=1e-6)

    def _uniform_pattern(pattern: torch.Tensor, _hook) -> torch.Tensor:
        patched = pattern.clone()
        patched[:, 0, :, :] = 1.0 / float(pattern.size(-1))
        return patched

    uniform_hook = HookSpec(name="pattern.0", fn=_uniform_pattern)
    uniform_logits = backend.run_with_hooks(prompt="", hooks=[uniform_hook], tokens=tokens).detach()
    assert not torch.allclose(uniform_logits, base_logits, atol=1e-7, rtol=1e-7)
