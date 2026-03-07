from __future__ import annotations

import importlib.util

import pytest
import torch

from aom.mechanistic.backends.transformer_lens import TransformerLensHookableBackend
from aom.mechanistic.logit_diff_decomposition import (
    decompose_logit_diff,
    logit_diff_direction,
    validate_decomposition,
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


def test_logit_diff_decomposition_sums_to_actual():
    model = _tiny_tl_model()
    backend = TransformerLensHookableBackend(model=model, tokenizer=getattr(model, "tokenizer", None))
    tokens = torch.randint(0, int(model.cfg.d_vocab), (1, 10), dtype=torch.long)

    cached = backend.run_with_cache(
        prompt="",
        capture=["embed", "pos_embed", "resid_final", "attn_out", "mlp_out"],
        tokens=tokens,
    )

    tok_a = int(tokens[0, -1].item())
    tok_b = int((tok_a + 1) % int(model.cfg.d_vocab))
    direction, bias_diff = logit_diff_direction(model, tok_a, tok_b)
    decomp = decompose_logit_diff(
        cached.cache,
        direction,
        pos=-1,
        mode="layer",
        final_norm=getattr(model, "ln_final", None),
        bias_diff=float(bias_diff),
    )

    actual = float((cached.logits[0, -1, tok_a] - cached.logits[0, -1, tok_b]).item())
    check = validate_decomposition(
        predicted_logit_diff=float(decomp.predicted_logit_diff),
        actual_logit_diff=float(actual),
        tol=1e-3,
    )
    assert check["within_tol"], f"decomposition error too high: {check}"
