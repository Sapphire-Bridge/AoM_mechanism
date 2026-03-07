"""Test Qwen architecture compatibility (offline, no downloads)."""

from __future__ import annotations

import pytest
import torch

from aom.interventions.activation_patching import (
    PatchSpanSite,
    detect_architecture,
    forward_with_patched_block_output_span,
    get_decoder_blocks,
    get_hidden_states,
    get_num_layers,
)


@pytest.fixture
def tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        use_cache=False,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def test_detect_qwen_architecture(tiny_qwen):
    arch = detect_architecture(tiny_qwen)
    assert arch == "qwen2"


def test_get_decoder_blocks_qwen(tiny_qwen):
    blocks = get_decoder_blocks(tiny_qwen)
    assert len(blocks) == 2
    assert blocks is tiny_qwen.model.layers


def test_get_num_layers_qwen(tiny_qwen):
    n = get_num_layers(tiny_qwen)
    assert n == 2


def test_get_hidden_states_qwen(tiny_qwen):
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    hs = get_hidden_states(tiny_qwen, input_ids)
    assert len(hs) == 3
    assert hs[0].shape == (1, 5, 64)


def test_patching_sham_noop_qwen(tiny_qwen):
    input_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)

    hs = get_hidden_states(tiny_qwen, input_ids)
    replacement = hs[1][0, 2, :].detach()

    base_logits = tiny_qwen(input_ids, use_cache=False).logits
    patched_logits = forward_with_patched_block_output_span(
        tiny_qwen,
        input_ids=input_ids,
        site=PatchSpanSite(layer=0, token_indices=(2,)),
        replacement=replacement,
    )

    assert torch.allclose(base_logits, patched_logits, atol=1e-5)


def test_patching_span_qwen(tiny_qwen):
    input_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)

    hs = get_hidden_states(tiny_qwen, input_ids)
    replacement = hs[1][0, [1, 3], :].detach()

    base_logits = tiny_qwen(input_ids, use_cache=False).logits
    patched_logits = forward_with_patched_block_output_span(
        tiny_qwen,
        input_ids=input_ids,
        site=PatchSpanSite(layer=0, token_indices=(1, 3)),
        replacement=replacement,
    )

    assert torch.allclose(base_logits, patched_logits, atol=1e-5)

