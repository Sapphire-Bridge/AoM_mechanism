import pytest
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from aom.interventions.activation_patching import PatchSpanSite
from aom.interventions.sae_adapter import (
    IdentityFeaturePolicy,
    SAEHookState,
    SAEInputTransform,
    SAEInterventionHook,
    SAEPatchConfig,
    RelativeThresholdPolicy,
)
from aom.interventions.sae_patching import forward_with_sae_feature_patching_span


class IdentitySAE(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self._d_in = int(d_in)
        self._d_sae = int(d_in)
        self.W_dec = nn.Parameter(torch.eye(self._d_sae))

    @property
    def d_in(self) -> int:
        return self._d_in

    @property
    def d_sae(self) -> int:
        return self._d_sae

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features


def test_sae_intervention_hook_identity_is_noop():
    sae = IdentitySAE(d_in=8)
    hook = SAEInterventionHook(
        sae=sae,
        token_indices=(1, 3),
        policy=IdentityFeaturePolicy(),
    )
    hidden = torch.randn(2, 5, 8)
    patched = hook(nn.Identity(), (), hidden)
    assert torch.allclose(hidden, patched, atol=0.0)


@pytest.mark.parametrize("scale,shift", [(2.7, -0.3), (0.1, 5.0)])
def test_sae_input_transform_inverse_forward_roundtrip(scale: float, shift: float):
    t = SAEInputTransform(scale=float(scale), shift=float(shift))
    h = torch.randn(2, 5, 8, dtype=torch.float32)
    h2 = t.inverse(t.forward(h))
    assert torch.allclose(h2, h, atol=1e-5)


def test_sae_intervention_hook_inverse_delta_scaling_zeroes_site_token():
    sae = IdentitySAE(d_in=8)
    policy = RelativeThresholdPolicy(threshold=1e9, scale_mode="max")
    hook = SAEInterventionHook(
        sae=sae,
        token_indices=(2,),
        policy=policy,
        transform=SAEInputTransform(scale=2.0),
    )
    hidden = torch.randn(2, 5, 8)
    patched = hook(nn.Identity(), (), hidden)
    assert torch.allclose(patched[:, 2, :], torch.zeros_like(patched[:, 2, :]), atol=1e-6)
    assert torch.allclose(patched[:, [0, 1, 3, 4], :], hidden[:, [0, 1, 3, 4], :], atol=0.0)


def test_sae_intervention_hook_delta_1decode_matches_safe_2decode_for_linear_sae():
    sae = IdentitySAE(d_in=8)
    policy = RelativeThresholdPolicy(threshold=1e9, scale_mode="max")
    hidden = torch.randn(2, 5, 8)

    safe_hook = SAEInterventionHook(
        sae=sae,
        token_indices=(2,),
        policy=policy,
        transform=SAEInputTransform(scale=2.0),
        config=SAEPatchConfig(decode_strategy="safe_2decode"),
    )
    fast_hook = SAEInterventionHook(
        sae=sae,
        token_indices=(2,),
        policy=policy,
        transform=SAEInputTransform(scale=2.0),
        config=SAEPatchConfig(decode_strategy="delta_1decode"),
    )

    safe_out = safe_hook(nn.Identity(), (), hidden)
    fast_out = fast_hook(nn.Identity(), (), hidden)
    assert torch.allclose(safe_out, fast_out, atol=1e-6)


def test_forward_with_sae_feature_patching_span_identity_is_noop():
    config = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=100, n_positions=32)
    model = GPT2LMHeadModel(config)
    model.eval()

    sae = IdentitySAE(d_in=32)
    input_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)

    base_logits = model(input_ids, use_cache=False).logits
    state = SAEHookState()
    patched_logits = forward_with_sae_feature_patching_span(
        model,
        input_ids=input_ids,
        site=PatchSpanSite(layer=0, token_indices=(2,)),
        sae=sae,
        policy=IdentityFeaturePolicy(),
        state=state,
    )

    assert torch.allclose(base_logits, patched_logits, atol=1e-6)
    assert state.total_active >= 0.0


def test_forward_with_sae_feature_patching_span_roundtrip_replace_identity_is_noop():
    config = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=100, n_positions=32)
    model = GPT2LMHeadModel(config)
    model.eval()

    sae = IdentitySAE(d_in=32)
    input_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)

    base_logits = model(input_ids, use_cache=False).logits
    patched_logits = forward_with_sae_feature_patching_span(
        model,
        input_ids=input_ids,
        site=PatchSpanSite(layer=0, token_indices=(2,)),
        sae=sae,
        policy=IdentityFeaturePolicy(),
        config=SAEPatchConfig(roundtrip_replace=True),
    )

    assert torch.allclose(base_logits, patched_logits, atol=1e-6)
