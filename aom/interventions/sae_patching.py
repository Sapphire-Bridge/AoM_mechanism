from __future__ import annotations

from typing import Optional

import torch
from transformers import PreTrainedModel

from aom.interventions.activation_patching import PatchSite, PatchSpanSite, get_decoder_blocks
from aom.interventions.sae_adapter import (
    FeaturePolicy,
    IdentityFeaturePolicy,
    SAEHookState,
    SAEInputTransform,
    SAEInterventionHook,
    SAEProtocol,
    SAEPatchConfig,
)


@torch.no_grad()
def forward_with_sae_feature_patching(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    site: PatchSite,
    sae: SAEProtocol,
    policy: FeaturePolicy,
    transform: Optional[SAEInputTransform] = None,
    config: Optional[SAEPatchConfig] = None,
    state: Optional[SAEHookState] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return forward_with_sae_feature_patching_span(
        model,
        input_ids=input_ids,
        site=PatchSpanSite(layer=int(site.layer), token_indices=(int(site.token_index),)),
        sae=sae,
        policy=policy,
        transform=transform,
        config=config,
        state=state,
        attention_mask=attention_mask,
    )


@torch.no_grad()
def forward_with_sae_feature_patching_span(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    site: PatchSpanSite,
    sae: SAEProtocol,
    policy: FeaturePolicy,
    transform: Optional[SAEInputTransform] = None,
    config: Optional[SAEPatchConfig] = None,
    state: Optional[SAEHookState] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    hook, blocks = _make_sae_patch_hook(
        model=model,
        input_ids=input_ids,
        site=site,
        sae=sae,
        policy=policy,
        transform=transform,
        config=config,
        state=state,
        attention_mask=attention_mask,
    )
    handle = blocks[site.layer].register_forward_hook(hook)
    try:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.logits  # type: ignore[return-value]
    finally:
        handle.remove()


def _make_sae_patch_hook(
    *,
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    site: PatchSpanSite,
    sae: SAEProtocol,
    policy: FeaturePolicy,
    transform: Optional[SAEInputTransform],
    config: Optional[SAEPatchConfig],
    state: Optional[SAEHookState],
    attention_mask: Optional[torch.Tensor],
):
    blocks = get_decoder_blocks(model)
    n_layers = len(blocks)
    if site.layer < 0 or site.layer >= n_layers:
        raise ValueError(f"layer {site.layer} out of range [0, {n_layers})")
    if not site.token_indices:
        raise ValueError("token_indices must be non-empty")

    seq_len = int(input_ids.size(1))
    for token_idx in site.token_indices:
        if token_idx < 0 or token_idx >= seq_len:
            raise ValueError(f"token index {token_idx} out of range [0, {seq_len})")

    token_mask = None
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape")
        token_mask = attention_mask.to(dtype=torch.bool)

    hook = SAEInterventionHook(
        sae=sae,
        token_indices=tuple(int(x) for x in site.token_indices),
        policy=policy,
        transform=transform,
        config=config,
        token_mask=token_mask,
        state=state,
    )
    return hook, blocks


@torch.no_grad()
def prefill_with_sae_feature_patching_span(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    site: PatchSpanSite,
    sae: SAEProtocol,
    policy: FeaturePolicy,
    transform: Optional[SAEInputTransform] = None,
    config: Optional[SAEPatchConfig] = None,
    state: Optional[SAEHookState] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    """
    Run a patched prompt prefill and return full model output (including cache).

    This is used to reuse a patched prefix cache across many continuation scorings.
    """
    hook, blocks = _make_sae_patch_hook(
        model=model,
        input_ids=input_ids,
        site=site,
        sae=sae,
        policy=policy,
        transform=transform,
        config=config,
        state=state,
        attention_mask=attention_mask,
    )
    handle = blocks[site.layer].register_forward_hook(hook)
    try:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    finally:
        handle.remove()


@torch.no_grad()
def forward_with_sae_feature_roundtrip_layer(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    layer: int,
    sae: SAEProtocol,
    transform: Optional[SAEInputTransform] = None,
    config: Optional[SAEPatchConfig] = None,
    state: Optional[SAEHookState] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Full-sequence SAE roundtrip replacement at a given layer.

    This uses `roundtrip_replace=True` to return the SAE reconstruction (in model space),
    rather than delta-injection (which is a no-op under identity policies).
    """
    blocks = get_decoder_blocks(model)
    n_layers = len(blocks)
    if int(layer) < 0 or int(layer) >= n_layers:
        raise ValueError(f"layer {int(layer)} out of range [0, {n_layers})")

    token_mask = None
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape")
        token_mask = attention_mask.to(dtype=torch.bool)

    cfg = config or SAEPatchConfig()
    if not bool(cfg.roundtrip_replace):
        cfg = SAEPatchConfig(
            decode_strategy=cfg.decode_strategy,
            check_normalization_stateful=cfg.check_normalization_stateful,
            random_seed=cfg.random_seed,
            eps_active=cfg.eps_active,
            dtype_policy=cfg.dtype_policy,
            roundtrip_replace=True,
        )

    hook = SAEInterventionHook(
        sae=sae,
        token_indices=None,  # apply to full sequence
        policy=IdentityFeaturePolicy(),
        transform=transform,
        config=cfg,
        token_mask=token_mask,
        state=state,
    )
    handle = blocks[int(layer)].register_forward_hook(hook)
    try:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.logits  # type: ignore[return-value]
    finally:
        handle.remove()


@torch.no_grad()
def forward_with_sae_feature_patching_layer(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    layer: int,
    sae: SAEProtocol,
    policy: FeaturePolicy,
    transform: Optional[SAEInputTransform] = None,
    config: Optional[SAEPatchConfig] = None,
    state: Optional[SAEHookState] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply SAE feature policy across the full sequence at the given layer (delta-injection path)."""
    blocks = get_decoder_blocks(model)
    n_layers = len(blocks)
    if int(layer) < 0 or int(layer) >= n_layers:
        raise ValueError(f"layer {int(layer)} out of range [0, {n_layers})")

    token_mask = None
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape")
        token_mask = attention_mask.to(dtype=torch.bool)

    hook = SAEInterventionHook(
        sae=sae,
        token_indices=None,  # full sequence
        policy=policy,
        transform=transform,
        config=config,
        token_mask=token_mask,
        state=state,
    )
    handle = blocks[int(layer)].register_forward_hook(hook)
    try:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.logits  # type: ignore[return-value]
    finally:
        handle.remove()
