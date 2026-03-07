from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aom.data.schemas import DisambPair
from aom.interventions.sae_adapter import SAEInputTransform, SAEProtocol, SAEPatchConfig
from aom.interventions.sae_patching import forward_with_sae_feature_roundtrip_layer
from aom.metrics.disamb import score_labels_next_continuations
from aom.utils import get_logprob_computation_config


@dataclass(frozen=True)
class SterilityResult:
    baseline_acc: float
    roundtrip_acc: float
    delta: float
    passed: bool
    recon_mse: float
    mean_token_kl: float


def _encode(tokenizer: PreTrainedTokenizerBase, text: str, device: torch.device) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"].to(device)


@torch.no_grad()
def _score_labels_next_continuations_roundtrip(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    prompt: str,
    choices: dict,
    device: torch.device,
    sae: SAEProtocol,
    layer: int,
    transform: SAEInputTransform,
    normalize_by_length: bool,
) -> dict[str, float]:
    from aom.metrics.disamb import _logmeanexp  # local import to avoid widening public API

    prompt_ids = _encode(tokenizer, prompt, device=device)
    logprobs_dtype, strict_finite = get_logprob_computation_config()
    scores: dict[str, float] = {}

    for label, continuations in choices.items():
        vals: List[float] = []
        for cont in continuations:
            cont_ids = _encode(tokenizer, str(cont), device=device)
            full_ids = torch.cat([prompt_ids, cont_ids], dim=1)

            logits = forward_with_sae_feature_roundtrip_layer(
                model,
                input_ids=full_ids,
                layer=int(layer),
                sae=sae,
                transform=transform,
                config=SAEPatchConfig(roundtrip_replace=True),
            )

            P = prompt_ids.size(1)
            C = cont_ids.size(1)
            logits_slice = logits[:, P - 1 : P + C - 1, :].to(dtype=logprobs_dtype)
            log_probs = torch.log_softmax(logits_slice, dim=-1)
            gathered = log_probs.gather(2, cont_ids.unsqueeze(-1)).squeeze(-1)  # (1, C)

            if not torch.isfinite(gathered).all():
                if strict_finite:
                    raise FloatingPointError("Non-finite log-probability detected during SAE roundtrip scoring.")
                gathered = torch.where(torch.isfinite(gathered), gathered, torch.full_like(gathered, -1e9))

            lp = gathered.mean(dim=1) if normalize_by_length else gathered.sum(dim=1)
            vals.append(float(lp.item()))
        scores[str(label)] = _logmeanexp(vals)
    return scores


@torch.no_grad()
def _disamb_accuracy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: List[DisambPair],
    device: torch.device,
    *,
    normalize_by_length: bool,
    sae: Optional[SAEProtocol] = None,
    layer: Optional[int] = None,
    transform: Optional[SAEInputTransform] = None,
) -> float:
    correct = 0
    total = 0
    for it in items:
        for side in (it.a, it.b):
            if sae is None:
                scores = score_labels_next_continuations(
                    model, tokenizer, side.prompt, it.choices, device, normalize_by_length=normalize_by_length
                ).by_label
            else:
                if layer is None or transform is None:
                    raise ValueError("layer and transform required for SAE roundtrip accuracy")
                scores = _score_labels_next_continuations_roundtrip(
                    model,
                    tokenizer,
                    prompt=side.prompt,
                    choices=it.choices,
                    device=device,
                    sae=sae,
                    layer=int(layer),
                    transform=transform,
                    normalize_by_length=normalize_by_length,
                )
            pred = max(scores.items(), key=lambda kv: kv[1])[0]
            correct += int(pred == side.expected_label)
            total += 1
    return float(correct / max(1, total))


@torch.no_grad()
def check_sae_sterility(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sae: SAEProtocol,
    layer: int,
    scale: float,
    disamb_items: List[DisambPair],
    device: torch.device,
    tolerance: float = 0.05,
    kl_tolerance: float = 0.1,
    normalize_by_length: bool = True,
    recon_prompt: Optional[str] = None,
) -> SterilityResult:
    transform = SAEInputTransform(scale=float(scale))

    baseline_acc = _disamb_accuracy(
        model, tokenizer, disamb_items, device, normalize_by_length=normalize_by_length
    )
    roundtrip_acc = _disamb_accuracy(
        model,
        tokenizer,
        disamb_items,
        device,
        normalize_by_length=normalize_by_length,
        sae=sae,
        layer=int(layer),
        transform=transform,
    )
    delta = float(roundtrip_acc - baseline_acc)

    # Secondary sterility gate: distributional shift under full-sequence SAE roundtrip replacement.
    # We use mean per-token KL(p_base || p_roundtrip) across the DISAMB prompts.
    logprobs_dtype, strict_finite = get_logprob_computation_config()
    kl_sum = 0.0
    kl_count = 0
    for it in disamb_items:
        for side in (it.a, it.b):
            prompt_ids = _encode(tokenizer, side.prompt, device=device)
            base_logits = model(prompt_ids, use_cache=False, return_dict=True).logits
            rt_logits = forward_with_sae_feature_roundtrip_layer(
                model,
                input_ids=prompt_ids,
                layer=int(layer),
                sae=sae,
                transform=transform,
                config=SAEPatchConfig(roundtrip_replace=True),
            )
            base_logp = torch.log_softmax(base_logits.to(dtype=logprobs_dtype), dim=-1)
            rt_logp = torch.log_softmax(rt_logits.to(dtype=logprobs_dtype), dim=-1)
            p = base_logp.exp()
            kl = (p * (base_logp - rt_logp)).sum(dim=-1)  # (1, S)
            if not torch.isfinite(kl).all():
                if strict_finite:
                    raise FloatingPointError("Non-finite KL detected during SAE sterility check.")
                kl = torch.where(torch.isfinite(kl), kl, torch.full_like(kl, float("inf")))
            kl_sum += float(kl.sum().item())
            kl_count += int(kl.numel())
    mean_token_kl = float(kl_sum / max(1, kl_count))

    passed = bool((abs(delta) <= float(tolerance)) and (mean_token_kl <= float(kl_tolerance)))

    # Reconstruction sanity check on a single prompt (cheap, but catches obvious scaling/stream mismatches).
    prompt = recon_prompt or disamb_items[0].a.prompt
    input_ids = _encode(tokenizer, prompt, device=device)
    # Capture logits just to force the hookpoint to run; MSE computed directly from SAE roundtrip below.
    _ = forward_with_sae_feature_roundtrip_layer(
        model,
        input_ids=input_ids,
        layer=int(layer),
        sae=sae,
        transform=transform,
        config=SAEPatchConfig(roundtrip_replace=True),
    )

    # Compute recon MSE on the same captured tensor semantics used by the hook: block output forward tensor.
    from aom.interventions.activation_patching import get_decoder_blocks

    blocks = get_decoder_blocks(model)
    captured: list[torch.Tensor] = []

    def _cap(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h.detach())

    handle = blocks[int(layer)].register_forward_hook(_cap)
    try:
        _ = model(input_ids=input_ids, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected 1 capture for recon MSE; got {len(captured)}")
    hidden = captured[0]
    recon = transform.inverse(sae.decode(sae.encode(transform.forward(hidden))))
    recon_mse = float(torch.mean((recon - hidden) ** 2).item())

    return SterilityResult(
        baseline_acc=float(baseline_acc),
        roundtrip_acc=float(roundtrip_acc),
        delta=float(delta),
        passed=passed,
        recon_mse=float(recon_mse),
        mean_token_kl=float(mean_token_kl),
    )
