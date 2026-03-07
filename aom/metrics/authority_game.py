from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aom.data.schemas import AuthorityPair, SubstringSpan
from aom.interventions.activation_patching import PatchSpanSite, get_block_outputs
from aom.interventions.patching.authority_protocol import AuthorityLanguageGameProtocol, AuthorityPatchingConfig
from aom.interventions.patching.scoring import LabelAggregation, score_labels_next_continuations_ids
from aom.interventions.sae_adapter import SAEInputTransform, SAEProtocol, SAEPatchConfig
from aom.interventions.sae_patching import forward_with_sae_feature_patching_span
from aom.metrics.disamb import _logmeanexp
from aom.token_spans import token_span_for_substring
from aom.utils import bootstrap_ci, get_logprob_computation_config


def _encode(tokenizer: PreTrainedTokenizerBase, text: str, device: torch.device) -> torch.Tensor:
    enc = tokenizer(str(text), return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"].to(device)


def _aggregate(vals: Sequence[float], *, agg: LabelAggregation) -> float:
    if len(vals) < 1:
        raise ValueError("aggregation requires at least one value")
    if agg == "logmeanexp":
        return float(_logmeanexp(list(vals)))
    if agg == "mean":
        return float(sum(float(x) for x in vals) / max(1, len(vals)))
    raise ValueError(f"Unknown label aggregation: {agg!r}")


def _margin(scores: Mapping[str, float], expected: str) -> float:
    if expected not in scores:
        raise KeyError(f"expected label {expected!r} missing from scores: {sorted(scores.keys())!r}")
    exp = float(scores[expected])
    best_other = max(float(v) for k, v in scores.items() if k != expected)
    return float(exp - best_other)


def _infer_sae_device_dtype(sae: SAEProtocol) -> tuple[torch.device, torch.dtype]:
    if isinstance(sae, torch.nn.Module):
        p = next(sae.parameters(), None)
        if p is not None:
            return p.device, p.dtype
    W_dec = getattr(sae, "W_dec", None)
    if isinstance(W_dec, torch.Tensor):
        return W_dec.device, W_dec.dtype
    return torch.device("cpu"), torch.float32


@dataclass(frozen=True)
class AuthoritySAESubsetResult:
    mean_effect: float
    ci_low: float
    ci_high: float
    n_cases_total: int
    n_cases_used: int
    mean_delta_norm: float
    mean_delta_norm_ci_low: float
    mean_delta_norm_ci_high: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_effect": float(self.mean_effect),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
            "n_cases_total": int(self.n_cases_total),
            "n_cases_used": int(self.n_cases_used),
            "mean_delta_norm": float(self.mean_delta_norm),
            "mean_delta_norm_ci_low": float(self.mean_delta_norm_ci_low),
            "mean_delta_norm_ci_high": float(self.mean_delta_norm_ci_high),
        }


@torch.no_grad()
def score_labels_next_continuations_sae_patched_ids(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,
    choices: Mapping[str, List[str]],
    device: torch.device,
    patch_site: PatchSpanSite,
    sae: SAEProtocol,
    transform: SAEInputTransform,
    policy,
    config: Optional[SAEPatchConfig],
    state: Optional[SAEHookState],
    normalize_by_length: bool,
    label_aggregation: LabelAggregation,
) -> Dict[str, float]:
    logprobs_dtype, strict_finite = get_logprob_computation_config()
    scores: Dict[str, float] = {}
    for label, continuations in choices.items():
        if len(continuations) < 1:
            raise ValueError(f"Empty continuation list for label={label}")
        vals: List[float] = []
        for cont in continuations:
            cont_ids = _encode(tokenizer, str(cont), device=device)
            full_ids = torch.cat([prompt_ids, cont_ids], dim=1)
            logits = forward_with_sae_feature_patching_span(
                model,
                input_ids=full_ids,
                site=patch_site,
                sae=sae,
                policy=policy,
                transform=transform,
                config=config,
                state=state,
            )

            P = prompt_ids.size(1)
            C = cont_ids.size(1)
            logits_slice = logits[:, P - 1 : P + C - 1, :].to(dtype=logprobs_dtype)
            log_probs = torch.log_softmax(logits_slice, dim=-1)
            gathered = log_probs.gather(2, cont_ids.unsqueeze(-1)).squeeze(-1)  # (1, C)
            if not torch.isfinite(gathered).all():
                if strict_finite:
                    raise FloatingPointError("Non-finite log-probability detected during SAE patched scoring.")
                gathered = torch.where(torch.isfinite(gathered), gathered, torch.full_like(gathered, -1e9))
            lp = gathered.mean(dim=1) if bool(normalize_by_length) else gathered.sum(dim=1)
            vals.append(float(lp.item()))
        scores[str(label)] = float(_aggregate(vals, agg=label_aggregation))
    return scores


def _span_for_site_side(
    *,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    spec: SubstringSpan,
) -> list[int]:
    span, _tok = token_span_for_substring(tokenizer, str(prompt), str(spec.substr), int(spec.occurrence))
    return span


@torch.inference_mode()
def select_top_features_by_activation_authority(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: Sequence[AuthorityPair],
    device: torch.device,
    layer: int,
    site: str,
    sae: SAEProtocol,
    transform: SAEInputTransform,
    n_features: int = 128,
    eps_active: float = 1e-6,
    max_pairs: int = 64,
) -> List[int]:
    counts: Optional[torch.Tensor] = None
    used = 0
    sae_device, sae_dtype = _infer_sae_device_dtype(sae)
    for it in items:
        if used >= int(max_pairs):
            break
        spec = it.sites.get(str(site), None)
        if spec is None:
            continue
        for side_name, side, side_spec in (
            ("a", it.a, spec.a),
            ("b", it.b, spec.b),
        ):
            try:
                span = _span_for_site_side(tokenizer=tokenizer, prompt=str(side.prompt), spec=side_spec)
            except Exception:
                continue
            if not span:
                continue
            ids = _encode(tokenizer, str(side.prompt), device=device)
            out = get_block_outputs(model, ids, layers=[int(layer)])[int(layer)]  # [1, seq, d_model]
            h = out[:, span, :].to(device=sae_device, dtype=sae_dtype)
            feats = sae.encode(transform.forward(h))
            active = (feats.abs() > float(eps_active)).sum(dim=(0, 1)).to(dtype=torch.float32)
            if counts is None:
                counts = active
            else:
                counts = counts + active
            _ = side_name
        used += 1
    if counts is None:
        return []
    k = min(int(n_features), int(counts.numel()))
    top = torch.topk(counts, k=k).indices.tolist()
    return [int(x) for x in top]


@torch.inference_mode()
def select_top_features_by_delta_authority(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: Sequence[AuthorityPair],
    device: torch.device,
    layer: int,
    site: str,
    sae: SAEProtocol,
    transform: SAEInputTransform,
    n_features: int = 128,
    max_pairs: int = 64,
    max_tokens_per_site: int | None = None,
    span_take: str = "last",
) -> List[int]:
    """
    Select top features by aggregated |Δz| at the patch site across donor→receiver directions.

    This is a better default for subset-copy recovery than activation frequency.
    """
    protocol = AuthorityLanguageGameProtocol(
        config=AuthorityPatchingConfig(
            include_sites=(str(site),),
            max_tokens_per_site=max_tokens_per_site,
            span_take=str(span_take),
        )
    )
    cases, _skips = protocol.build_cases(tokenizer=tokenizer, items=list(items), device=device)
    if int(max_pairs) > 0:
        # cases are 2 directions per pair; cap approximately by pairs.
        cases = cases[: int(max_pairs) * 2]
    if not cases:
        return []

    sae_device, sae_dtype = _infer_sae_device_dtype(sae)
    scores: torch.Tensor | None = None
    for case in cases:
        donor_out = get_block_outputs(model, case.donor_ids, layers=[int(layer)])[int(layer)]
        recv_out = get_block_outputs(model, case.receiver_ids, layers=[int(layer)])[int(layer)]
        donor_slice = (
            donor_out[0, list(case.donor_span), :].detach().unsqueeze(0).to(device=sae_device, dtype=sae_dtype)
        )
        recv_slice = (
            recv_out[0, list(case.receiver_span), :].detach().unsqueeze(0).to(device=sae_device, dtype=sae_dtype)
        )
        donor_feats = sae.encode(transform.forward(donor_slice))
        recv_feats = sae.encode(transform.forward(recv_slice))
        if donor_feats.shape != recv_feats.shape:
            continue
        delta = (donor_feats - recv_feats).abs().sum(dim=(0, 1)).to(dtype=torch.float32)
        if scores is None:
            scores = delta
        else:
            scores = scores + delta
    if scores is None or int(scores.numel()) < 1:
        return []
    k = min(int(n_features), int(scores.numel()))
    top = torch.topk(scores, k=k).indices.tolist()
    return [int(x) for x in top]


@torch.inference_mode()
def compute_authority_sae_subset_copy_effect(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: Sequence[AuthorityPair],
    device: torch.device,
    layer: int,
    site: str,
    sae: SAEProtocol,
    transform: SAEInputTransform,
    feature_ids: Sequence[int],
    config: Optional[SAEPatchConfig] = None,
    normalize_by_length: bool = True,
    max_tokens_per_site: int | None = None,
    span_take: str = "last",
    ci: float = 0.95,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
) -> AuthoritySAESubsetResult:
    """
    Donor→receiver subset-copy in SAE feature space at a specific site and layer.

    replacement_features = recv_features + mask*(donor_features - recv_features)
    where mask selects `feature_ids`.
    """
    protocol = AuthorityLanguageGameProtocol(
        config=AuthorityPatchingConfig(
            include_sites=(str(site),),
            max_tokens_per_site=max_tokens_per_site,
            span_take=str(span_take),
        )
    )
    cases, _skips = protocol.build_cases(tokenizer=tokenizer, items=list(items), device=device)

    sae_device, sae_dtype = _infer_sae_device_dtype(sae)
    mask: Optional[torch.Tensor] = None

    effects: List[float] = []
    delta_norms: List[float] = []
    for case in cases:
        # Baseline receiver scores.
        base_scores = score_labels_next_continuations_ids(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=case.receiver_ids,
            choices=case.choices,
            device=device,
            normalize_by_length=bool(normalize_by_length),
            label_aggregation=case.label_aggregation,
        )
        base_margin = _margin(base_scores, expected=str(case.expected_label))

        donor_out = get_block_outputs(model, case.donor_ids, layers=[int(layer)])[int(layer)]
        recv_out = get_block_outputs(model, case.receiver_ids, layers=[int(layer)])[int(layer)]

        donor_slice = (
            donor_out[0, list(case.donor_span), :].detach().unsqueeze(0).to(device=sae_device, dtype=sae_dtype)
        )
        recv_slice = (
            recv_out[0, list(case.receiver_span), :].detach().unsqueeze(0).to(device=sae_device, dtype=sae_dtype)
        )
        donor_feats = sae.encode(transform.forward(donor_slice))
        recv_feats = sae.encode(transform.forward(recv_slice))
        if donor_feats.shape != recv_feats.shape:
            continue

        if mask is None:
            d_sae = int(recv_feats.size(-1))
            mask = torch.zeros((1, 1, d_sae), device=recv_feats.device, dtype=recv_feats.dtype)
            for fid in feature_ids:
                if 0 <= int(fid) < d_sae:
                    mask[:, :, int(fid)] = 1.0

        delta = donor_feats - recv_feats
        if mask is None:
            raise RuntimeError("mask not initialized")
        delta_masked = delta * mask.expand_as(delta)
        delta_norms.append(float(delta_masked.norm().item()))
        replacement = recv_feats + delta_masked

        from aom.metrics.sae_patching import ReplaceFeaturesAtIndicesPolicy

        policy = ReplaceFeaturesAtIndicesPolicy(
            token_indices=list(case.receiver_span),
            replacement_features=replacement,
        )
        patched_scores = score_labels_next_continuations_sae_patched_ids(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=case.receiver_ids,
            choices=case.choices,
            device=device,
            patch_site=PatchSpanSite(layer=int(layer), token_indices=tuple(int(x) for x in case.receiver_span)),
            sae=sae,
            transform=transform,
            policy=policy,
            config=config,
            state=None,
            normalize_by_length=bool(normalize_by_length),
            label_aggregation=case.label_aggregation,
        )
        patched_margin = _margin(patched_scores, expected=str(case.expected_label))
        eff = float(float(case.effect_sign) * (patched_margin - base_margin))
        effects.append(float(eff))

    mean_eff, lo, hi = bootstrap_ci(effects, n_bootstrap=int(bootstrap_n), ci=float(ci), seed=int(bootstrap_seed))
    mean_dn, dn_lo, dn_hi = bootstrap_ci(delta_norms, n_bootstrap=int(bootstrap_n), ci=float(ci), seed=int(bootstrap_seed))
    return AuthoritySAESubsetResult(
        mean_effect=float(mean_eff),
        ci_low=float(lo),
        ci_high=float(hi),
        n_cases_total=int(len(cases)),
        n_cases_used=int(len(effects)),
        mean_delta_norm=float(mean_dn),
        mean_delta_norm_ci_low=float(dn_lo),
        mean_delta_norm_ci_high=float(dn_hi),
    )


def semantic_gap(*, e_raw: float, e_sae: float, eps: float = 1e-8) -> float:
    if not math.isfinite(float(e_raw)) or abs(float(e_raw)) < float(eps):
        return float("nan")
    return float(1.0 - (float(e_sae) / float(e_raw)))


def family_synergy(*, e_family: float, e_atoms: Iterable[float]) -> float:
    return float(float(e_family) - sum(float(x) for x in e_atoms))
