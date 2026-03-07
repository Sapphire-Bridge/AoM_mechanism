from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aom.data.schemas import DisambPair
from aom.interventions.activation_patching import PatchSpanSite, get_block_outputs
from aom.interventions.sae_adapter import FeaturePolicy, SAEInputTransform, SAEProtocol
from aom.interventions.sae_patching import forward_with_sae_feature_patching_span
from aom.metrics.disamb import _encode_prompt, _logmeanexp, _margin, score_labels_next_continuations
from aom.token_spans import token_span_for_substring
from aom.utils import get_logprob_computation_config


@dataclass(frozen=True)
class FeatureEffectMatrix:
    feature_ids: tuple[int, ...]
    condition_ids: tuple[str, ...]
    effects: Dict[int, Dict[str, float]]

    def to_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for fid in self.feature_ids:
            row = {"feature_id": int(fid)}
            vals = self.effects.get(int(fid), {})
            for cid in self.condition_ids:
                row[str(cid)] = float(vals.get(str(cid), float("nan")))
            rows.append(row)
        return rows


@dataclass(frozen=True)
class ZeroFeaturePolicy(FeaturePolicy):
    feature_ids: tuple[int, ...]

    def apply(
        self,
        features: torch.Tensor,
        *,
        site_mask: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,  # noqa: ARG002
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape (batch, seq, d_sae)")
        if site_mask.ndim != 3 or int(site_mask.size(-1)) != 1:
            raise ValueError("site_mask must have shape (batch, seq, 1)")
        patched = features.clone()
        for fid in self.feature_ids:
            if int(fid) < 0 or int(fid) >= int(features.size(-1)):
                continue
            m = site_mask.expand_as(features[:, :, int(fid) : int(fid) + 1]).squeeze(-1)
            patched[:, :, int(fid)] = torch.where(m, torch.zeros_like(patched[:, :, int(fid)]), patched[:, :, int(fid)])
        return patched


def _score_labels_next_continuations_sae_policy_patched(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    choices: Mapping[str, List[str]],
    device: torch.device,
    layer: int,
    token_indices: Sequence[int],
    sae: SAEProtocol,
    transform: SAEInputTransform,
    policy: FeaturePolicy,
    normalize_by_length: bool,
) -> Dict[str, float]:
    prompt_ids = _encode_prompt(tokenizer, prompt, device=device)
    logprobs_dtype, strict_finite = get_logprob_computation_config()
    scores: Dict[str, float] = {}
    for label, continuations in choices.items():
        vals: List[float] = []
        for cont in continuations:
            cont_ids = tokenizer(str(cont), return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            full_ids = torch.cat([prompt_ids, cont_ids], dim=1)
            logits = forward_with_sae_feature_patching_span(
                model,
                input_ids=full_ids,
                site=PatchSpanSite(layer=int(layer), token_indices=tuple(int(i) for i in token_indices)),
                sae=sae,
                policy=policy,
                transform=transform,
                config=None,
            )
            p = prompt_ids.size(1)
            c = cont_ids.size(1)
            logits_slice = logits[:, p - 1 : p + c - 1, :].to(dtype=logprobs_dtype)
            log_probs = torch.log_softmax(logits_slice, dim=-1)
            gathered = log_probs.gather(2, cont_ids.unsqueeze(-1)).squeeze(-1)
            if not torch.isfinite(gathered).all():
                if strict_finite:
                    raise FloatingPointError("Non-finite log-probability detected in feature effect scoring.")
                gathered = torch.where(torch.isfinite(gathered), gathered, torch.full_like(gathered, -1e9))
            lp = gathered.mean(dim=1) if normalize_by_length else gathered.sum(dim=1)
            vals.append(float(lp.item()))
        scores[str(label)] = _logmeanexp(vals)
    return scores


@torch.inference_mode()
def select_top_features_by_activation(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: Sequence[DisambPair],
    device: torch.device,
    layer: int,
    sae: SAEProtocol,
    transform: SAEInputTransform,
    n_features: int = 128,
    eps_active: float = 1e-6,
    max_pairs: int = 64,
) -> List[int]:
    counts: Optional[torch.Tensor] = None
    used = 0
    for it in items:
        if used >= int(max_pairs):
            break
        for side in (it.a, it.b):
            span, _ = token_span_for_substring(tokenizer, side.prompt, it.target, it.target_occurrence)
            if not span:
                continue
            ids = _encode_prompt(tokenizer, side.prompt, device=device)
            out = get_block_outputs(model, ids, layers=[int(layer)])[int(layer)]  # [1, seq, d_model]
            h = out[:, span, :]
            feats = sae.encode(transform.forward(h))
            active = (feats.abs() > float(eps_active)).sum(dim=(0, 1)).to(dtype=torch.float32)
            if counts is None:
                counts = active
            else:
                counts = counts + active
        used += 1
    if counts is None:
        return []
    k = min(int(n_features), int(counts.numel()))
    top = torch.topk(counts, k=k).indices.tolist()
    return [int(x) for x in top]


@torch.inference_mode()
def build_disamb_feature_effect_matrix(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: Sequence[DisambPair],
    device: torch.device,
    layer: int,
    sae: SAEProtocol,
    transform: SAEInputTransform,
    feature_ids: Sequence[int],
    max_pairs: int = 64,
    normalize_by_length: bool = True,
) -> FeatureEffectMatrix:
    rows = list(items)
    if int(max_pairs) > 0:
        rows = rows[: int(max_pairs)]
    condition_ids = [str(it.pair_id) for it in rows]

    effects: Dict[int, Dict[str, float]] = {}
    for fid in feature_ids:
        policy = ZeroFeaturePolicy(feature_ids=(int(fid),))
        per_pair: Dict[str, float] = {}
        for it in rows:
            vals: List[float] = []
            for side in (it.a, it.b):
                span, _ = token_span_for_substring(tokenizer, side.prompt, it.target, it.target_occurrence)
                if not span:
                    continue
                base_scores = score_labels_next_continuations(
                    model,
                    tokenizer,
                    side.prompt,
                    it.choices,
                    device,
                    normalize_by_length=bool(normalize_by_length),
                )
                patched_scores = _score_labels_next_continuations_sae_policy_patched(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=side.prompt,
                    choices=it.choices,
                    device=device,
                    layer=int(layer),
                    token_indices=span,
                    sae=sae,
                    transform=transform,
                    policy=policy,
                    normalize_by_length=bool(normalize_by_length),
                )
                base_margin = _margin(base_scores, expected=str(side.expected_label))
                patched_margin = float(
                    patched_scores[str(side.expected_label)]
                    - max(v for k, v in patched_scores.items() if str(k) != str(side.expected_label))
                )
                vals.append(float(patched_margin - base_margin))
            if vals:
                per_pair[str(it.pair_id)] = float(sum(vals) / len(vals))
        effects[int(fid)] = per_pair

    return FeatureEffectMatrix(
        feature_ids=tuple(int(x) for x in feature_ids),
        condition_ids=tuple(str(x) for x in condition_ids),
        effects=effects,
    )


def matrix_from_rows(rows: Iterable[Mapping[str, Any]]) -> FeatureEffectMatrix:
    rows_list = list(rows)
    if not rows_list:
        return FeatureEffectMatrix(feature_ids=(), condition_ids=(), effects={})
    feature_ids = sorted({int(r["feature_id"]) for r in rows_list})
    condition_ids = sorted({str(k) for r in rows_list for k in r.keys() if str(k) != "feature_id"})
    effects: Dict[int, Dict[str, float]] = {}
    for r in rows_list:
        fid = int(r["feature_id"])
        effects.setdefault(fid, {})
        for cid in condition_ids:
            v = r.get(cid, float("nan"))
            if isinstance(v, (int, float)):
                effects[fid][str(cid)] = float(v)
    return FeatureEffectMatrix(
        feature_ids=tuple(int(x) for x in feature_ids),
        condition_ids=tuple(str(x) for x in condition_ids),
        effects=effects,
    )
