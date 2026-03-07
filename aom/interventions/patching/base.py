from __future__ import annotations

import json
import hashlib
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aom.interventions.activation_patching import PatchSpanSite, get_block_outputs, get_num_layers
from aom.stats import bootstrap_ci_metric, cohens_d

from .scoring import LabelAggregation, score_labels_next_continuations_ids, score_labels_next_continuations_patched_ids


def _slug(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "x"


def _margin(scores: Mapping[str, float], expected: str) -> float:
    if expected not in scores:
        raise KeyError(f"expected label {expected!r} missing from scores: {sorted(scores.keys())!r}")
    exp = float(scores[expected])
    best_other = max(float(v) for k, v in scores.items() if k != expected)
    return float(exp - best_other)


def _argmax_label(scores: Mapping[str, float]) -> str:
    return max(scores.items(), key=lambda kv: float(kv[1]))[0]


@dataclass(frozen=True)
class PatchingCase:
    case_id: str
    receiver_prompt: str
    donor_prompt: str
    receiver_ids: torch.Tensor
    donor_ids: torch.Tensor
    receiver_span: Tuple[int, ...]
    donor_span: Tuple[int, ...]
    choices: Mapping[str, List[str]]
    expected_label: str
    strata: Mapping[str, str] = field(default_factory=dict)
    label_aggregation: LabelAggregation = "logmeanexp"
    # effect = effect_sign * (patched_margin - base_margin). Use -1.0 when the expected effect is a *decrease*
    # in the expected-label margin (e.g., constraint ablation intended to degrade coherence).
    effect_sign: float = 1.0


@dataclass(frozen=True)
class CaseSkip:
    case_id: str
    reason: str


@dataclass(frozen=True)
class ComparisonSpec:
    name: str
    stratum_key: str
    a_value: str
    b_value: str


class ActivationPatchingProtocol(ABC):
    """
    Protocol interface: subclasses own span identification and donor/receiver construction.

    The base runner owns hook plumbing, sham construction, and metric aggregation.
    """

    name: str = "patching_protocol"

    @abstractmethod
    def build_cases(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        items: Sequence[Any],
        device: torch.device,
    ) -> Tuple[List[PatchingCase], List[CaseSkip]]:
        raise NotImplementedError

    def primary_comparisons(self) -> Sequence[ComparisonSpec]:
        return ()


@torch.no_grad()
def run_activation_patching(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    protocol: ActivationPatchingProtocol,
    items: Sequence[Any],
    device: torch.device,
    layers: Optional[Sequence[int]] = None,
    normalize_by_length: bool = True,
    ci: float = 0.95,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    if layers is None:
        n_layers = int(get_num_layers(model))
        layers = list(range(n_layers))
    layer_list = [int(l) for l in layers]
    if not layer_list:
        raise ValueError("layers must be non-empty")

    cases, skips = protocol.build_cases(tokenizer=tokenizer, items=items, device=device)

    per_layer_sum = {int(l): 0.0 for l in layer_list}
    per_layer_n = {int(l): 0 for l in layer_list}
    sham_per_layer_sum = {int(l): 0.0 for l in layer_list}
    sham_per_layer_n = {int(l): 0 for l in layer_list}

    max_effects: List[float] = []
    max_layers: List[int] = []
    max_flips: List[float] = []
    max_effects_norm: List[float] = []
    sham_max_effects: List[float] = []

    # Stratified aggregates: for each (key,value), collect max_effects/sham_max_effects.
    by_stratum_effects: Dict[str, List[float]] = {}
    by_stratum_sham_effects: Dict[str, List[float]] = {}
    by_stratum_flips: Dict[str, List[float]] = {}

    n_total = int(len(cases))
    n_used = 0
    n_span_mismatch = 0

    def _record_strata(case: PatchingCase, *, eff: float, sham_eff: float, flip: float) -> None:
        for k, v in case.strata.items():
            key = f"{_slug(str(k))}__{_slug(str(v))}"
            by_stratum_effects.setdefault(key, []).append(float(eff))
            by_stratum_sham_effects.setdefault(key, []).append(float(sham_eff))
            by_stratum_flips.setdefault(key, []).append(float(flip))

    for case in cases:
        if len(case.receiver_span) < 1 or len(case.donor_span) < 1:
            n_span_mismatch += 1
            continue
        if len(case.receiver_span) != len(case.donor_span):
            n_span_mismatch += 1
            continue

        recv_ids = case.receiver_ids
        donor_ids = case.donor_ids

        # Baseline receiver scores (unpatched).
        base_scores = score_labels_next_continuations_ids(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=recv_ids,
            choices=case.choices,
            device=device,
            normalize_by_length=bool(normalize_by_length),
            label_aggregation=case.label_aggregation,
        )
        base_pred = _argmax_label(base_scores)
        base_margin = _margin(base_scores, expected=str(case.expected_label))

        best_eff: float | None = None
        best_layer: int | None = None
        best_flip: float | None = None
        best_norm: float | None = None
        best_sham: float | None = None

        for layer in layer_list:
            donor_block = get_block_outputs(model, donor_ids, layers=[int(layer)])
            recv_block = get_block_outputs(model, recv_ids, layers=[int(layer)])
            donor_h = donor_block[int(layer)][0, list(case.donor_span), :].detach()
            recv_h = recv_block[int(layer)][0, list(case.receiver_span), :].detach()

            patched_scores = score_labels_next_continuations_patched_ids(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=recv_ids,
                choices=case.choices,
                device=device,
                patch_site=PatchSpanSite(layer=int(layer), token_indices=tuple(int(i) for i in case.receiver_span)),
                replacement=donor_h,
                normalize_by_length=bool(normalize_by_length),
                label_aggregation=case.label_aggregation,
            )
            sham_scores = score_labels_next_continuations_patched_ids(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=recv_ids,
                choices=case.choices,
                device=device,
                patch_site=PatchSpanSite(layer=int(layer), token_indices=tuple(int(i) for i in case.receiver_span)),
                replacement=recv_h,
                normalize_by_length=bool(normalize_by_length),
                label_aggregation=case.label_aggregation,
            )

            patched_pred = _argmax_label(patched_scores)
            patched_margin = _margin(patched_scores, expected=str(case.expected_label))
            sham_margin = _margin(sham_scores, expected=str(case.expected_label))

            raw_delta = float(patched_margin - base_margin)
            raw_sham_delta = float(sham_margin - base_margin)
            eff = float(float(case.effect_sign) * raw_delta)
            sham_eff = float(float(case.effect_sign) * raw_sham_delta)
            norm_eff = float(eff / (abs(base_margin) + 1e-8))

            if float(case.effect_sign) >= 0.0:
                flip = float((base_pred != str(case.expected_label)) and (patched_pred == str(case.expected_label)))
            else:
                flip = float((base_pred == str(case.expected_label)) and (patched_pred != str(case.expected_label)))

            per_layer_sum[int(layer)] += eff
            per_layer_n[int(layer)] += 1
            sham_per_layer_sum[int(layer)] += sham_eff
            sham_per_layer_n[int(layer)] += 1

            if best_eff is None or eff > best_eff:
                best_eff = float(eff)
                best_layer = int(layer)
                best_flip = float(flip)
                best_norm = float(norm_eff)
            if best_sham is None or sham_eff > best_sham:
                best_sham = float(sham_eff)

        if best_eff is None or best_layer is None or best_sham is None:
            continue
        n_used += 1
        max_effects.append(float(best_eff))
        max_layers.append(int(best_layer))
        if best_flip is not None:
            max_flips.append(float(best_flip))
        if best_norm is not None:
            max_effects_norm.append(float(best_norm))
        sham_max_effects.append(float(best_sham))
        _record_strata(case, eff=float(best_eff), sham_eff=float(best_sham), flip=float(best_flip or 0.0))

    def _boot(name: str, values: List[float]) -> Dict[str, Any]:
        mv, lo, hi = bootstrap_ci_metric(values, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed)
        out: Dict[str, Any] = {
            name: float(mv.value),
            f"{name}_ci_low": float(lo),
            f"{name}_ci_high": float(hi),
            f"{name}_n": int(mv.n),
            f"{name}_valid": bool(mv.valid),
        }
        if mv.reason is not None:
            out[f"{name}_reason"] = str(mv.reason)
        return out

    out: Dict[str, Any] = {
        "protocol": str(getattr(protocol, "name", "patching_protocol")),
        "layers": ",".join(str(int(l)) for l in layer_list),
        "n_cases_total": int(n_total),
        "n_cases_used": int(n_used),
        "n_cases_skipped_protocol": int(len(skips)),
        "n_cases_skipped_span_mismatch": int(n_span_mismatch),
        "skip_reason_counts_json": json.dumps(
            {str(k): int(v) for k, v in sorted(Counter(str(s.reason) for s in skips).items())}, sort_keys=True
        )
        if skips
        else "",
        "skip_reasons_sha256": hashlib.sha256(
            ("\n".join(sorted(f"{s.case_id}\t{s.reason}" for s in skips))).encode("utf-8")
        ).hexdigest()
        if skips
        else "",
        **_boot("mean_max_effect", max_effects),
        "mean_argmax_layer": float(sum(max_layers) / max(1, len(max_layers))) if max_layers else float("nan"),
        **_boot("flip_rate_at_best_layer", max_flips),
        **_boot("mean_norm_max_effect", max_effects_norm),
        **_boot("mean_sham_max_effect", sham_max_effects),
    }

    for l in layer_list:
        out[f"effect_layer_{int(l)}"] = float(per_layer_sum[int(l)] / max(1, per_layer_n[int(l)]))
        out[f"sham_effect_layer_{int(l)}"] = float(sham_per_layer_sum[int(l)] / max(1, sham_per_layer_n[int(l)]))

    # Stratified outputs: each (key,value) pair gets its own mean/sham/flip summaries.
    for key, vals in sorted(by_stratum_effects.items()):
        prefix = f"stratum_{key}"
        out.update({f"{prefix}_{k}": v for k, v in _boot("mean_max_effect", vals).items()})
        sham_vals = by_stratum_sham_effects.get(key, [])
        out.update({f"{prefix}_{k}": v for k, v in _boot("mean_sham_max_effect", sham_vals).items()})
        flip_vals = by_stratum_flips.get(key, [])
        out.update({f"{prefix}_{k}": v for k, v in _boot("flip_rate_at_best_layer", flip_vals).items()})
        out[f"{prefix}_n_cases_used"] = int(len(vals))

    # Effect sizes for primary reviewer-facing comparisons.
    for spec in protocol.primary_comparisons():
        k = f"{_slug(str(spec.stratum_key))}__{_slug(str(spec.a_value))}"
        j = f"{_slug(str(spec.stratum_key))}__{_slug(str(spec.b_value))}"
        a = by_stratum_effects.get(k, [])
        b = by_stratum_effects.get(j, [])
        name = _slug(str(spec.name))
        out[f"comparison_{name}_n_a"] = int(len(a))
        out[f"comparison_{name}_n_b"] = int(len(b))
        out[f"comparison_{name}_mean_diff"] = float(sum(a) / max(1, len(a)) - (sum(b) / max(1, len(b)))) if a and b else float("nan")
        out[f"comparison_{name}_cohens_d"] = float(cohens_d(a, b))

    return out
