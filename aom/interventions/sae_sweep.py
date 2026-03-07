from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, replace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aom.data.schemas import DisambPair
from aom.interventions.activation_patching import PatchSpanSite
from aom.interventions.sae_adapter import (
    FeaturePolicy,
    IdentityFeaturePolicy,
    RandomMatchedActiveMaskPolicy,
    RelativeThresholdPolicy,
    SAEHookState,
    SAEInputTransform,
    SAEPatchConfig,
    SAEProtocol,
)
from aom.interventions.sae_patching import forward_with_sae_feature_patching_span
from aom.metrics.disamb import score_labels_next_continuations
from aom.token_spans import token_span_for_substring
from aom.utils import get_logprob_computation_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineMetrics:
    accuracy: float
    mean_expected_nll: float
    n_pairs: int


@dataclass(frozen=True)
class SweepPoint:
    condition: str
    threshold: float
    target_layer: int
    control_layer: Optional[int]
    sparsity_gain: float
    accuracy: float
    delta_accuracy: float
    mean_expected_nll: float
    delta_expected_nll: float
    n_pairs_used: int
    h2_falsified: bool = False

    def to_row(self) -> Dict[str, object]:
        return asdict(self)


def _encode(tokenizer: PreTrainedTokenizerBase, text: str, device: torch.device) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"].to(device)


def _logmeanexp(xs: Sequence[float]) -> float:
    if len(xs) < 1:
        raise ValueError("logmeanexp requires at least one value")
    t = torch.tensor(list(xs), dtype=torch.float64)
    return float(torch.logsumexp(t, dim=0) - math.log(len(xs)))


def _mean(xs: Sequence[float]) -> float:
    return float(sum(float(x) for x in xs) / max(1, len(xs)))


def _baseline_per_pair(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: List[DisambPair],
    device: torch.device,
    normalize_by_length: bool,
) -> Dict[str, Tuple[float, float]]:
    """
    Returns per-pair (accuracy, expected_nll) where expected_nll = -score(expected_label).

    Note: score() is the label log-score used by AoM-DISAMB (logmeanexp over continuation logprobs),
    so expected_nll is a proxy (useful for deltas), not necessarily token-level cross-entropy.
    """
    out: Dict[str, Tuple[float, float]] = {}
    for it in items:
        side_acc: List[float] = []
        side_nll: List[float] = []
        for side in (it.a, it.b):
            scores = score_labels_next_continuations(
                model, tokenizer, side.prompt, it.choices, device, normalize_by_length=normalize_by_length
            )
            pred = scores.argmax_label()
            side_acc.append(float(pred == side.expected_label))
            side_nll.append(float(-scores.by_label[side.expected_label]))
        out[str(it.pair_id)] = (_mean(side_acc), _mean(side_nll))
    return out


def _score_labels_next_continuations_sae_policy_patched_ids(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,
    choices: Mapping[str, List[str]],
    device: torch.device,
    layer: int,
    token_indices: List[int],
    sae: SAEProtocol,
    transform: SAEInputTransform,
    policy: FeaturePolicy,
    config: Optional[SAEPatchConfig],
    state: SAEHookState,
    normalize_by_length: bool,
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
                site=PatchSpanSite(layer=int(layer), token_indices=tuple(int(x) for x in token_indices)),
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
                    raise FloatingPointError("Non-finite log-probability detected during SAE sweep scoring.")
                gathered = torch.where(torch.isfinite(gathered), gathered, torch.full_like(gathered, -1e9))
            lp = gathered.mean(dim=1) if normalize_by_length else gathered.sum(dim=1)
            vals.append(float(lp.item()))
        scores[str(label)] = _logmeanexp(vals)
    return scores


SpanSelector = Callable[[torch.Tensor, List[int], int], Optional[List[int]]]


def _patched_per_pair(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: List[DisambPair],
    device: torch.device,
    layer: int,
    sae_by_layer: Mapping[int, SAEProtocol],
    transform_by_layer: Mapping[int, SAEInputTransform],
    policy: FeaturePolicy,
    config: Optional[SAEPatchConfig],
    normalize_by_length: bool,
    span_selector: Optional[SpanSelector],
) -> tuple[Dict[str, Tuple[float, float]], SAEHookState]:
    if int(layer) not in sae_by_layer:
        raise ValueError(f"Missing SAE for layer {int(layer)}")
    if int(layer) not in transform_by_layer:
        raise ValueError(f"Missing transform for layer {int(layer)}")

    sae = sae_by_layer[int(layer)]
    transform = transform_by_layer[int(layer)]

    state = SAEHookState()
    per_pair: Dict[str, Tuple[float, float]] = {}

    for it in items:
        side_acc: List[float] = []
        side_nll: List[float] = []
        skip = False
        for side in (it.a, it.b):
            prompt_ids = _encode(tokenizer, side.prompt, device=device)
            target_span, _target_token_ids = token_span_for_substring(
                tokenizer, side.prompt, it.target, it.target_occurrence
            )
            seq_len = int(prompt_ids.size(1))
            token_indices = target_span
            if span_selector is not None:
                sel = span_selector(prompt_ids, target_span, seq_len)
                if sel is None:
                    skip = True
                    break
                token_indices = sel

            scores = _score_labels_next_continuations_sae_policy_patched_ids(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                choices=it.choices,
                device=device,
                layer=int(layer),
                token_indices=list(token_indices),
                sae=sae,
                transform=transform,
                policy=policy,
                config=config,
                state=state,
                normalize_by_length=normalize_by_length,
            )
            pred = max(scores.items(), key=lambda kv: kv[1])[0]
            side_acc.append(float(pred == side.expected_label))
            side_nll.append(float(-scores[side.expected_label]))

        if skip:
            continue
        per_pair[str(it.pair_id)] = (_mean(side_acc), _mean(side_nll))

    return per_pair, state


def _select_offtarget_span(
    prompt_ids: torch.Tensor, target_span: List[int], seq_len: int, *, buffer: int
) -> Optional[List[int]]:
    _ = prompt_ids
    if len(target_span) < 1:
        return None
    if seq_len < 1:
        return None
    k = int(len(target_span))
    if k > seq_len:
        return None

    lo = max(0, int(min(target_span)) - int(buffer))
    hi = min(int(seq_len) - 1, int(max(target_span)) + int(buffer))
    excluded = set(range(lo, hi + 1))

    for start in range(0, seq_len - k + 1):
        cand = list(range(int(start), int(start) + int(k)))
        if excluded.isdisjoint(cand):
            return cand
    return None


@torch.no_grad()
def run_threshold_sweep(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: List[DisambPair],
    device: torch.device,
    target_layer: int,
    sae_by_layer: Mapping[int, SAEProtocol],
    transform_by_layer: Mapping[int, SAEInputTransform],
    thresholds: Sequence[float],
    scale_mode: str = "quantile",
    quantile: float = 0.95,
    token_buffer: int = 2,
    off_target_layer_offset: int = 2,
    config: Optional[SAEPatchConfig] = None,
    normalize_by_length: bool = True,
) -> tuple[BaselineMetrics, List[SweepPoint]]:
    base = _baseline_per_pair(
        model=model, tokenizer=tokenizer, items=items, device=device, normalize_by_length=normalize_by_length
    )
    base_acc = _mean([acc for (acc, _nll) in base.values()])
    base_nll = _mean([nll for (_acc, nll) in base.values()])
    baseline = BaselineMetrics(accuracy=float(base_acc), mean_expected_nll=float(base_nll), n_pairs=int(len(base)))

    results: List[SweepPoint] = []

    def _add_result(
        *,
        condition: str,
        threshold: float,
        layer: int,
        control_layer: Optional[int],
        per_pair: Dict[str, Tuple[float, float]],
        state: SAEHookState,
    ) -> Optional[int]:
        if not per_pair:
            return None
        ids = sorted(per_pair.keys())
        base_subset = [base[i] for i in ids if i in base]
        if len(base_subset) != len(ids):
            raise RuntimeError("Baseline coverage mismatch during sweep aggregation.")

        base_acc_s = _mean([acc for (acc, _nll) in base_subset])
        base_nll_s = _mean([nll for (_acc, nll) in base_subset])
        acc = _mean([acc for (acc, _nll) in per_pair.values()])
        nll = _mean([nll for (_acc, nll) in per_pair.values()])

        results.append(
            SweepPoint(
                condition=str(condition),
                threshold=float(threshold),
                target_layer=int(layer),
                control_layer=int(control_layer) if control_layer is not None else None,
                sparsity_gain=float(state.sparsity_gain()),
                accuracy=float(acc),
                delta_accuracy=float(acc - base_acc_s),
                mean_expected_nll=float(nll),
                delta_expected_nll=float(nll - base_nll_s),
                n_pairs_used=int(len(per_pair)),
                h2_falsified=False,
            )
        )
        return int(len(results) - 1)

    # Identity/sham once (hook active, policy identity).
    per_pair, state = _patched_per_pair(
        model=model,
        tokenizer=tokenizer,
        items=items,
        device=device,
        layer=int(target_layer),
        sae_by_layer=sae_by_layer,
        transform_by_layer=transform_by_layer,
        policy=IdentityFeaturePolicy(),
        config=config,
        normalize_by_length=normalize_by_length,
        span_selector=None,
    )
    _ = _add_result(
        condition="identity",
        threshold=0.0,
        layer=int(target_layer),
        control_layer=None,
        per_pair=per_pair,
        state=state,
    )

    # Sweep thresholds.
    for t in thresholds:
        base_policy = RelativeThresholdPolicy(
            threshold=float(t),
            scale_mode=str(scale_mode),
            quantile=float(quantile),
            keep="high",
        )
        per_pair, state = _patched_per_pair(
            model=model,
            tokenizer=tokenizer,
            items=items,
            device=device,
            layer=int(target_layer),
            sae_by_layer=sae_by_layer,
            transform_by_layer=transform_by_layer,
            policy=base_policy,
            config=config,
            normalize_by_length=normalize_by_length,
            span_selector=None,
        )
        primary_idx = _add_result(
            condition="threshold",
            threshold=float(t),
            layer=int(target_layer),
            control_layer=None,
            per_pair=per_pair,
            state=state,
        )

        rand_policy = RandomMatchedActiveMaskPolicy(
            base_policy=base_policy,
            random_seed=int(config.random_seed if config is not None else 42),
            eps_active=float(config.eps_active if config is not None else 1e-6),
        )
        per_pair, state = _patched_per_pair(
            model=model,
            tokenizer=tokenizer,
            items=items,
            device=device,
            layer=int(target_layer),
            sae_by_layer=sae_by_layer,
            transform_by_layer=transform_by_layer,
            policy=rand_policy,
            config=config,
            normalize_by_length=normalize_by_length,
            span_selector=None,
        )
        _ = _add_result(
            condition="random_matched_active",
            threshold=float(t),
            layer=int(target_layer),
            control_layer=None,
            per_pair=per_pair,
            state=state,
        )

        anti_policy = RelativeThresholdPolicy(
            threshold=float(t),
            scale_mode=str(scale_mode),
            quantile=float(quantile),
            keep="low",
        )
        per_pair, state = _patched_per_pair(
            model=model,
            tokenizer=tokenizer,
            items=items,
            device=device,
            layer=int(target_layer),
            sae_by_layer=sae_by_layer,
            transform_by_layer=transform_by_layer,
            policy=anti_policy,
            config=config,
            normalize_by_length=normalize_by_length,
            span_selector=None,
        )
        anti_idx = _add_result(
            condition="anti_keep_low",
            threshold=float(t),
            layer=int(target_layer),
            control_layer=None,
            per_pair=per_pair,
            state=state,
        )

        if primary_idx is not None and anti_idx is not None:
            primary_pt = results[int(primary_idx)]
            anti_pt = results[int(anti_idx)]
            primary_effect = abs(float(primary_pt.delta_expected_nll))
            anti_effect = abs(float(anti_pt.delta_expected_nll))
            h2_falsified = bool(primary_effect > 0.0 and anti_effect >= primary_effect)
            if h2_falsified:
                logger.warning(
                    "Anti-control |Δexpected_nll| (%.4f; raw=%.4f) >= primary |Δexpected_nll| (%.4f; raw=%.4f) "
                    "at target_layer=%d threshold=%.3g — H2 potentially falsified at this operating point",
                    anti_effect,
                    float(anti_pt.delta_expected_nll),
                    primary_effect,
                    float(primary_pt.delta_expected_nll),
                    int(target_layer),
                    float(t),
                )
                results[int(primary_idx)] = replace(primary_pt, h2_falsified=True)
                results[int(anti_idx)] = replace(anti_pt, h2_falsified=True)

        def _offtarget_selector(prompt_ids: torch.Tensor, target_span: List[int], seq_len: int) -> Optional[List[int]]:
            return _select_offtarget_span(prompt_ids, target_span, seq_len, buffer=int(token_buffer))

        per_pair, state = _patched_per_pair(
            model=model,
            tokenizer=tokenizer,
            items=items,
            device=device,
            layer=int(target_layer),
            sae_by_layer=sae_by_layer,
            transform_by_layer=transform_by_layer,
            policy=base_policy,
            config=config,
            normalize_by_length=normalize_by_length,
            span_selector=_offtarget_selector,
        )
        _ = _add_result(
            condition="off_target_token",
            threshold=float(t),
            layer=int(target_layer),
            control_layer=None,
            per_pair=per_pair,
            state=state,
        )

        for control_layer in (int(target_layer) - int(off_target_layer_offset), int(target_layer) + int(off_target_layer_offset)):
            if control_layer == int(target_layer):
                continue
            if control_layer < 0:
                continue
            if int(control_layer) not in sae_by_layer or int(control_layer) not in transform_by_layer:
                continue
            per_pair, state = _patched_per_pair(
                model=model,
                tokenizer=tokenizer,
                items=items,
                device=device,
                layer=int(control_layer),
                sae_by_layer=sae_by_layer,
                transform_by_layer=transform_by_layer,
                policy=base_policy,
                config=config,
                normalize_by_length=normalize_by_length,
                span_selector=None,
            )
            _add_result(
                condition="off_target_layer",
                threshold=float(t),
                layer=int(target_layer),
                control_layer=int(control_layer),
                per_pair=per_pair,
                state=state,
            )

    return baseline, results
