from __future__ import annotations

import math
from typing import Dict, List, Literal, Mapping

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aom.interventions.activation_patching import (
    PatchSpanSite,
    forward_with_patched_block_output_span,
    prefill_with_patched_block_output_span,
)
from aom.utils import (
    get_logprob_computation_config,
    get_scoring_performance_config,
    logprob_of_continuation_candidates_shared_prompt,
    logprob_of_continuation_candidates_with_prefill,
)


LabelAggregation = Literal["logmeanexp", "mean"]


def _encode(tokenizer: PreTrainedTokenizerBase, text: str, device: torch.device) -> torch.Tensor:
    enc = tokenizer(str(text), return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"].to(device)


def _logmeanexp(xs: List[float]) -> float:
    if len(xs) < 1:
        raise ValueError("logmeanexp requires at least one value")
    t = torch.tensor(list(xs), dtype=torch.float64)
    return float(torch.logsumexp(t, dim=0) - math.log(len(xs)))


def _aggregate(vals: List[float], *, agg: LabelAggregation) -> float:
    if len(vals) < 1:
        raise ValueError("aggregation requires at least one value")
    if agg == "logmeanexp":
        return float(_logmeanexp(vals))
    if agg == "mean":
        return float(sum(float(x) for x in vals) / max(1, len(vals)))
    raise ValueError(f"Unknown label aggregation: {agg!r}")


@torch.no_grad()
def score_labels_next_continuations_ids(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,
    choices: Mapping[str, List[str]],
    device: torch.device,
    normalize_by_length: bool,
    label_aggregation: LabelAggregation,
) -> Dict[str, float]:
    score_batch_size, use_prefix_cache = get_scoring_performance_config()
    pad_token_id = (
        int(tokenizer.pad_token_id)
        if tokenizer.pad_token_id is not None
        else (int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else 0)
    )
    flat_conts: List[torch.Tensor] = []
    spans: Dict[str, tuple[int, int]] = {}
    for label, continuations in choices.items():
        if len(continuations) < 1:
            raise ValueError(f"Empty continuation list for label={label}")
        start = len(flat_conts)
        encoded = [_encode(tokenizer, str(cont), device=device).squeeze(0) for cont in continuations]
        flat_conts.extend(encoded)
        spans[str(label)] = (start, len(flat_conts))

    flat_logps = logprob_of_continuation_candidates_shared_prompt(
        model=model,
        prompt_ids=prompt_ids,
        continuation_id_list=flat_conts,
        normalize_by_length=bool(normalize_by_length),
        batch_size=int(score_batch_size),
        pad_token_id=int(pad_token_id),
        use_prefix_cache=bool(use_prefix_cache),
    )

    scores: Dict[str, float] = {}
    for label in spans.keys():
        start, end = spans[str(label)]
        vals = [float(x) for x in flat_logps[start:end].tolist()]
        scores[str(label)] = float(_aggregate(vals, agg=label_aggregation))
    return scores


@torch.no_grad()
def score_labels_next_continuations_patched_ids(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,
    choices: Mapping[str, List[str]],
    device: torch.device,
    patch_site: PatchSpanSite,
    replacement: torch.Tensor,
    normalize_by_length: bool,
    label_aggregation: LabelAggregation,
) -> Dict[str, float]:
    """
    Score choices under a patched forward pass (block output patched at `patch_site`).

    This re-runs the patched model per continuation for clarity and reproducibility.
    """
    logprobs_dtype, strict_finite = get_logprob_computation_config()
    score_batch_size, use_prefix_cache = get_scoring_performance_config()
    pad_token_id = (
        int(tokenizer.pad_token_id)
        if tokenizer.pad_token_id is not None
        else (int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else 0)
    )
    flat_conts: List[torch.Tensor] = []
    spans: Dict[str, tuple[int, int]] = {}
    for label, continuations in choices.items():
        if len(continuations) < 1:
            raise ValueError(f"Empty continuation list for label={label}")
        start = len(flat_conts)
        encoded = [_encode(tokenizer, str(cont), device=device).squeeze(0) for cont in continuations]
        flat_conts.extend(encoded)
        spans[str(label)] = (start, len(flat_conts))

    flat_logps: torch.Tensor | None = None
    if bool(use_prefix_cache):
        try:
            prefill_out = prefill_with_patched_block_output_span(
                model,
                input_ids=prompt_ids,
                site=patch_site,
                replacement=replacement,
                attention_mask=torch.ones_like(prompt_ids),
            )
            flat_logps = logprob_of_continuation_candidates_with_prefill(
                model=model,
                prompt_ids=prompt_ids,
                continuation_id_list=flat_conts,
                prefill_logits_last=prefill_out.logits[:, -1, :],  # type: ignore[attr-defined]
                prefill_past_key_values=prefill_out.past_key_values,  # type: ignore[attr-defined]
                normalize_by_length=bool(normalize_by_length),
                batch_size=int(score_batch_size),
                pad_token_id=int(pad_token_id),
            )
        except Exception:
            # Preserve correctness on cache incompatibilities by falling back
            # to the full-forward batched path.
            flat_logps = None
    if flat_logps is None:
        P = int(prompt_ids.size(1))
        flat_scores: List[torch.Tensor] = []
        chunk_size = int(max(1, score_batch_size))
        for start in range(0, len(flat_conts), chunk_size):
            batch_conts = flat_conts[start : start + chunk_size]
            B = int(len(batch_conts))
            lengths = torch.tensor([int(c.numel()) for c in batch_conts], device=device, dtype=torch.long)
            max_len = int(lengths.max().item())
            cont_pad = torch.full((B, max_len), int(pad_token_id), device=device, dtype=torch.long)
            cont_mask = torch.zeros((B, max_len), device=device, dtype=torch.bool)
            for i, c in enumerate(batch_conts):
                L = int(c.numel())
                cont_pad[i, :L] = c
                cont_mask[i, :L] = True
            full_ids = torch.cat([prompt_ids.expand(B, -1), cont_pad], dim=1)
            attn_mask = torch.cat(
                [
                    torch.ones((B, P), device=device, dtype=torch.long),
                    cont_mask.to(dtype=torch.long),
                ],
                dim=1,
            )
            logits = forward_with_patched_block_output_span(
                model,
                input_ids=full_ids,
                site=patch_site,
                replacement=replacement,
                attention_mask=attn_mask,
            )
            logits_slice = logits[:, P - 1 : P + max_len - 1, :].to(dtype=logprobs_dtype)
            log_probs = torch.log_softmax(logits_slice, dim=-1)
            gathered = log_probs.gather(2, cont_pad.unsqueeze(-1)).squeeze(-1)
            if not torch.isfinite(gathered[cont_mask]).all():
                if strict_finite:
                    raise FloatingPointError("Non-finite log-probability detected during patched scoring.")
                gathered = torch.where(torch.isfinite(gathered), gathered, torch.full_like(gathered, -1e9))
            gathered = torch.where(cont_mask, gathered, torch.zeros_like(gathered))
            lp = gathered.sum(dim=1)
            if bool(normalize_by_length):
                lp = lp / lengths.to(dtype=lp.dtype)
            flat_scores.append(lp)
        flat_logps = torch.cat(flat_scores, dim=0) if flat_scores else torch.empty((0,), device=device, dtype=logprobs_dtype)

    scores: Dict[str, float] = {}
    for label in spans.keys():
        start, end = spans[str(label)]
        vals = [float(x) for x in flat_logps[start:end].tolist()]
        scores[str(label)] = float(_aggregate(vals, agg=label_aggregation))
    return scores
