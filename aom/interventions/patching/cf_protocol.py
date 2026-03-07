from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from aom.data.schemas import CounterfactualPair
from aom.token_diff import divergence_span

from .base import ActivationPatchingProtocol, CaseSkip, ComparisonSpec, PatchingCase


@dataclass(frozen=True)
class CFPatchingConfig:
    max_total_len_delta: int = 5
    include_expected_effects: Tuple[str, ...] = ("shift", "invariant")
    span_mode: Literal["divergent_only", "divergent_plus_downstream", "left_aligned_truncated"] = "divergent_only"


class CFInterventionSwapProtocol(ActivationPatchingProtocol):
    """
    Context/intervention-swap activation patching for AoM-CF.

    Receiver: base prompt (x)
    Donor: intervention prompt (x')
    Patch site: minimal contiguous divergent token span (token-diff)
    Effect: donor-directed increase in margin for donor expected label
    """

    name = "cf_intervention_swap"

    def __init__(self, *, config: CFPatchingConfig | None = None):
        self.config = CFPatchingConfig() if config is None else config

    def primary_comparisons(self) -> Sequence[ComparisonSpec]:
        return (
            ComparisonSpec(
                name="expected_effect_shift_vs_invariant",
                stratum_key="expected_effect",
                a_value="shift",
                b_value="invariant",
            ),
        )

    def build_cases(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        items: Sequence[Any],
        device: torch.device,
    ) -> tuple[List[PatchingCase], List[CaseSkip]]:
        cases: List[PatchingCase] = []
        skips: List[CaseSkip] = []

        for raw in items:
            if not isinstance(raw, CounterfactualPair):
                skips.append(CaseSkip(case_id=str(getattr(raw, "item_id", "cf_item")), reason="type_mismatch"))
                continue
            it: CounterfactualPair = raw
            if str(it.expected_effect) not in set(self.config.include_expected_effects):
                skips.append(CaseSkip(case_id=str(it.item_id), reason=f"excluded_expected_effect:{it.expected_effect}"))
                continue

            recv_text = str(it.base.prompt)
            donor_text = str(it.cf.prompt)
            recv_ids = tokenizer(recv_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            donor_ids = tokenizer(donor_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

            span = divergence_span(recv_ids[0].tolist(), donor_ids[0].tolist())
            if span is None:
                skips.append(CaseSkip(case_id=str(it.item_id), reason="no_divergence"))
                continue

            total_len_delta = abs(int(span.a_len_total) - int(span.b_len_total))
            if total_len_delta > int(self.config.max_total_len_delta):
                skips.append(CaseSkip(case_id=str(it.item_id), reason=f"total_len_delta>{int(self.config.max_total_len_delta)}"))
                continue

            if int(span.a_len) != int(span.b_len):
                if self.config.span_mode != "left_aligned_truncated":
                    skips.append(CaseSkip(case_id=str(it.item_id), reason="divergent_span_len_mismatch"))
                    continue

            if self.config.span_mode == "divergent_plus_downstream":
                if not span.total_len_equal:
                    skips.append(CaseSkip(case_id=str(it.item_id), reason="span_mode_requires_equal_total_len"))
                    continue
                recv_span = tuple(range(int(span.a_start), int(span.a_len_total)))
                donor_span = tuple(range(int(span.b_start), int(span.b_len_total)))
            elif self.config.span_mode == "left_aligned_truncated":
                patch_len = min(int(span.a_len), int(span.b_len))
                recv_span = tuple(range(int(span.a_start), int(span.a_start) + patch_len))
                donor_span = tuple(range(int(span.b_start), int(span.b_start) + patch_len))
            else:
                recv_span = tuple(range(int(span.a_start), int(span.a_end)))
                donor_span = tuple(range(int(span.b_start), int(span.b_end)))
            if not recv_span or not donor_span:
                skips.append(CaseSkip(case_id=str(it.item_id), reason="empty_span"))
                continue

            cases.append(
                PatchingCase(
                    case_id=str(it.item_id),
                    receiver_prompt=recv_text,
                    donor_prompt=donor_text,
                    receiver_ids=recv_ids,
                    donor_ids=donor_ids,
                    receiver_span=tuple(int(i) for i in recv_span),
                    donor_span=tuple(int(i) for i in donor_span),
                    choices=it.choices,
                    expected_label=str(it.cf.expected_label),
                    strata={
                        "expected_effect": str(it.expected_effect),
                        "intervention_type": str(it.intervention_type),
                    },
                    label_aggregation="logmeanexp",
                    effect_sign=1.0,
                )
            )

        return cases, skips
