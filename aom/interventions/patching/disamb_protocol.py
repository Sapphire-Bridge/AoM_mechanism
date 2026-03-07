from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from aom.data.schemas import DisambPair
from aom.token_spans import token_span_for_substring

from .base import ActivationPatchingProtocol, CaseSkip, PatchingCase


@dataclass(frozen=True)
class DISAMBPatchingConfig:
    require_token_id_match: bool = True


class DISAMBContextSwapProtocol(ActivationPatchingProtocol):
    """
    DISAMB context-swap protocol.

    Receiver: one side of the pair
    Donor: opposite side
    Patch site: target substring span
    Effect: donor-directed increase in expected-label margin
    """

    name = "disamb_context_swap"

    def __init__(self, *, config: DISAMBPatchingConfig | None = None):
        self.config = DISAMBPatchingConfig() if config is None else config

    def build_cases(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        items: List[Any],
        device: torch.device,
    ) -> Tuple[List[PatchingCase], List[CaseSkip]]:
        cases: List[PatchingCase] = []
        skips: List[CaseSkip] = []
        for raw in items:
            if not isinstance(raw, DisambPair):
                skips.append(CaseSkip(case_id=str(getattr(raw, "pair_id", "disamb_pair")), reason="type_mismatch"))
                continue
            it: DisambPair = raw
            for tag, donor, recv in (
                ("a_to_b", it.a, it.b),
                ("b_to_a", it.b, it.a),
            ):
                donor_span, donor_tok = token_span_for_substring(tokenizer, donor.prompt, it.target, it.target_occurrence)
                recv_span, recv_tok = token_span_for_substring(tokenizer, recv.prompt, it.target, it.target_occurrence)
                if len(donor_span) != len(recv_span):
                    skips.append(CaseSkip(case_id=f"{it.pair_id}__{tag}", reason="span_len_mismatch"))
                    continue
                if bool(self.config.require_token_id_match) and donor_tok != recv_tok:
                    skips.append(CaseSkip(case_id=f"{it.pair_id}__{tag}", reason="token_id_mismatch"))
                    continue
                if not donor_span or not recv_span:
                    skips.append(CaseSkip(case_id=f"{it.pair_id}__{tag}", reason="empty_span"))
                    continue

                donor_ids = tokenizer(donor.prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                recv_ids = tokenizer(recv.prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                cases.append(
                    PatchingCase(
                        case_id=f"{it.pair_id}__{tag}",
                        receiver_prompt=str(recv.prompt),
                        donor_prompt=str(donor.prompt),
                        receiver_ids=recv_ids,
                        donor_ids=donor_ids,
                        receiver_span=tuple(int(x) for x in recv_span),
                        donor_span=tuple(int(x) for x in donor_span),
                        choices=it.choices,
                        expected_label=str(donor.expected_label),
                        strata={"direction": str(tag)},
                        label_aggregation="logmeanexp",
                        effect_sign=1.0,
                    )
                )
        return cases, skips
