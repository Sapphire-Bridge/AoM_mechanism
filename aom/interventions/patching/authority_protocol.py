from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from aom.data.schemas import AuthorityPair, SubstringSpan
from aom.token_spans import token_span_for_substring

from .base import ActivationPatchingProtocol, CaseSkip, PatchingCase


@dataclass(frozen=True)
class AuthorityPatchingConfig:
    include_sites: Tuple[str, ...] | None = None
    require_token_id_match: bool = False
    # Optional span reduction control (to avoid "patched more tokens" confounds).
    # If set, spans longer than this are truncated to at most this many tokens.
    max_tokens_per_site: int | None = None
    span_take: Literal["first", "last"] = "last"


def _span_for_side(
    *,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    spec: SubstringSpan,
) -> tuple[list[int], list[int]]:
    return token_span_for_substring(
        tokenizer,
        str(prompt),
        str(spec.substr),
        int(spec.occurrence),
    )


def _reduce_span(
    span: list[int],
    token_ids: list[int],
    *,
    max_tokens: int | None,
    span_take: str,
) -> tuple[list[int], list[int]]:
    if max_tokens is None:
        return span, token_ids
    k = int(max_tokens)
    if k <= 0:
        raise ValueError("max_tokens_per_site must be positive when provided")
    if len(span) <= k:
        return span, token_ids
    if span_take == "first":
        return span[:k], token_ids[:k]
    if span_take == "last":
        return span[-k:], token_ids[-k:]
    raise ValueError(f"Unknown span_take={span_take!r}")


def _decode_tokens(tokenizer: PreTrainedTokenizerBase, token_ids: list[int]) -> str:
    text = str(tokenizer.decode([int(x) for x in token_ids], clean_up_tokenization_spaces=False))
    return text.replace("\n", "\\n")


class AuthorityLanguageGameProtocol(ActivationPatchingProtocol):
    """
    Authority language-game protocol.

    Receiver: one side of the pair.
    Donor: opposite side.
    Patch site: substring-defined span (marker / imperative / distractor).
    Effect: donor-directed increase in expected-label margin.
    """

    name = "authority_language_game"

    def __init__(self, *, config: AuthorityPatchingConfig | None = None) -> None:
        self.config = AuthorityPatchingConfig() if config is None else config

    def build_cases(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        items: List[Any],
        device: torch.device,
    ) -> tuple[List[PatchingCase], List[CaseSkip]]:
        cases: List[PatchingCase] = []
        skips: List[CaseSkip] = []

        include = None
        if self.config.include_sites is not None:
            include = {str(x) for x in self.config.include_sites}

        for raw in items:
            if not isinstance(raw, AuthorityPair):
                skips.append(CaseSkip(case_id=str(getattr(raw, "pair_id", "authority_pair")), reason="type_mismatch"))
                continue
            it: AuthorityPair = raw

            for tag, donor, recv, donor_key, recv_key in (
                ("a_to_b", it.a, it.b, "a", "b"),
                ("b_to_a", it.b, it.a, "b", "a"),
            ):
                donor_ids = tokenizer(donor.prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                recv_ids = tokenizer(recv.prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

                for site_name, site in sorted(it.sites.items(), key=lambda kv: str(kv[0])):
                    if include is not None and str(site_name) not in include:
                        continue

                    donor_spec = site.a if donor_key == "a" else site.b
                    recv_spec = site.a if recv_key == "a" else site.b
                    try:
                        donor_span, donor_tok = _span_for_side(
                            tokenizer=tokenizer,
                            prompt=str(donor.prompt),
                            spec=donor_spec,
                        )
                        recv_span, recv_tok = _span_for_side(
                            tokenizer=tokenizer,
                            prompt=str(recv.prompt),
                            spec=recv_spec,
                        )
                    except Exception as e:
                        skips.append(
                            CaseSkip(
                                case_id=f"{it.pair_id}__{tag}__{site_name}",
                                reason=f"span_error:{type(e).__name__}",
                            )
                        )
                        continue

                    if not donor_span or not recv_span:
                        skips.append(
                            CaseSkip(case_id=f"{it.pair_id}__{tag}__{site_name}", reason="empty_span")
                        )
                        continue

                    donor_span, donor_tok = _reduce_span(
                        donor_span,
                        donor_tok,
                        max_tokens=self.config.max_tokens_per_site,
                        span_take=str(self.config.span_take),
                    )
                    recv_span, recv_tok = _reduce_span(
                        recv_span,
                        recv_tok,
                        max_tokens=self.config.max_tokens_per_site,
                        span_take=str(self.config.span_take),
                    )
                    if len(donor_span) != len(recv_span):
                        reason = (
                            "span_len_mismatch"
                            if self.config.max_tokens_per_site is None
                            else "span_len_mismatch_reduced"
                        )
                        skips.append(
                            CaseSkip(case_id=f"{it.pair_id}__{tag}__{site_name}", reason=str(reason))
                        )
                        continue

                    if bool(self.config.require_token_id_match) and donor_tok != recv_tok:
                        skips.append(CaseSkip(case_id=f"{it.pair_id}__{tag}__{site_name}", reason="token_id_mismatch"))
                        continue

                    cases.append(
                        PatchingCase(
                            case_id=f"{it.pair_id}__{tag}__{site_name}",
                            receiver_prompt=str(recv.prompt),
                            donor_prompt=str(donor.prompt),
                            receiver_ids=recv_ids,
                            donor_ids=donor_ids,
                            receiver_span=tuple(int(x) for x in recv_span),
                            donor_span=tuple(int(x) for x in donor_span),
                            choices=it.choices,
                            expected_label=str(donor.expected_label),
                            strata={
                                "direction": str(tag),
                                "site": str(site_name),
                                "patched_text": _decode_tokens(tokenizer, recv_tok),
                            },
                            label_aggregation="logmeanexp",
                            effect_sign=1.0,
                        )
                    )

        return cases, skips
