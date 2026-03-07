"""Tests for CFInterventionSwapProtocol span_mode and include_expected_effects."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from aom.data.schemas import CounterfactualPair, PromptSide
from aom.interventions.patching.cf_protocol import CFInterventionSwapProtocol, CFPatchingConfig


def _make_tokenizer(vocab: dict[str, int] | None = None):
    """Return a mock tokenizer whose __call__ produces deterministic token ids."""
    if vocab is None:
        vocab = {}
    tok = MagicMock()

    def _tokenize(text, return_tensors="pt", add_special_tokens=False):
        # Deterministic: each character → its ordinal.  Crude but sufficient
        # for testing span arithmetic.
        ids = [ord(c) for c in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    tok.side_effect = _tokenize
    tok.__call__ = _tokenize
    return tok


def _make_cf_item(
    item_id: str = "cf-0",
    base_prompt: str = "AABCC",
    cf_prompt: str = "AAXCC",
    expected_effect: str = "shift",
) -> CounterfactualPair:
    return CounterfactualPair(
        item_id=item_id,
        base=PromptSide(prompt=base_prompt, expected_label="a"),
        cf=PromptSide(prompt=cf_prompt, expected_label="b"),
        choices={"a": [" a"], "b": [" b"]},
        intervention_type="test",
        expected_effect=expected_effect,
    )


# ── Fix 1: include_expected_effects ─────────────────────────────────────


class TestIncludeExpectedEffects:
    def test_graded_excluded_by_default(self):
        proto = CFInterventionSwapProtocol()
        item = _make_cf_item(expected_effect="graded")
        cases, skips = proto.build_cases(
            tokenizer=_make_tokenizer(), items=[item], device=torch.device("cpu")
        )
        assert len(cases) == 0
        assert len(skips) == 1
        assert "excluded_expected_effect:graded" in skips[0].reason

    def test_graded_included_when_configured(self):
        cfg = CFPatchingConfig(include_expected_effects=("shift", "invariant", "graded"))
        proto = CFInterventionSwapProtocol(config=cfg)
        item = _make_cf_item(expected_effect="graded")
        cases, skips = proto.build_cases(
            tokenizer=_make_tokenizer(), items=[item], device=torch.device("cpu")
        )
        assert len(cases) == 1
        assert len(skips) == 0
        assert cases[0].strata["expected_effect"] == "graded"

    def test_shift_and_invariant_included_by_default(self):
        proto = CFInterventionSwapProtocol()
        items = [
            _make_cf_item(item_id="s", expected_effect="shift"),
            _make_cf_item(item_id="i", expected_effect="invariant"),
        ]
        cases, skips = proto.build_cases(
            tokenizer=_make_tokenizer(), items=items, device=torch.device("cpu")
        )
        assert len(cases) == 2
        assert len(skips) == 0


# ── Fix 2: left_aligned_truncated span_mode ──────────────────────────────


class TestLeftAlignedTruncated:
    def test_unequal_spans_skipped_in_strict_mode(self):
        """divergent_only skips items where divergent span lengths differ."""
        proto = CFInterventionSwapProtocol(
            config=CFPatchingConfig(span_mode="divergent_only")
        )
        # "AABCC" vs "AAXYCC" — divergent span is 1 vs 2 tokens
        item = _make_cf_item(base_prompt="AABCC", cf_prompt="AAXYCC")
        cases, skips = proto.build_cases(
            tokenizer=_make_tokenizer(), items=[item], device=torch.device("cpu")
        )
        assert len(cases) == 0
        reasons = [s.reason for s in skips]
        assert any("divergent_span_len_mismatch" in r or "total_len_delta" in r for r in reasons)

    def test_unequal_spans_recovered_in_truncated_mode(self):
        """left_aligned_truncated recovers items with unequal divergent span lengths."""
        cfg = CFPatchingConfig(
            span_mode="left_aligned_truncated",
            max_total_len_delta=10,  # relax so the item isn't skipped for total len
        )
        proto = CFInterventionSwapProtocol(config=cfg)
        # "AABCC" vs "AAXYCC" — divergent region differs in length
        item = _make_cf_item(base_prompt="AABCC", cf_prompt="AAXYCC")
        cases, skips = proto.build_cases(
            tokenizer=_make_tokenizer(), items=[item], device=torch.device("cpu")
        )
        # Should produce a case, not a skip
        assert len(cases) == 1, f"Expected 1 case, got {len(cases)} cases and skips: {[s.reason for s in skips]}"
        case = cases[0]
        # Spans must be equal length (the truncation invariant)
        assert len(case.receiver_span) == len(case.donor_span)
        # Length should be min of the two divergent span lengths
        assert len(case.receiver_span) >= 1

    def test_equal_spans_work_in_truncated_mode(self):
        """left_aligned_truncated also works fine when spans are already equal."""
        cfg = CFPatchingConfig(span_mode="left_aligned_truncated")
        proto = CFInterventionSwapProtocol(config=cfg)
        # "AABCC" vs "AAXCC" — single-char difference, equal span lengths
        item = _make_cf_item(base_prompt="AABCC", cf_prompt="AAXCC")
        cases, skips = proto.build_cases(
            tokenizer=_make_tokenizer(), items=[item], device=torch.device("cpu")
        )
        assert len(cases) == 1
        case = cases[0]
        assert len(case.receiver_span) == len(case.donor_span)
        assert len(case.receiver_span) == 1  # single differing char

    def test_truncated_spans_are_left_aligned(self):
        """Truncated spans start at the divergence start for both receiver and donor."""
        cfg = CFPatchingConfig(
            span_mode="left_aligned_truncated",
            max_total_len_delta=10,
        )
        proto = CFInterventionSwapProtocol(config=cfg)
        # "AABCC" (5 chars) vs "AAXYZCC" (7 chars)
        # Divergence starts at index 2 for both.
        # a span: [2,3) len=1, b span: [2,5) len=3
        # Truncated: both get len=1, starting at their respective starts.
        item = _make_cf_item(base_prompt="AABCC", cf_prompt="AAXYZCC")
        cases, skips = proto.build_cases(
            tokenizer=_make_tokenizer(), items=[item], device=torch.device("cpu")
        )
        assert len(cases) == 1, f"skips: {[s.reason for s in skips]}"
        case = cases[0]
        # Both spans should start at index 2
        assert case.receiver_span[0] == 2
        assert case.donor_span[0] == 2
        # Both spans should have the same (truncated) length
        assert len(case.receiver_span) == len(case.donor_span)
