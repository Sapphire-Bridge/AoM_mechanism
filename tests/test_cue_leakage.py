from __future__ import annotations

from aom.data.schemas import DisambPair, PromptSide
from aom.metrics.cue_leakage import cue_stats_for_disamb_pair


def test_cue_stats_flags_asymmetric_ge2_overlap() -> None:
    pair = DisambPair(
        pair_id="p0",
        target="bank",
        target_occurrence=1,
        a=PromptSide(prompt="Alice went to the bank to deposit money.", expected_label="financial"),
        b=PromptSide(prompt="Alice sat on the bank of the river and watched water.", expected_label="river"),
        choices={
            "financial": ["deposit money", "loan"],
            "river": ["shore", "river"],
        },
        metadata=None,
    )
    stats = cue_stats_for_disamb_pair(pair)
    assert stats.a_shared_count == 2
    assert stats.b_shared_count == 1
    assert stats.asymmetric_flag_ge2 is True


def test_cue_stats_balanced_when_both_sides_ge2() -> None:
    pair = DisambPair(
        pair_id="p1",
        target="bank",
        target_occurrence=1,
        a=PromptSide(prompt="Alice went to the bank to deposit money.", expected_label="financial"),
        b=PromptSide(prompt="Alice sat on the bank of the river and watched water.", expected_label="river"),
        choices={
            "financial": ["deposit money", "loan"],
            "river": ["river watched water", "shore"],
        },
        metadata=None,
    )
    stats = cue_stats_for_disamb_pair(pair)
    assert stats.a_shared_count >= 2
    assert stats.b_shared_count >= 2
    assert stats.asymmetric_flag_ge2 is False

