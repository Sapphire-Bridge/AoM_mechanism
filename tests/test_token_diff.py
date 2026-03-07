from __future__ import annotations

from aom.token_diff import divergence_span


def test_divergence_span_identical_returns_none() -> None:
    assert divergence_span([1, 2, 3], [1, 2, 3]) is None


def test_divergence_span_substitution() -> None:
    span = divergence_span([1, 2, 3, 4], [1, 2, 9, 4])
    assert span is not None
    assert (span.a_start, span.a_end) == (2, 3)
    assert (span.b_start, span.b_end) == (2, 3)
    assert span.total_len_equal is True
    assert span.a_len == 1
    assert span.b_len == 1


def test_divergence_span_insertion() -> None:
    span = divergence_span([1, 2, 3, 4], [1, 2, 3, 8, 4])
    assert span is not None
    assert (span.a_start, span.a_end) == (3, 3)
    assert (span.b_start, span.b_end) == (3, 4)
    assert span.total_len_equal is False
    assert span.a_len == 0
    assert span.b_len == 1


def test_divergence_span_deletion() -> None:
    span = divergence_span([1, 2, 3, 4], [1, 2, 4])
    assert span is not None
    assert (span.a_start, span.a_end) == (2, 3)
    assert (span.b_start, span.b_end) == (2, 2)
    assert span.total_len_equal is False
    assert span.a_len == 1
    assert span.b_len == 0

