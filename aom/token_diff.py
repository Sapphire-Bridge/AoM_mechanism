from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DivergenceSpan:
    """
    Minimal contiguous divergence region for two token sequences.

    Spans are half-open: [start, end).
    """

    a_start: int
    a_end: int
    b_start: int
    b_end: int
    a_len_total: int
    b_len_total: int

    @property
    def a_len(self) -> int:
        return int(self.a_end - self.a_start)

    @property
    def b_len(self) -> int:
        return int(self.b_end - self.b_start)

    @property
    def total_len_equal(self) -> bool:
        return int(self.a_len_total) == int(self.b_len_total)


def divergence_span(a: Sequence[int], b: Sequence[int]) -> DivergenceSpan | None:
    """
    Compute the minimal contiguous token-span that covers all differences between sequences.

    Returns None when sequences are identical.
    """
    a_list = list(int(x) for x in a)
    b_list = list(int(x) for x in b)
    sm = difflib.SequenceMatcher(a=a_list, b=b_list, autojunk=False)
    opcodes = sm.get_opcodes()
    changed = [op for op in opcodes if op[0] != "equal"]
    if not changed:
        return None

    a_start = min(int(i1) for _tag, i1, _i2, _j1, _j2 in changed)
    a_end = max(int(i2) for _tag, _i1, i2, _j1, _j2 in changed)
    b_start = min(int(j1) for _tag, _i1, _i2, j1, _j2 in changed)
    b_end = max(int(j2) for _tag, _i1, _i2, _j1, j2 in changed)

    return DivergenceSpan(
        a_start=int(a_start),
        a_end=int(a_end),
        b_start=int(b_start),
        b_end=int(b_end),
        a_len_total=int(len(a_list)),
        b_len_total=int(len(b_list)),
    )

