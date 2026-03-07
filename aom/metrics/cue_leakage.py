from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from aom.data.schemas import DisambPair, PromptSide


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")

# Small, explicit stopword list (extend if needed; keep deterministic across environments).
_STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "so",
    "that",
    "the",
    "their",
    "theirs",
    "then",
    "there",
    "therefore",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
    "yours",
}


def _content_tokens(text: str) -> list[str]:
    toks = [m.group(0).lower() for m in _WORD_RE.finditer(str(text))]
    return [t for t in toks if t and (t not in _STOPWORDS)]


def _content_set_excluding_target(*, prompt: str, target: str) -> set[str]:
    target_lc = str(target).lower().strip()
    out = set(_content_tokens(prompt))
    if target_lc:
        out.discard(target_lc)
    return out


def _continuation_token_set(continuations: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for c in continuations:
        out.update(_content_tokens(str(c)))
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union > 0 else 0.0


@dataclass(frozen=True)
class CueStats:
    pair_id: str
    target: str
    a_expected_label: str
    b_expected_label: str
    a_shared_count: int
    b_shared_count: int
    a_jaccard: float
    b_jaccard: float
    pair_jaccard_mean: float
    pair_jaccard_max: float
    asymmetric_flag_ge2: bool


def cue_stats_for_disamb_pair(pair: DisambPair) -> CueStats:
    """
    Compute a lightweight cue-leakage diagnostic for a DISAMB minimal pair.

    Protocol (per reviewer-robust guidance):
    - Tokenize context content words (excluding the ambiguous target token).
    - Tokenize each sense's continuation content words (union across continuations).
    - Measure overlap between each context and its *expected* sense continuations.
    - Flag asymmetric cueing when exactly one side has >=2 shared content words.
    """
    target = str(pair.target)
    choices: Mapping[str, list[str]] = {str(k): list(v) for k, v in pair.choices.items()}

    def _side_stats(side: PromptSide) -> tuple[int, float]:
        ctx = _content_set_excluding_target(prompt=str(side.prompt), target=target)
        conts = choices.get(str(side.expected_label), [])
        cont_set = _continuation_token_set(conts)
        shared = len(ctx & cont_set)
        return int(shared), float(_jaccard(ctx, cont_set))

    a_shared, a_j = _side_stats(pair.a)
    b_shared, b_j = _side_stats(pair.b)
    asym = bool((a_shared >= 2) != (b_shared >= 2))
    mean_j = 0.5 * (float(a_j) + float(b_j))
    max_j = float(max(float(a_j), float(b_j)))

    return CueStats(
        pair_id=str(pair.pair_id),
        target=str(target),
        a_expected_label=str(pair.a.expected_label),
        b_expected_label=str(pair.b.expected_label),
        a_shared_count=int(a_shared),
        b_shared_count=int(b_shared),
        a_jaccard=float(a_j),
        b_jaccard=float(b_j),
        pair_jaccard_mean=float(mean_j),
        pair_jaccard_max=float(max_j),
        asymmetric_flag_ge2=bool(asym),
    )

