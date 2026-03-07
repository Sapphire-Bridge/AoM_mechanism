from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class PromptSide:
    prompt: str
    expected_label: str


@dataclass(frozen=True)
class SubstringSpan:
    """
    Identify a patch site by substring match in a prompt.

    occurrence is 0-indexed (0 = first occurrence).
    """

    substr: str
    occurrence: int = 0


@dataclass(frozen=True)
class AuthoritySite:
    """
    Authority language-game patch site.

    We specify separate substring spans for the A and B prompts to support cases
    where the marker text differs across conditions.
    """

    a: SubstringSpan
    b: SubstringSpan


@dataclass(frozen=True)
class AuthorityPair:
    """
    Minimal pair for the authority-disambiguation "language game".

    A and B are matched prompts with different authoritative status (and possibly
    different expected labels).

    `sites` maps site name -> substring spans for each side.
    """

    pair_id: str
    a: PromptSide
    b: PromptSide
    choices: Mapping[str, List[str]]
    sites: Mapping[str, AuthoritySite]
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class DisambPair:
    """
    Minimal pair for lexical/structural disambiguation.

    `choices` maps labels -> list of continuation strings to score as next tokens/phrases.
    """
    pair_id: str
    target: str
    target_occurrence: int
    a: PromptSide
    b: PromptSide
    choices: Mapping[str, List[str]]
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class CounterfactualPair:
    """
    Minimal-pair intervention sensitivity item.

    `base` and `cf` share `choices` but (optionally) differ in expected label.
    """
    item_id: str
    base: PromptSide
    cf: PromptSide
    choices: Mapping[str, List[str]]
    intervention_type: str
    # Optional explicit contrast labels used for AoM-CF shift calculations.
    # Stored in JSONL as a 2-list; interpreted as (base_label, cf_label).
    contrast_labels: Optional[Tuple[str, str]] = None
    # Expected effect of the intervention on the contrast:
    #  - "shift": expected label flips between base and cf
    #  - "invariant": meaning-preserving sham control (no flip)
    #  - "graded": meaning-relevant partial intervention (no flip; expect intermediate shift magnitude)
    expected_effect: str = "shift"
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class CoherenceItem:
    item_id: str
    context: str
    valid_continuations: List[str]
    invalid_continuations: List[str]
    constraint_type: str
    # Optional grouping for controls (e.g., "main", "ablate_relevant", "ablate_irrelevant").
    group: str = "main"
    metadata: Optional[Dict[str, Any]] = None
