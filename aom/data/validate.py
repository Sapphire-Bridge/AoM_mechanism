from __future__ import annotations

from typing import Mapping, Sequence

from .schemas import AuthorityPair, CoherenceItem, CounterfactualPair, DisambPair


def _validate_prompt_cont_boundary(*, prompt: str, continuation: str, item_id: str, field: str) -> None:
    if not prompt:
        raise ValueError(f"{item_id}: empty prompt/context for {field}")
    if not continuation:
        raise ValueError(f"{item_id}: empty continuation in {field}")
    if (not prompt[-1].isspace()) and (not continuation[0].isspace()):
        raise ValueError(
            f"{item_id}: continuation boundary likely wrong in {field} "
            f"(prompt does not end with whitespace and continuation does not start with whitespace)"
        )


def _validate_choices(
    *,
    choices: Mapping[str, Sequence[str]],
    expected_labels: Sequence[str],
    prompts: Sequence[str],
    item_id: str,
    field: str,
    require_equal_counts: bool,
) -> None:
    if not choices:
        raise ValueError(f"{item_id}: empty choices for {field}")

    for lab in expected_labels:
        if lab not in choices:
            raise ValueError(f"{item_id}: expected_label={lab!r} missing from choices for {field}")

    counts = set()
    for label, conts in choices.items():
        if not isinstance(label, str) or not label:
            raise ValueError(f"{item_id}: invalid label in choices for {field}: {label!r}")
        if not conts:
            raise ValueError(f"{item_id}: empty continuation list for label={label!r} in {field}")
        counts.add(len(conts))
        for cont in conts:
            if not isinstance(cont, str):
                raise ValueError(f"{item_id}: non-string continuation for label={label!r} in {field}")
            for prompt in prompts:
                _validate_prompt_cont_boundary(prompt=prompt, continuation=cont, item_id=item_id, field=field)

    if require_equal_counts and len(counts) > 1:
        raise ValueError(f"{item_id}: unequal continuation counts per label in {field}: {sorted(counts)}")


def validate_disamb_pairs(items: Sequence[DisambPair], *, require_equal_choice_counts: bool = True) -> None:
    for it in items:
        _validate_choices(
            choices=it.choices,
            expected_labels=(it.a.expected_label, it.b.expected_label),
            prompts=(it.a.prompt, it.b.prompt),
            item_id=str(it.pair_id),
            field="disamb.choices",
            require_equal_counts=require_equal_choice_counts,
        )


def _find_nth(haystack: str, needle: str, n: int) -> int:
    if n < 0:
        return -1
    idx = -1
    start = 0
    for _ in range(n + 1):
        idx = haystack.find(needle, start)
        if idx < 0:
            return -1
        start = idx + len(needle)
    return idx


def validate_authority_pairs(items: Sequence[AuthorityPair], *, require_equal_choice_counts: bool = True) -> None:
    for it in items:
        _validate_choices(
            choices=it.choices,
            expected_labels=(it.a.expected_label, it.b.expected_label),
            prompts=(it.a.prompt, it.b.prompt),
            item_id=str(it.pair_id),
            field="authority.choices",
            require_equal_counts=require_equal_choice_counts,
        )
        if not it.sites:
            raise ValueError(f"{it.pair_id}: empty sites for authority experiment")
        for name, site in it.sites.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{it.pair_id}: invalid site name in sites: {name!r}")
            if not str(site.a.substr).strip() or not str(site.b.substr).strip():
                raise ValueError(f"{it.pair_id}: empty substring for site={name!r}")
            if int(site.a.occurrence) < 0 or int(site.b.occurrence) < 0:
                raise ValueError(f"{it.pair_id}: negative occurrence for site={name!r}")
            if _find_nth(str(it.a.prompt), str(site.a.substr), int(site.a.occurrence)) < 0:
                raise ValueError(
                    f"{it.pair_id}: site={name!r} substring not found in A prompt (substr={site.a.substr!r}, occurrence={int(site.a.occurrence)})"
                )
            if _find_nth(str(it.b.prompt), str(site.b.substr), int(site.b.occurrence)) < 0:
                raise ValueError(
                    f"{it.pair_id}: site={name!r} substring not found in B prompt (substr={site.b.substr!r}, occurrence={int(site.b.occurrence)})"
                )


def validate_counterfactual_pairs(items: Sequence[CounterfactualPair], *, require_equal_choice_counts: bool = True) -> None:
    for it in items:
        if it.expected_effect not in {"shift", "invariant", "graded"}:
            raise ValueError(
                f"{it.item_id}: invalid expected_effect={it.expected_effect!r} (expected 'shift'|'invariant'|'graded')"
            )
        if it.expected_effect == "shift" and it.base.expected_label == it.cf.expected_label:
            raise ValueError(
                f"{it.item_id}: shift item must satisfy base.expected_label!=cf.expected_label "
                f"(got {it.base.expected_label!r})"
            )

        if it.contrast_labels is None:
            if it.expected_effect in {"invariant", "graded"}:
                raise ValueError(f"{it.item_id}: {it.expected_effect} CF item requires contrast_labels")
        else:
            L0, L1 = it.contrast_labels
            if L0 == L1:
                raise ValueError(f"{it.item_id}: contrast_labels must differ (got {it.contrast_labels!r})")
            if L0 not in it.choices or L1 not in it.choices:
                raise ValueError(f"{it.item_id}: contrast_labels not found in choices: {it.contrast_labels!r}")
            if it.expected_effect == "shift":
                if it.base.expected_label != L0 or it.cf.expected_label != L1:
                    raise ValueError(
                        f"{it.item_id}: shift item must satisfy base.expected_label==contrast_labels[0] "
                        f"and cf.expected_label==contrast_labels[1] "
                        f"(got base={it.base.expected_label!r} cf={it.cf.expected_label!r} contrast={it.contrast_labels!r})"
                    )
            elif it.expected_effect in {"invariant", "graded"}:
                if it.base.expected_label != it.cf.expected_label:
                    raise ValueError(
                        f"{it.item_id}: {it.expected_effect} item must satisfy base.expected_label==cf.expected_label "
                        f"(got base={it.base.expected_label!r} cf={it.cf.expected_label!r})"
                    )
                if it.base.expected_label != L0:
                    raise ValueError(
                        f"{it.item_id}: {it.expected_effect} item must satisfy base.expected_label==contrast_labels[0] "
                        f"(got base={it.base.expected_label!r} contrast={it.contrast_labels!r})"
                    )

        _validate_choices(
            choices=it.choices,
            expected_labels=(it.base.expected_label, it.cf.expected_label),
            prompts=(it.base.prompt, it.cf.prompt),
            item_id=str(it.item_id),
            field="cf.choices",
            require_equal_counts=require_equal_choice_counts,
        )


def validate_coherence_items(items: Sequence[CoherenceItem]) -> None:
    for it in items:
        if not it.valid_continuations:
            raise ValueError(f"{it.item_id}: empty valid_continuations")
        if not it.invalid_continuations:
            raise ValueError(f"{it.item_id}: empty invalid_continuations")
        for cont in list(it.valid_continuations) + list(it.invalid_continuations):
            _validate_prompt_cont_boundary(prompt=it.context, continuation=cont, item_id=str(it.item_id), field="coh")


def validate_evidence_metadata(*, metadata: object, item_id: str) -> None:
    """
    Validate optional metadata-first evidence annotations.

    Supported keys:
    - query_pos: int
    - evidence_spans: list[[start, end] | {"start": int, "end": int}]
      where span is half-open [start, end) in token indices.
    """
    if metadata is None:
        return
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{item_id}: metadata must be a mapping when evidence annotations are present")

    if "query_pos" in metadata:
        q = metadata.get("query_pos", None)
        if not isinstance(q, int):
            raise ValueError(f"{item_id}: metadata.query_pos must be int")

    if "evidence_spans" in metadata:
        spans = metadata.get("evidence_spans", None)
        if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes, bytearray)):
            raise ValueError(f"{item_id}: metadata.evidence_spans must be a list")
        for idx, sp in enumerate(spans):
            if isinstance(sp, Mapping):
                a = sp.get("start", None)
                b = sp.get("end", None)
            elif isinstance(sp, Sequence) and not isinstance(sp, (str, bytes, bytearray)) and len(sp) == 2:
                a, b = sp[0], sp[1]
            else:
                raise ValueError(f"{item_id}: invalid evidence span at index {idx}: {sp!r}")
            if not isinstance(a, int) or not isinstance(b, int):
                raise ValueError(f"{item_id}: evidence span indices must be ints at index {idx}")
            if a < 0 or b < 0 or b <= a:
                raise ValueError(f"{item_id}: evidence span must satisfy 0 <= start < end at index {idx}")
