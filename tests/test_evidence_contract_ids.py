from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_CONTRACT_ID_CELL_RE = re.compile(r"^`([ERC]\d+[a-z]?)`$")
_EVIDENCE_TAG_CONTENT_RE = re.compile(
    r"^\s*[ERC]\d+[a-z]?(?:\s*,\s*[ERC]\d+[a-z]?)*\s*$"
)
_EVIDENCE_ID_RE = re.compile(r"^[ERC]\d+[a-z]?$")
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_TABLE_SEPARATOR_RE = re.compile(r"^\|\s*-{3,}\s*(\|\s*-{3,}\s*)+\|?$")


@dataclass(frozen=True)
class AppendixA2Row:
    evidence_ids: tuple[str, ...]
    artifact_paths: tuple[str, ...]


def _extract_contract_ids(contract_text: str) -> set[str]:
    ids: set[str] = set()
    for line in contract_text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 3:
            continue
        first_cell = cells[1]
        m = _CONTRACT_ID_CELL_RE.match(first_cell)
        if m is not None:
            ids.add(m.group(1))
    return ids


def _extract_paper_cited_ids(paper_text: str) -> set[str]:
    ids: set[str] = set()
    for m in _BRACKET_RE.finditer(paper_text):
        content = m.group(1)
        if _EVIDENCE_TAG_CONTENT_RE.fullmatch(content) is None:
            continue
        for part in content.split(","):
            ids.add(part.strip())
    return ids


def _extract_appendix_a2_section(paper_text: str) -> str:
    marker = "### A.2 Claim-to-artifact map"
    start = paper_text.find(marker)
    if start == -1:
        return ""
    tail = paper_text[start:]
    next_heading = re.search(r"^###\s", tail[len(marker) :], flags=re.MULTILINE)
    if next_heading is None:
        return tail
    return tail[: len(marker) + next_heading.start()]


def _extract_backticked(text: str) -> list[str]:
    return _BACKTICK_RE.findall(text)


def _iter_appendix_a2_rows(section_text: str) -> list[AppendixA2Row]:
    rows: list[AppendixA2Row] = []
    for line in section_text.splitlines():
        if not line.startswith("|"):
            continue
        if _TABLE_SEPARATOR_RE.fullmatch(line.strip()) is not None:
            continue

        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cells or cells[0] == "Evidence ID(s)":
            continue
        if len(cells) < 5:
            continue

        ids = tuple(tok for tok in _extract_backticked(cells[0]) if _EVIDENCE_ID_RE.fullmatch(tok))
        artifacts = tuple(_extract_backticked(cells[2]))
        rows.append(AppendixA2Row(evidence_ids=ids, artifact_paths=artifacts))
    return rows


def test_paper_evidence_ids_exist_in_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    contract_path = root / "MoM_evidence_contract.md"
    paper_path = root / "paper" / "MoM_paper.md"

    contract_text = contract_path.read_text(encoding="utf-8")
    paper_text = paper_path.read_text(encoding="utf-8")

    contract_ids = _extract_contract_ids(contract_text)
    paper_ids = _extract_paper_cited_ids(paper_text)
    appendix_rows = _iter_appendix_a2_rows(_extract_appendix_a2_section(paper_text))
    appendix_ids = {evidence_id for row in appendix_rows for evidence_id in row.evidence_ids}

    assert contract_ids, "No Evidence IDs found in contract"
    assert paper_ids or appendix_ids, "No Evidence IDs found in inline tags or Appendix A.2"

    for row in appendix_rows:
        assert row.evidence_ids, "Appendix A.2 row is missing Evidence ID(s)"
        assert row.artifact_paths, "Appendix A.2 row is missing artifact path(s)"
        for artifact_path in row.artifact_paths:
            assert (root / artifact_path).exists(), f"Appendix A.2 artifact path is missing: {artifact_path}"

    unknown = sorted((paper_ids | appendix_ids) - contract_ids)
    assert not unknown, f"Unknown Evidence IDs cited in paper: {unknown}"
