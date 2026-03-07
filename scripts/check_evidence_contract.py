from __future__ import annotations

import argparse
import re
import sys
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
    paper_role: str


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
        rows.append(AppendixA2Row(evidence_ids=ids, artifact_paths=artifacts, paper_role=cells[3]))
    return rows


def _validate_appendix_a2_rows(rows: list[AppendixA2Row], contract_ids: set[str], repo_root: Path) -> list[str]:
    problems: list[str] = []
    for idx, row in enumerate(rows, start=1):
        if not row.evidence_ids:
            problems.append(f"A.2 row {idx} is missing Evidence ID(s)")
            continue
        unknown = sorted(set(row.evidence_ids) - contract_ids)
        if unknown:
            problems.append(f"A.2 row {idx} cites unknown Evidence ID(s): {', '.join(unknown)}")
        if not row.artifact_paths:
            problems.append(f"A.2 row {idx} is missing artifact path(s)")
            continue
        missing_paths = [path for path in row.artifact_paths if not (repo_root / path).exists()]
        if missing_paths:
            problems.append(f"A.2 row {idx} has missing artifact path(s): {', '.join(missing_paths)}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check manuscript evidence tags vs evidence contract IDs.")
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("MoM_evidence_contract.md"),
        help="Path to evidence contract markdown (default: MoM_evidence_contract.md).",
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path("paper/MoM_paper.md"),
        help="Path to paper markdown (default: paper/MoM_paper.md).",
    )
    args = parser.parse_args(argv)

    contract_text = args.contract.read_text(encoding="utf-8")
    paper_text = args.paper.read_text(encoding="utf-8")

    contract_ids = _extract_contract_ids(contract_text)
    paper_ids = _extract_paper_cited_ids(paper_text)
    appendix_section = _extract_appendix_a2_section(paper_text)
    appendix_rows = _iter_appendix_a2_rows(appendix_section)
    appendix_ids = {evidence_id for row in appendix_rows for evidence_id in row.evidence_ids}

    if not contract_ids:
        print(f"[error] No Evidence IDs found in contract: {args.contract}", file=sys.stderr)
        return 2

    repo_root = args.contract.resolve().parent
    appendix_problems = _validate_appendix_a2_rows(appendix_rows, contract_ids, repo_root)
    if appendix_problems:
        for problem in appendix_problems:
            print(f"[error] {problem}", file=sys.stderr)
        return 2

    cited_ids = paper_ids | appendix_ids
    if not cited_ids:
        print(
            f"[error] No Evidence IDs found in inline tags or Appendix A.2 crosswalk: {args.paper}",
            file=sys.stderr,
        )
        return 2

    unknown = sorted(cited_ids - contract_ids)
    if unknown:
        print(f"[error] Unknown Evidence IDs cited in paper ({len(unknown)}): {', '.join(unknown)}", file=sys.stderr)
        return 2

    unused = sorted(contract_ids - cited_ids)
    print(
        "[ok] Contract IDs: "
        f"{len(contract_ids)}; inline cited: {len(paper_ids)}; A.2 cited: {len(appendix_ids)}; "
        f"unused IDs: {len(unused)}"
    )
    if unused:
        print(f"[warn] Unused contract IDs (may be fine): {', '.join(unused)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
