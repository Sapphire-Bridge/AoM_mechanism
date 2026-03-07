from __future__ import annotations

from pathlib import Path

from scripts.check_evidence_contract import (
    AppendixA2Row,
    _extract_appendix_a2_section,
    _iter_appendix_a2_rows,
    _validate_appendix_a2_rows,
)


def test_iter_appendix_a2_rows_parses_valid_table() -> None:
    paper_text = """
### A.2 Claim-to-artifact map

| Evidence ID(s) | Claim | Artifact(s) | Paper role | Notes |
| --- | --- | --- | --- | --- |
| `E1`, `R1` | Demo claim | `results/demo.csv` | Main text | ok |
| `E2` | Another claim | `results/demo.json` | Appendix | ok |

### A.3 Next section
"""

    rows = _iter_appendix_a2_rows(_extract_appendix_a2_section(paper_text))

    assert rows == [
        AppendixA2Row(
            evidence_ids=("E1", "R1"),
            artifact_paths=("results/demo.csv",),
            paper_role="Main text",
        ),
        AppendixA2Row(
            evidence_ids=("E2",),
            artifact_paths=("results/demo.json",),
            paper_role="Appendix",
        ),
    ]


def test_validate_appendix_a2_rows_flags_missing_evidence_ids(tmp_path: Path) -> None:
    artifact = tmp_path / "results" / "demo.csv"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("ok\n", encoding="utf-8")

    problems = _validate_appendix_a2_rows(
        [AppendixA2Row(evidence_ids=(), artifact_paths=("results/demo.csv",), paper_role="Main text")],
        {"E1"},
        tmp_path,
    )

    assert problems == ["A.2 row 1 is missing Evidence ID(s)"]


def test_validate_appendix_a2_rows_flags_unknown_ids_and_missing_artifacts(tmp_path: Path) -> None:
    problems = _validate_appendix_a2_rows(
        [AppendixA2Row(evidence_ids=("E9",), artifact_paths=("results/missing.csv",), paper_role="Main text")],
        {"E1", "E2"},
        tmp_path,
    )

    assert problems == [
        "A.2 row 1 cites unknown Evidence ID(s): E9",
        "A.2 row 1 has missing artifact path(s): results/missing.csv",
    ]


def test_validate_appendix_a2_rows_flags_missing_artifact_cell(tmp_path: Path) -> None:
    problems = _validate_appendix_a2_rows(
        [AppendixA2Row(evidence_ids=("E1",), artifact_paths=(), paper_role="Appendix")],
        {"E1"},
        tmp_path,
    )

    assert problems == ["A.2 row 1 is missing artifact path(s)"]
