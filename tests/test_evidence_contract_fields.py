from __future__ import annotations

from pathlib import Path

from scripts.check_evidence_contract_fields import check_contract_field_lists


def test_contract_fields_exist_in_listed_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    contract_path = root / "MoM_evidence_contract.md"

    problems = check_contract_field_lists(contract_path=contract_path, repo_root=root)
    assert not problems, f"Contract fields missing from artifacts: {problems}"
