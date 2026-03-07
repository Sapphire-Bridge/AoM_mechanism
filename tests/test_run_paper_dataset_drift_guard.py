from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_paper import ensure_paper_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_ensure_paper_dataset_refuses_flag_mismatch_in_existing_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "bundle"
    data_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        data_dir / "disamb_pairs.jsonl",
        [
            {
                "pair_id": "p0",
                "target": "bank",
                "target_occurrence": 0,
                "a": {"prompt": "I sat by the bank and watched the", "expected_label": "river"},
                "b": {"prompt": "I went to the bank to discuss the", "expected_label": "loan"},
                "choices": {"river": [" river"], "loan": [" loan"]},
            }
        ],
    )
    _write_jsonl(
        data_dir / "counterfactual.jsonl",
        [
            {
                "item_id": "cf-0",
                "base": {"prompt": "Alice went to the bank and", "expected_label": "river"},
                "cf": {"prompt": "Alice went to the bank and", "expected_label": "loan"},
                "choices": {"river": [" river"], "loan": [" loan"]},
                "intervention_type": "test",
                "expected_effect": "shift",
            }
        ],
    )
    _write_jsonl(
        data_dir / "coherence.jsonl",
        [
            {
                "item_id": "coh-0__main",
                "context": "Alice left the house. After an hour, Alice arrived at the",
                "valid_continuations": [" store."],
                "invalid_continuations": [" house."],
                "constraint_type": "location",
                "group": "main",
                "metadata": {"n_constraints": 1},
            }
        ],
    )

    # Create an existing manifest that declares graded items are included.
    (data_dir / "DATASET_MANIFEST.json").write_text(
        json.dumps(
            {
                "name": "paper_hardened_custom",
                "seed": 0,
                "disamb_mode": "hardened",
                "cf_include_shams": True,
                "cf_include_graded": True,
                "n_coh": 80,
                "coh_include_controls": True,
                "files": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different dataset/manifest configuration"):
        ensure_paper_dataset(
            data_dir=data_dir,
            seed=0,
            disamb_mode="hardened",
            cf_include_shams=True,
            cf_include_graded=False,  # mismatch
            n_coh=80,
            coh_include_controls=True,
            dry_run=False,
        )

