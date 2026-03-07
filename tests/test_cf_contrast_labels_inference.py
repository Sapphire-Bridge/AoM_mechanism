from __future__ import annotations

import json
from pathlib import Path

import pytest

from aom.data.loaders import load_counterfactual_pairs
from aom.data.validate import validate_counterfactual_pairs


def test_shift_item_missing_contrast_labels_is_inferred(tmp_path: Path) -> None:
    cf_path = tmp_path / "counterfactual.jsonl"
    row = {
        "item_id": "cf-0",
        "base": {"prompt": "Alice went to the bank and", "expected_label": "river"},
        "cf": {"prompt": "Alice went to the bank and", "expected_label": "loan"},
        "choices": {"river": [" river"], "loan": [" loan"]},
        "intervention_type": "test",
        "expected_effect": "shift",
        # contrast_labels intentionally omitted
    }
    cf_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    items = load_counterfactual_pairs(str(cf_path), validate=False)
    assert len(items) == 1
    assert items[0].contrast_labels == ("river", "loan")
    validate_counterfactual_pairs(items)


def test_shift_item_requires_label_flip(tmp_path: Path) -> None:
    cf_path = tmp_path / "counterfactual.jsonl"
    row = {
        "item_id": "cf-1",
        "base": {"prompt": "Alice went to the bank and", "expected_label": "river"},
        "cf": {"prompt": "Alice went to the bank and", "expected_label": "river"},
        "choices": {"river": [" river"], "loan": [" loan"]},
        "intervention_type": "test",
        "expected_effect": "shift",
    }
    cf_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    items = load_counterfactual_pairs(str(cf_path), validate=False)
    with pytest.raises(ValueError, match="shift item must satisfy base.expected_label!=cf.expected_label"):
        validate_counterfactual_pairs(items)

