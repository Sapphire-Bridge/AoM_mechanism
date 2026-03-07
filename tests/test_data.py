from pathlib import Path

from aom.data.loaders import load_coherence_items, load_counterfactual_pairs, load_disamb_pairs


BASE = Path(__file__).resolve().parents[1]


def test_load_disamb_pairs_smoke():
    items = load_disamb_pairs(str(BASE / "data" / "disamb_pairs.jsonl"))
    assert len(items) >= 1
    assert items[0].pair_id


def test_load_counterfactual_pairs_smoke():
    items = load_counterfactual_pairs(str(BASE / "data" / "counterfactual.jsonl"))
    assert len(items) >= 1
    assert items[0].intervention_type


def test_load_coherence_items_smoke():
    items = load_coherence_items(str(BASE / "data" / "coherence.jsonl"))
    assert len(items) >= 1
    assert items[0].constraint_type
